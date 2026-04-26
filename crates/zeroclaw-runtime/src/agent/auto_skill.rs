//! Auto-skill creation + improvement hook.
//!
//! Post-turn fire-and-forget hook that asks the LLM, given a single
//! conversation turn, whether to:
//!
//!   - SAVE   a brand-new reusable workflow as a fresh skill
//!   - UPDATE an existing skill that just misbehaved or could be improved
//!   - NONE   leave the skill catalog alone (most turns)
//!
//! The LLM sees the current skill catalogue (name + description) so it can
//! choose between SAVE and UPDATE intelligently. UPDATE overwrites the
//! existing SKILL.md (with version bump) and any helper files specified.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::fs;
use zeroclaw_api::channel::{Channel, SendMessage};
use zeroclaw_api::provider::Provider;

const SYSTEM_PROMPT: &str = "\
You are a skill-maintenance system. Review this conversation turn and decide whether to:

  SAVE   — save a brand-new reusable workflow as a fresh skill
  UPDATE — overwrite an existing skill that just FAILED, returned wrong output, or
           that you can clearly improve based on what just happened
  NONE   — leave the skill catalog alone (most turns)

GUIDELINES

Save NEW (SAVE) only when ALL hold:
- Turn involved a procedure (≥2 actions, API calls, or transformations) that could be re-applied to a similar task later.
- No existing skill in the catalogue already covers it.
- Outcome was successful.

Examples that SHOULD be saved as a NEW skill:
- Fetching BTC price from Binance (2 API calls + parsing) → fetch_btc_price
- Checking Laravel Horizon health (supervisor + redis + psql + horizon:failed) → laravel_horizon_health_check

Update an existing skill (UPDATE) when at least one holds:
- An existing skill was invoked this turn and returned an error / non-zero exit / clearly wrong output.
- The user complained about the result.
- The agent worked around an existing skill with inline commands because the skill was incomplete or wrong.
- A schema/API the skill depends on changed and the agent had to compensate.
- You found a strictly better implementation than what the skill currently has.

Do NOT update a skill that worked correctly. Do NOT save a skill that already exists by another name.

If saving (SAVE), output EXACTLY:

SAVE
name: lowercase_snake_case_max_40_chars
description: <≤120 chars, concrete, mentions when to use it>
body:
<markdown body — instructions for invoking. Reference any FILE you emit by name. Use ${SKILL_DIR} as a placeholder for the skill directory.>

(Optional FILE blocks — see below.)

If improving an existing skill (UPDATE), output EXACTLY:

UPDATE
name: <MUST match an existing skill name exactly, lowercase_snake_case>
description: <updated ≤120-char description>
body:
<full new markdown body — overwrites the old SKILL.md body>

(Optional FILE blocks — overwrite or add helper files.)

FILE blocks (used by both SAVE and UPDATE):

--- BEGIN FILE: <relative_filename_no_subdirs> ---
<file content, exactly as it should land on disk>
--- END FILE ---

(Filenames must be plain (e.g. `fetch.py`, `parse.sh`); no slashes, no leading dots, no parent paths. Allowed extensions: .py .sh .txt .md .json .yaml .toml.)

If neither saving nor updating, output EXACTLY: NONE";

/// Evaluate the turn and either save a new skill, improve an existing one, or
/// do nothing. Best-effort: every failure is debug-logged and otherwise
/// swallowed, never affecting the user-facing reply.
#[allow(clippy::too_many_arguments)]
pub async fn evaluate_and_save(
    provider: Arc<dyn Provider>,
    model: String,
    user_msg: String,
    assistant_reply: String,
    sender: String,
    workspace_dir: PathBuf,
    notify_channel: Option<Arc<dyn Channel>>,
    notify_target: Option<String>,
) {
    if user_msg.trim().is_empty() || assistant_reply.trim().is_empty() {
        return;
    }

    // Build a compact list of existing skills (name + description) so the LLM
    // can decide between SAVE and UPDATE.
    let existing_catalogue = list_existing_skills(&workspace_dir).await;
    let catalogue_block = if existing_catalogue.is_empty() {
        "EXISTING SKILLS: (none)".to_string()
    } else {
        let lines: Vec<String> = existing_catalogue
            .iter()
            .map(|(n, d)| format!("- {n}: {d}"))
            .collect();
        format!("EXISTING SKILLS:\n{}", lines.join("\n"))
    };

    let u = truncate_chars(&user_msg, 1500);
    let a = truncate_chars(&assistant_reply, 4000);
    let user_prompt = format!(
        "{catalogue_block}\n\nUSER MESSAGE:\n{u}\n\nASSISTANT REPLY:\n{a}\n\nDecide:"
    );

    let raw = match tokio::time::timeout(
        Duration::from_secs(45),
        provider.chat_with_system(Some(SYSTEM_PROMPT), &user_prompt, &model, 0.1),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::debug!(error = %e, sender = %sender, "auto-skill: LLM call failed");
            return;
        }
        Err(_) => {
            tracing::debug!(sender = %sender, "auto-skill: timed out");
            return;
        }
    };

    let trimmed = raw.trim();
    tracing::info!(
        sender = %sender,
        raw_preview = %truncate_chars(trimmed, 200),
        "auto-skill: LLM responded"
    );

    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NONE") {
        return;
    }

    let parsed = match parse_skill_response(trimmed) {
        Some(p) => p,
        None => {
            tracing::debug!(sender = %sender, "auto-skill: unparseable response");
            return;
        }
    };

    if !is_valid_skill_name(&parsed.name) || parsed.body.trim().is_empty() {
        tracing::debug!(sender = %sender, name = %parsed.name, "auto-skill: rejected (invalid name or empty body)");
        return;
    }
    for f in &parsed.files {
        if !is_safe_filename(&f.name) {
            tracing::debug!(sender = %sender, file = %f.name, "auto-skill: rejected (unsafe file name)");
            return;
        }
    }

    let skill_dir = workspace_dir.join("skills").join(&parsed.name);
    let dir_exists = skill_dir.exists();

    match parsed.action {
        SkillAction::Save => {
            if dir_exists {
                tracing::debug!(
                    sender = %sender,
                    name = %parsed.name,
                    "auto-skill: SAVE skipped — skill already exists (would conflict)"
                );
                return;
            }
            if let Err(e) = fs::create_dir_all(&skill_dir).await {
                tracing::debug!(error = %e, sender = %sender, "auto-skill: mkdir failed");
                return;
            }
            write_skill_files(&skill_dir, &parsed, &sender, "0.1.0").await;
            tracing::info!(
                sender = %sender,
                name = %parsed.name,
                files = ?parsed.files.iter().map(|f| &f.name).collect::<Vec<_>>(),
                "auto-skill: wrote new skill"
            );
            notify(notify_channel, notify_target, &format!(
                "\u{1F4BE} Skill `{name}` created{extra}.\n_{desc}_",
                name = parsed.name,
                desc = parsed.description,
                extra = file_count_suffix(&parsed.files),
            )).await;
        }
        SkillAction::Update => {
            if !dir_exists {
                tracing::debug!(
                    sender = %sender,
                    name = %parsed.name,
                    "auto-skill: UPDATE skipped — target skill does not exist"
                );
                return;
            }
            // Bump version: read old SKILL.md, parse `version: X.Y.Z`, +1 patch.
            let old_md_path = skill_dir.join("SKILL.md");
            let next_version = match fs::read_to_string(&old_md_path).await {
                Ok(content) => bump_version(&content),
                Err(_) => "0.1.1".to_string(),
            };
            write_skill_files(&skill_dir, &parsed, &sender, &next_version).await;
            tracing::info!(
                sender = %sender,
                name = %parsed.name,
                version = %next_version,
                files = ?parsed.files.iter().map(|f| &f.name).collect::<Vec<_>>(),
                "auto-skill: updated existing skill"
            );
            notify(notify_channel, notify_target, &format!(
                "\u{1F527} Skill `{name}` updated to v{version}{extra}.\n_{desc}_",
                name = parsed.name,
                version = next_version,
                desc = parsed.description,
                extra = file_count_suffix(&parsed.files),
            )).await;
        }
    }
}

/// Find existing skills under `workspace_dir/skills/<name>/SKILL.md` and read
/// their name+description. Cheap (~few file reads); only the description line
/// is parsed. Best-effort: failures yield an empty list.
async fn list_existing_skills(workspace_dir: &PathBuf) -> Vec<(String, String)> {
    let skills_dir = workspace_dir.join("skills");
    let mut out = Vec::new();
    let mut rd = match fs::read_dir(&skills_dir).await {
        Ok(r) => r,
        Err(_) => return out,
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let md = path.join("SKILL.md");
        let Ok(content) = fs::read_to_string(&md).await else {
            continue;
        };
        let name = parse_frontmatter_field(&content, "name")
            .or_else(|| {
                path.file_name()
                    .and_then(|n| n.to_str().map(ToString::to_string))
            })
            .unwrap_or_default();
        let desc = parse_frontmatter_field(&content, "description").unwrap_or_default();
        if !name.is_empty() {
            out.push((name, desc));
        }
    }
    out
}

fn parse_frontmatter_field(md: &str, field: &str) -> Option<String> {
    // Crude but sufficient: scan first 30 lines for `<field>: ...` after `---`.
    let mut in_fm = false;
    for (i, line) in md.lines().take(30).enumerate() {
        if i == 0 && line.trim() == "---" {
            in_fm = true;
            continue;
        }
        if in_fm && line.trim() == "---" {
            break;
        }
        if !in_fm {
            continue;
        }
        let lower = line.trim_start().to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix(&format!("{field}:")) {
            let _ = rest;
            // Take the substring after the first ':' in the original (to preserve case).
            if let Some(idx) = line.find(':') {
                let value = line[idx + 1..].trim().trim_matches('"');
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Bump the `version: X.Y.Z` in SKILL.md frontmatter by incrementing the
/// patch number. Returns the new version string. Defaults to `0.1.1` if the
/// existing version is missing or malformed.
fn bump_version(md: &str) -> String {
    let current = parse_frontmatter_field(md, "version").unwrap_or_else(|| "0.1.0".to_string());
    let parts: Vec<&str> = current.split('.').collect();
    if parts.len() != 3 {
        return "0.1.1".to_string();
    }
    let (Ok(major), Ok(minor), Ok(patch)) = (
        parts[0].parse::<u32>(),
        parts[1].parse::<u32>(),
        parts[2].parse::<u32>(),
    ) else {
        return "0.1.1".to_string();
    };
    format!("{major}.{minor}.{}", patch + 1)
}

async fn write_skill_files(
    skill_dir: &std::path::Path,
    parsed: &ParsedSkill,
    sender: &str,
    version: &str,
) {
    let tag = format!("[auto, {}]", sanitize_tag(sender));
    let frontmatter = format!(
        "---\nname: {}\ndescription: {}\nversion: {}\nauthor: auto-skill\ntags: {}\n---\n\n",
        parsed.name,
        escape_yaml_value(&parsed.description),
        version,
        tag,
    );
    let skill_md = skill_dir.join("SKILL.md");
    let _ = fs::write(&skill_md, format!("{frontmatter}{}\n", parsed.body.trim())).await;

    for f in &parsed.files {
        let path = skill_dir.join(&f.name);
        if let Err(e) = fs::write(&path, f.content.as_bytes()).await {
            tracing::debug!(error = %e, path = %path.display(), "auto-skill: file write failed");
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if f.name.ends_with(".sh") {
                if let Ok(meta) = std::fs::metadata(&path) {
                    let mut perms = meta.permissions();
                    perms.set_mode(0o755);
                    let _ = std::fs::set_permissions(&path, perms);
                }
            }
        }
    }
}

fn file_count_suffix(files: &[ParsedFile]) -> String {
    if files.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
        format!(" ({} file(s): {})", files.len(), names.join(", "))
    }
}

async fn notify(channel: Option<Arc<dyn Channel>>, target: Option<String>, body: &str) {
    if let (Some(ch), Some(t)) = (channel, target) {
        let _ = ch.send(&SendMessage::new(body, t.as_str())).await;
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SkillAction {
    Save,
    Update,
}

#[derive(Debug)]
struct ParsedSkill {
    action: SkillAction,
    name: String,
    description: String,
    body: String,
    files: Vec<ParsedFile>,
}

#[derive(Debug)]
struct ParsedFile {
    name: String,
    content: String,
}

/// Parse the LLM response. Accepts a leading `SAVE` or `UPDATE` line, then
/// `name:`/`description:`/`body:` followed by the body and optional FILE
/// blocks.
fn parse_skill_response(raw: &str) -> Option<ParsedSkill> {
    let mut lines = raw.lines();
    let header = lines.next()?.trim();
    let action = if header.eq_ignore_ascii_case("SAVE") {
        SkillAction::Save
    } else if header.eq_ignore_ascii_case("UPDATE") {
        SkillAction::Update
    } else {
        return None;
    };

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();
    let mut files: Vec<ParsedFile> = Vec::new();
    let mut state = HeaderOrBody::Header;
    let mut current_file: Option<(String, Vec<&str>)> = None;

    enum HeaderOrBody {
        Header,
        Body,
    }

    for line in lines {
        match state {
            HeaderOrBody::Header => {
                let t = line.trim_start();
                if let Some(rest) = strip_ci_prefix(t, "name:") {
                    let _ = rest;
                    name = line
                        .find(':')
                        .map(|i| line[i + 1..].trim().to_string());
                } else if strip_ci_prefix(t, "description:").is_some() {
                    description = line
                        .find(':')
                        .map(|i| line[i + 1..].trim().to_string());
                } else if t.to_ascii_lowercase().starts_with("body:") {
                    state = HeaderOrBody::Body;
                }
            }
            HeaderOrBody::Body => {
                let trimmed = line.trim();
                if let Some(fname) = strip_file_open(trimmed) {
                    current_file = Some((fname, Vec::new()));
                    continue;
                }
                if trimmed == "--- END FILE ---" || trimmed == "--- END ---" {
                    if let Some((fname, fl)) = current_file.take() {
                        files.push(ParsedFile {
                            name: fname,
                            content: fl.join("\n") + "\n",
                        });
                    }
                    continue;
                }
                if let Some((_, ref mut fl)) = current_file {
                    fl.push(line);
                } else {
                    body_lines.push(line);
                }
            }
        }
    }

    Some(ParsedSkill {
        action,
        name: name?.trim().to_string(),
        description: description?.trim().to_string(),
        body: body_lines.join("\n").trim().to_string(),
        files,
    })
}

fn strip_file_open(line: &str) -> Option<String> {
    let t = line.trim();
    let prefix = "--- BEGIN FILE:";
    let suffix = "---";
    if !t.starts_with(prefix) || !t.ends_with(suffix) || t.len() <= prefix.len() + suffix.len() {
        return None;
    }
    let inner = t[prefix.len()..t.len() - suffix.len()].trim();
    if inner.is_empty() {
        return None;
    }
    Some(inner.to_string())
}

fn strip_ci_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() < prefix.len() {
        return None;
    }
    let head = &s[..prefix.len()];
    if head.eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

fn is_valid_skill_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 40
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !s.starts_with('_')
        && !s.ends_with('_')
}

fn is_safe_filename(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    if name.starts_with('.') || name.contains('/') || name.contains('\\') || name.contains("..") {
        return false;
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return false;
    }
    let allowed_extensions = [".py", ".sh", ".txt", ".md", ".json", ".yaml", ".toml"];
    allowed_extensions.iter().any(|ext| name.ends_with(ext))
}

fn sanitize_tag(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn escape_yaml_value(s: &str) -> String {
    if s.contains(':') || s.contains('"') || s.contains('\n') {
        format!(
            "\"{}\"",
            s.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', " ")
        )
    } else {
        s.to_string()
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = s[..end].to_string();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_save() {
        let raw = "SAVE\nname: foo\ndescription: do foo\nbody:\nrun foo";
        let p = parse_skill_response(raw).expect("parse");
        assert_eq!(p.action, SkillAction::Save);
        assert_eq!(p.name, "foo");
    }

    #[test]
    fn parses_update_with_file() {
        let raw = "UPDATE\nname: btc\ndescription: fixed\nbody:\nbash $SKILL_DIR/parse.sh\n--- BEGIN FILE: parse.sh ---\necho fixed\n--- END FILE ---";
        let p = parse_skill_response(raw).expect("parse");
        assert_eq!(p.action, SkillAction::Update);
        assert_eq!(p.files.len(), 1);
        assert_eq!(p.files[0].name, "parse.sh");
    }

    #[test]
    fn rejects_unknown_action() {
        assert!(parse_skill_response("DELETE\nname: foo").is_none());
    }

    #[test]
    fn bump_version_works() {
        assert_eq!(
            bump_version("---\nname: x\nversion: 0.1.7\n---\nbody"),
            "0.1.8"
        );
        assert_eq!(bump_version("---\nname: x\nversion: 1.2.3\n---\n"), "1.2.4");
        assert_eq!(bump_version("---\nname: x\n---\n"), "0.1.1");
        assert_eq!(bump_version("---\nname: x\nversion: bogus\n---\n"), "0.1.1");
    }
}

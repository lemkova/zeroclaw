//! Auto-skill creation hook (v0.6: multi-file + channel notification).
//!
//! Post-turn fire-and-forget hook that asks the LLM whether the just-completed
//! turn produced a reusable workflow worth saving, and if so writes the
//! resulting `SKILL.md` (plus any code/script files the LLM emitted alongside
//! it) under `~/.zeroclaw/workspace/skills/<name>/`.
//!
//! v0.6 additions vs v0.5:
//! 1. Multi-file skills — the LLM may emit `--- BEGIN FILE: <name> --- ...
//!    --- END FILE ---` blocks alongside the markdown body. This lets the
//!    agent write Python/shell helpers when the procedure is too complex for
//!    inline shell commands (auth headers, request signing, response parsing).
//! 2. Channel notification — when a `Channel` reference is provided, emit a
//!    single `💾 Skill 'name' created` message after a successful write so
//!    the operator gets immediate feedback (Hermes-style).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::fs;
use zeroclaw_api::channel::{Channel, SendMessage};
use zeroclaw_api::provider::Provider;

const SYSTEM_PROMPT: &str = "\
You are a workflow distillation system. Decide whether a conversation turn produced a reusable PROCEDURAL workflow worth saving as a skill — a recipe the agent can re-run for a similar task.

NOTE: You are NOT distilling user facts (a separate system handles that). Your only job is procedural workflows.

SAVE when the turn involves a sequence of API calls, shell commands, or transformations that could be replayed for a similar request.

Examples that SHOULD be saved (worth a skill):
- Fetching BTC price + 24h change from Binance (2 API calls + parsing) → fetch_btc_price
- Checking Laravel Horizon health (supervisorctl + redis-cli + psql + horizon:failed) → laravel_horizon_health_check
- Querying GitHub commits for a repo with auth (curl + parse JSON) → github_recent_commits

Examples that should NOT be saved:
- Single shell command (`ls`, `df -h`, `pwd`)
- Pure Q&A or explanation with no commands
- Failed attempts, debugging, exploratory commands
- One-off requests with no replayable structure
- Already-existing skill (we will skip duplicate names automatically)

If the procedure needs more than a one-liner — request signing, HMAC/OAuth, multi-step JSON transforms, retries — emit one or more code FILES alongside the body so future runs can `python ${SKILL_DIR}/<file>.py` (or `bash ${SKILL_DIR}/<file>.sh`) instead of inlining fragile commands.

If saving, output EXACTLY this format and nothing else:

SAVE
name: lowercase_snake_case_max_40_chars
description: <≤120 chars, concrete, mentions when to use it>
body:
<markdown body — the recipe / how to invoke. Reference any FILE you emit by name. Use ${SKILL_DIR} as a placeholder for the skill's own directory.>

--- BEGIN FILE: <relative_filename_no_subdirs> ---
<file content, exactly as it should land on disk>
--- END FILE ---

(Repeat the BEGIN/END FILE block for each additional file. Filenames must be plain (e.g. `fetch.py`, `parse.sh`); no slashes, no leading dots, no parent paths. Omit FILE blocks entirely for shell-only skills.)

If not saving, output EXACTLY: NONE";

/// Evaluate the turn for skill-worthiness; if worth saving, write SKILL.md
/// (plus any code files the LLM emitted) and optionally post a creation
/// notification to a channel. Best-effort: every failure is debug-logged
/// and otherwise swallowed.
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
    let u = truncate_chars(&user_msg, 1500);
    let a = truncate_chars(&assistant_reply, 4000);
    let user_prompt = format!("USER MESSAGE:\n{u}\n\nASSISTANT REPLY:\n{a}\n\nDecide:");

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
    let Some(parsed) = parse_skill_response(trimmed) else {
        tracing::debug!(
            sender = %sender,
            raw = %truncate_chars(trimmed, 200),
            "auto-skill: unparseable response"
        );
        return;
    };

    if !is_valid_skill_name(&parsed.name) || parsed.body.trim().is_empty() {
        tracing::debug!(
            sender = %sender,
            name = %parsed.name,
            "auto-skill: rejected (invalid name or empty body)"
        );
        return;
    }
    for f in &parsed.files {
        if !is_safe_filename(&f.name) {
            tracing::debug!(
                sender = %sender,
                name = %parsed.name,
                file = %f.name,
                "auto-skill: rejected (unsafe file name)"
            );
            return;
        }
    }

    let skill_dir = workspace_dir.join("skills").join(&parsed.name);
    if skill_dir.exists() {
        tracing::debug!(
            sender = %sender,
            name = %parsed.name,
            "auto-skill: skill already exists, skipping"
        );
        return;
    }

    if let Err(e) = fs::create_dir_all(&skill_dir).await {
        tracing::debug!(
            error = %e,
            sender = %sender,
            name = %parsed.name,
            "auto-skill: mkdir failed"
        );
        return;
    }

    let tag = format!("[auto, {}]", sanitize_tag(&sender));
    let frontmatter = format!(
        "---\nname: {}\ndescription: {}\nversion: 0.1.0\nauthor: auto-skill\ntags: {}\n---\n\n",
        parsed.name,
        escape_yaml_value(&parsed.description),
        tag,
    );
    let skill_md = skill_dir.join("SKILL.md");
    if let Err(e) = fs::write(&skill_md, format!("{frontmatter}{}\n", parsed.body.trim())).await {
        tracing::debug!(
            error = %e,
            path = %skill_md.display(),
            "auto-skill: SKILL.md write failed"
        );
        return;
    }

    let mut written_files = Vec::with_capacity(parsed.files.len());
    for f in &parsed.files {
        let path = skill_dir.join(&f.name);
        if let Err(e) = fs::write(&path, f.content.as_bytes()).await {
            tracing::debug!(
                error = %e,
                path = %path.display(),
                "auto-skill: code file write failed"
            );
            continue;
        }
        // Mark shell scripts executable (Python is invoked via `python …`).
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
        written_files.push(f.name.clone());
    }

    tracing::info!(
        sender = %sender,
        name = %parsed.name,
        files = ?written_files,
        path = %skill_md.display(),
        "auto-skill: wrote new skill"
    );

    // Channel notification (Hermes-style "💾 Skill '<name>' created").
    if let (Some(ch), Some(target)) = (notify_channel.as_ref(), notify_target.as_ref()) {
        let extra = if written_files.is_empty() {
            String::new()
        } else {
            format!(" ({} file(s): {})", written_files.len(), written_files.join(", "))
        };
        let body = format!(
            "\u{1F4BE} Skill `{}` created{}.\n_{}_",
            parsed.name, extra, parsed.description
        );
        let _ = ch.send(&SendMessage::new(&body, target.as_str())).await;
    }
}

#[derive(Debug)]
struct ParsedSkill {
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

/// Parse the LLM response into name/description/body and any FILE blocks.
fn parse_skill_response(raw: &str) -> Option<ParsedSkill> {
    let mut lines = raw.lines();
    if !lines.next()?.trim().eq_ignore_ascii_case("SAVE") {
        return None;
    }

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();
    let mut files: Vec<ParsedFile> = Vec::new();
    let mut state = State::Header;
    let mut current_file: Option<(String, Vec<&str>)> = None;

    enum State {
        Header,
        Body,
    }

    for line in lines {
        match state {
            State::Header => {
                let t = line.trim_start();
                if let Some(rest) = strip_ci_prefix(t, "name:") {
                    name = Some(rest.trim().to_string());
                } else if let Some(rest) = strip_ci_prefix(t, "description:") {
                    description = Some(rest.trim().to_string());
                } else if t.eq_ignore_ascii_case("body:") || t.to_ascii_lowercase().starts_with("body:") {
                    state = State::Body;
                }
            }
            State::Body => {
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
        name: name?.trim().to_string(),
        description: description?.trim().to_string(),
        body: body_lines.join("\n").trim().to_string(),
        files,
    })
}

/// Match `--- BEGIN FILE: foo.py ---` (allowing minor whitespace variation).
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

/// Allow only plain filenames, no path components, no leading dots.
/// Restricts to safe Python/shell/text extensions.
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
    fn parses_simple_save() {
        let raw = "SAVE\nname: foo_bar\ndescription: do foo\nbody:\nrun foo";
        let p = parse_skill_response(raw).expect("parse");
        assert_eq!(p.name, "foo_bar");
        assert_eq!(p.description, "do foo");
        assert!(p.files.is_empty());
    }

    #[test]
    fn parses_multi_file() {
        let raw = "SAVE\n\
                   name: btc_holders\n\
                   description: fetch holders\n\
                   body:\n\
                   Run: shell `python ${SKILL_DIR}/fetch.py`\n\
                   --- BEGIN FILE: fetch.py ---\n\
                   import os\n\
                   print(\"hi\")\n\
                   --- END FILE ---";
        let p = parse_skill_response(raw).expect("parse");
        assert_eq!(p.files.len(), 1);
        assert_eq!(p.files[0].name, "fetch.py");
        assert!(p.files[0].content.contains("import os"));
    }

    #[test]
    fn rejects_unsafe_filename() {
        assert!(!is_safe_filename("../etc/passwd"));
        assert!(!is_safe_filename("/etc/passwd"));
        assert!(!is_safe_filename(".hidden.py"));
        assert!(!is_safe_filename("normal.exe"));
        assert!(is_safe_filename("fetch.py"));
        assert!(is_safe_filename("parse_data.py"));
    }
}

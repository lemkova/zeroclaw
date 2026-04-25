//! Auto-skill creation hook.
//!
//! Post-turn fire-and-forget hook that asks the LLM whether the just-completed
//! conversation turn produced a reusable procedural workflow worth saving as a
//! skill (a `SKILL.md` file under `~/.zeroclaw/workspace/skills/<name>/`).
//!
//! This is the procedural counterpart to the auto-dialectic hook, which
//! captures declarative facts. Together they let the agent grow its own
//! capability surface — facts (declarative memory) + workflows (procedural
//! memory) — without the user having to call save tools manually.
//!
//! The LLM is instructed to be conservative: most turns produce nothing
//! (return `NONE`). Only turns that look like multi-step recipes are saved.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::fs;
use zeroclaw_api::provider::Provider;

const SYSTEM_PROMPT: &str = "\
You are a workflow distillation system. Decide whether a conversation turn produced a reusable PROCEDURAL workflow worth saving as a skill — a multi-step recipe the agent could re-run later for a similar task.

Save ONLY when ALL of these hold:
- The turn involved a multi-step procedure (≥3 distinct actions, commands, or API calls)
- The procedure could be re-applied to a *similar* task later (not just this exact one)
- The outcome was successful (not a failed exploration)

DO NOT save:
- Single-shell-command tasks (the `shell` primitive already handles them)
- Q&A or explanations with no procedure
- Failed attempts, partial work, or debugging sessions
- One-off requests with no reusable structure
- Things obviously covered by an existing primitive (file_read, web_search, etc.)

If saving, output EXACTLY this format and nothing else:

SAVE
name: lowercase_snake_case_max_40_chars
description: <≤120 chars, concrete, mentions when to use it>
body:
<markdown body, 100-600 chars, with concrete commands/steps the agent should follow next time>

If not saving, output EXACTLY: NONE";

/// Evaluate the turn for skill-worthiness and write the file if appropriate.
/// Best-effort: any failure is debug-logged and otherwise swallowed.
pub async fn evaluate_and_save(
    provider: Arc<dyn Provider>,
    model: String,
    user_msg: String,
    assistant_reply: String,
    sender: String,
    workspace_dir: PathBuf,
) {
    if user_msg.trim().is_empty() || assistant_reply.trim().is_empty() {
        return;
    }
    let u = truncate_chars(&user_msg, 1500);
    let a = truncate_chars(&assistant_reply, 2500);
    let user_prompt = format!("USER MESSAGE:\n{u}\n\nASSISTANT REPLY:\n{a}\n\nDecide:");

    let raw = match tokio::time::timeout(
        Duration::from_secs(30),
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
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NONE") {
        return;
    }
    let Some(parsed) = parse_skill_response(trimmed) else {
        tracing::debug!(sender = %sender, raw = %truncate_chars(trimmed, 200), "auto-skill: unparseable response");
        return;
    };

    if !is_valid_skill_name(&parsed.name) || parsed.body.trim().is_empty() {
        tracing::debug!(sender = %sender, name = %parsed.name, "auto-skill: rejected (invalid name or empty body)");
        return;
    }

    let skill_dir = workspace_dir.join("skills").join(&parsed.name);
    if skill_dir.exists() {
        // Don't clobber an existing skill — leave the operator to merge or
        // delete first. This keeps the hook idempotent across re-asks.
        tracing::debug!(sender = %sender, name = %parsed.name, "auto-skill: skill already exists, skipping");
        return;
    }

    if let Err(e) = fs::create_dir_all(&skill_dir).await {
        tracing::debug!(error = %e, sender = %sender, name = %parsed.name, "auto-skill: mkdir failed");
        return;
    }

    let content = format!(
        "---\nname: {}\ndescription: {}\nversion: 0.1.0\nauthor: auto-skill\ntags: [auto, {}]\n---\n\n{}\n",
        parsed.name,
        escape_yaml_value(&parsed.description),
        sanitize_tag(&sender),
        parsed.body.trim(),
    );
    let skill_md = skill_dir.join("SKILL.md");
    if let Err(e) = fs::write(&skill_md, &content).await {
        tracing::debug!(error = %e, path = %skill_md.display(), "auto-skill: write failed");
        return;
    }
    tracing::info!(
        sender = %sender,
        name = %parsed.name,
        path = %skill_md.display(),
        "auto-skill: wrote new skill"
    );
}

#[derive(Debug)]
struct ParsedSkill {
    name: String,
    description: String,
    body: String,
}

/// Parse the LLM's response into a structured skill. Format expected:
///   SAVE
///   name: foo
///   description: bar
///   body:
///   <markdown>
fn parse_skill_response(raw: &str) -> Option<ParsedSkill> {
    let mut lines = raw.lines();
    let header = lines.next()?.trim();
    if !header.eq_ignore_ascii_case("SAVE") {
        return None;
    }
    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();
    let mut in_body = false;

    for line in lines {
        if in_body {
            body_lines.push(line);
            continue;
        }
        let lower = line.trim_start().to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("name:") {
            let value = line.trim_start()[lower.find(':')? + 1..].trim();
            let _ = rest; // shut up unused
            name = Some(value.to_string());
        } else if let Some(_) = lower.strip_prefix("description:") {
            let value = line.trim_start()[lower.find(':')? + 1..].trim();
            description = Some(value.to_string());
        } else if lower.starts_with("body:") {
            in_body = true;
        }
    }

    let body = body_lines.join("\n").trim().to_string();
    Some(ParsedSkill {
        name: name?.trim().to_string(),
        description: description?.trim().to_string(),
        body,
    })
}

fn is_valid_skill_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 40
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !s.starts_with('_')
        && !s.ends_with('_')
}

fn sanitize_tag(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn escape_yaml_value(s: &str) -> String {
    // Single-line; escape only colon and quote pitfalls by wrapping in quotes
    // when the value contains markers that would confuse the simple frontmatter
    // parser. Quotes themselves get escaped.
    if s.contains(':') || s.contains('"') || s.contains('\n') {
        format!(
            "\"{}\"",
            s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
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

//! Auto-dialectic memory hook.
//!
//! Distills (user_msg, assistant_reply) pairs into per-user durable Core
//! memory entries using a cheap LLM call. Fired fire-and-forget after each
//! turn so it doesn't block the user-facing reply.
//!
//! Inspired by Hermes/Honcho's approach: build an evolving user model from
//! conversation pairs without requiring the agent to remember to call a
//! "save memory" tool.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;
use zeroclaw_api::provider::Provider;
use zeroclaw_memory::traits::{Memory, MemoryCategory};

const SYSTEM_PROMPT: &str = "\
You are a memory-distillation system. Extract DURABLE FACTS about the user from a conversation turn so they can be recalled later.

Rules:
- Only facts about the USER's stack, projects, preferences, identities, or stable context.
- SKIP transient items (today's task, temporary errors, the assistant's own work, command output, debug noise).
- SKIP facts that would be obvious from a typical system prompt or are restatements of the user's literal request.
- Output 0–5 facts, one per line, in this exact format: `- key: value`
  - `key` is lowercase snake_case (e.g. `current_project`, `preferred_stack`, `deploys_with`).
  - `value` is ≤140 characters, factual, self-contained.
- If nothing durable, output exactly: `NONE`.
Nothing else. No preamble, no explanation.";

/// Distill durable user facts from one turn and store them as Core memory
/// entries scoped to the sender. Best-effort: any failure (provider error,
/// timeout, parse error) is logged at debug level and otherwise swallowed —
/// this hook must never affect the user-facing reply path.
pub async fn distill_and_store(
    provider: Arc<dyn Provider>,
    model: String,
    user_msg: String,
    assistant_reply: String,
    sender: String,
    memory: Arc<dyn Memory>,
) {
    if user_msg.trim().is_empty() || assistant_reply.trim().is_empty() {
        return;
    }
    let u = truncate_chars(&user_msg, 1500);
    let a = truncate_chars(&assistant_reply, 1500);
    let user_prompt = format!("USER MESSAGE:\n{u}\n\nASSISTANT REPLY:\n{a}\n\nFacts:");

    // Hard 30-second timeout — distillation is best-effort.
    let raw = match tokio::time::timeout(
        Duration::from_secs(30),
        provider.chat_with_system(Some(SYSTEM_PROMPT), &user_prompt, &model, 0.1),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::debug!(error = %e, sender = %sender, "auto-dialectic: LLM call failed");
            return;
        }
        Err(_) => {
            tracing::debug!(sender = %sender, "auto-dialectic: timed out");
            return;
        }
    };

    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("NONE")
        || trimmed.lines().next().is_some_and(|l| l.trim().eq_ignore_ascii_case("NONE"))
    {
        return;
    }

    let mut stored = 0usize;
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Tolerate `- key: value`, `* key: value`, or `key: value`.
        let stripped = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .unwrap_or(line);
        let Some((key_raw, value_raw)) = stripped.split_once(':') else {
            continue;
        };
        let key = key_raw
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
            .collect::<String>();
        let value = value_raw.trim();
        if key.is_empty() || value.is_empty() || key.len() > 64 || value.len() > 240 {
            continue;
        }
        // Random suffix prevents collisions across turns; the underlying memory
        // backend handles supersession via the conflict module when entries
        // semantically overlap.
        let memory_key = format!("dialectic_{sender}_{key}_{}", short_id());
        if let Err(e) = memory
            .store(&memory_key, value, MemoryCategory::Core, Some(&sender))
            .await
        {
            tracing::debug!(error = %e, sender = %sender, "auto-dialectic: store failed");
            continue;
        }
        stored += 1;
        if stored >= 5 {
            break;
        }
    }
    if stored > 0 {
        tracing::info!(sender = %sender, count = stored, "auto-dialectic: stored user facts");
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

fn short_id() -> String {
    Uuid::new_v4()
        .to_string()
        .split_once('-')
        .map(|(p, _)| p.to_string())
        .unwrap_or_else(|| "x".into())
}

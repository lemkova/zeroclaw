//! `daemon_restart` tool — re-exec the running zeroclaw daemon so config or
//! provider/MCP changes can take effect without operator assistance.
//!
//! Implementation: spawn a short-delay background task that calls `exec()` to
//! atomically replace the current process image with a fresh invocation of
//! `zeroclaw daemon`. The PID is preserved across the call. The tool returns
//! immediately with a confirmation so the in-flight reply still reaches the
//! user before the process turns over.

use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;
use zeroclaw_api::tool::{Tool, ToolResult};

pub struct DaemonRestartTool;

impl DaemonRestartTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DaemonRestartTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for DaemonRestartTool {
    fn name(&self) -> &str {
        "daemon_restart"
    }

    fn description(&self) -> &str {
        "Re-exec the zeroclaw daemon to pick up config or provider changes. \
         The current process image is replaced atomically; the PID is preserved. \
         Use after editing ~/.zeroclaw/config.toml or registering a new MCP server. \
         Returns immediately with a confirmation; the actual exec fires ~2s later \
         so the in-flight reply still reaches the user."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "delay_secs": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30,
                    "default": 2,
                    "description": "Seconds to wait before exec (lets the reply ship). Default 2."
                },
                "reason": {
                    "type": "string",
                    "description": "Optional human-readable reason recorded in logs."
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let delay = args
            .get("delay_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(2)
            .min(30);
        let reason = args
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("agent-initiated")
            .to_string();

        // Fire-and-forget re-exec on a separate task so this tool can return
        // a confirmation response before the process image turns over.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(delay)).await;
            tracing::warn!(reason = %reason, "daemon_restart: re-execing process");
            let exe = match std::env::current_exe() {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(error = %e, "daemon_restart: current_exe() failed");
                    return;
                }
            };
            // Skip argv[0] (program path); rebuild argv from remaining args so
            // the new process gets the same subcommand + flags as the running one.
            let args: Vec<String> = std::env::args().skip(1).collect();

            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                let err = std::process::Command::new(&exe).args(&args).exec();
                // exec() only returns on failure.
                tracing::error!(error = %err, exe = %exe.display(), "daemon_restart: exec failed");
                std::process::exit(1);
            }
            #[cfg(not(unix))]
            {
                // No exec on Windows; spawn + exit is the closest equivalent.
                let _ = std::process::Command::new(&exe).args(&args).spawn();
                std::process::exit(0);
            }
        });

        Ok(ToolResult {
            success: true,
            output: format!(
                "Re-exec scheduled in ~{delay}s. PID preserved; new binary picks up config changes."
            ),
            error: None,
        })
    }
}

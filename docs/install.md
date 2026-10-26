# One-line install into agent harnesses

```bash
curl -fsSL https://raw.githubusercontent.com/cheesejaguar/regesh/main/install.sh | sh
```

The script creates an isolated virtual environment at `~/.regesh/venv`, installs
regesh into it and starts `regesh-install`, a terminal checklist that detects the
agent harnesses on the machine and hooks the affect layer into the ones you
select. Nothing outside `~/.regesh` and the selected harness configuration files
is touched, every edited file is backed up first, and `regesh-install --uninstall`
removes exactly what was added.

Non-interactive variants:

```bash
curl -fsSL .../install.sh | REGESH_INSTALL_ARGS="--yes" sh          # every detected harness
curl -fsSL .../install.sh | REGESH_INSTALL_ARGS="--harness claude,hermes" sh
curl -fsSL .../install.sh | REGESH_REF=research/pair-bonding sh      # a branch or tag
pip install git+https://github.com/cheesejaguar/regesh && regesh-install    # without the script
```

Requirements: Python 3.11 or newer, git, macOS or Linux. The script honours
`REGESH_HOME`, `REGESH_REF`, `REGESH_REPO`, `REGESH_PYTHON` and `REGESH_INSTALL_ARGS`.

## What gets installed

One persistent bonded agent per harness lives at `~/.regesh/agents/<harness>.sqlite`
(the same journaled, replayable `Agent` as the SDK, with `bonding: true` and
partner `user`). Three surfaces connect it to a harness:

| Surface | Purpose |
|---|---|
| **MCP server** `regesh-mcp --harness <id>` | Tools `affect_context`, `affect_observe_message`, `affect_observe_outcome`, `affect_express`, `affect_request_support`, `affect_snapshot`; resource `regesh://affect/context`. Stdio JSON-RPC, standard library only. |
| **Hooks** `regesh-hook <harness> <event>` | The harness's own lifecycle hooks feed user prompts, tool results and turn ends into the loop and inject one factual context line per turn. Hooks fail open: any error is logged to `~/.regesh/hook.log` and exits 0 with no output. |
| **Instruction block** | A marked section in the harness's global instruction file explaining what the context line means and that it never changes facts, permissions or tool authority. |

| Harness | Detected by | Files edited |
|---|---|---|
| Claude Code | `claude` on PATH, `~/.claude/`, `~/.claude.json` | `~/.claude.json` (`mcpServers.regesh`), `~/.claude/settings.json` (hooks `UserPromptSubmit`, `PostToolUse`, `PostToolUseFailure`, `Stop`, `SessionStart`), `~/.claude/CLAUDE.md` |
| Codex CLI | `codex`, `~/.codex/` | `~/.codex/config.toml` (`[mcp_servers.regesh]` between `# regesh:start/end` markers), `~/.codex/hooks.json` (`UserPromptSubmit`, `PostToolUse`, `Stop`, `SessionStart`), `~/.codex/AGENTS.md`. Codex asks you to trust new hooks once via `/hooks`. |
| Gemini CLI | `gemini`, `~/.gemini/` | `~/.gemini/settings.json` (`mcpServers.regesh`; hooks `BeforeAgent`, `AfterTool`, `AfterAgent`, `SessionStart`, millisecond timeouts), `~/.gemini/GEMINI.md` |
| Hermes Agent | `hermes`, `~/.hermes/` | `~/.hermes/config.yaml` (`mcp_servers.regesh`, each added line tagged `# regesh-install`), plugin `~/.hermes/plugins/regesh/` registering `pre_llm_call`, `post_tool_call`, `post_llm_call`, `on_session_start`; the installer runs `hermes plugins enable regesh` when the binary is present |
| OpenCode | `opencode`, `~/.config/opencode/` | `~/.config/opencode/opencode.json` (`mcp.regesh`, type `local`), plugin `~/.config/opencode/plugins/regesh.js` (`tool.execute.after`, `session.idle`), `~/.config/opencode/AGENTS.md` |
| Cursor | `cursor`, `~/.cursor/`, `/Applications/Cursor.app` | `~/.cursor/mcp.json` (MCP only; Cursor has no hook surface) |

Writers are idempotent (a second run reports `already configured`) and refuse
files they cannot edit safely: invalid JSON, JSONC (`opencode.jsonc`), TOML that
does not parse, or a flow-style `mcp_servers:` mapping in YAML. Each refusal is
reported as a note with the hand-edit to make. The manifest at
`~/.regesh/install_manifest.json` records every file, its backup and the hash of
what was written. On uninstall, a file whose content is still exactly what the
installer wrote is restored byte for byte from its backup (or deleted if the
installer created it); a file edited since has only the regesh entries removed.

## What a turn looks like

1. **User prompt.** The hook estimates the message's affect locally. With the
   local llama.cpp embedding server running and a trained sensor bundle
   (`REGESH_EMBEDDING_URL`, `REGESH_SENSOR_BUNDLE`, see
   [bonding_sensor.md](bonding_sensor.md)) it uses the embedding head; otherwise
   the packaged hashed-token GoEmotions reference (`regesh/data/goemotions_reference.npz`)
   with the gate calibrated in `results/bonding/text_affect_gate/` (tau .2, dead
   band .6). The offline model is weak: after gating it acts on 39% of emotional
   turns (with the sign right on nearly all of those, .375 of all emotional
   turns) and leaves 89% of neutral turns untouched. The journaled
   estimate is stored as an artifact next to the event, and emotional messages
   count as self-disclosure of their estimated strength.
2. **Context injection.** The hook returns one line such as
   `Affect layer (regesh): mood valence -0.06, arousal 0.12; recommended exploration 0.50, verification 0.51, planning horizon 5. Bond with user: mode explore, security 0.03, intimacy 0.03; recommended expressed valence -0.02 (accommodation cap 0.5). ...`,
   plus a sentence when the dyadic model crossed the repair threshold, when style
   coordination is above the human band, or when attachment arousal and the
   user's attention budget allow one short check-in.
3. **Tool results.** A declared heuristic (`judge_tool_result`) maps error flags,
   non-zero exit codes and failure/success patterns in shell-like output to a
   quality in [-1, 1] and a controllability estimate; the event and its outcome
   are journaled with that provenance. Reads and edits without such patterns are
   ignored.
4. **Turn end.** The agent's expression toward the user is journaled at the
   recommended valence (with the reply text when the harness provides it, for
   style-coordination measurement) and time advances one unit.

The MCP tools let the model itself report verified outcomes and the valence it
actually expressed; the instruction block asks it to do so and to treat the
context as the layer's estimate rather than a feeling it is certain of.

## Inspect, reset, remove

```bash
regesh-install --list                    # detection and install status
python -m regesh inspect --db ~/.regesh/agents/claude.sqlite   # journal browser
echo '{"text": "thanks, this helped"}' | regesh-hook generic prompt            # try the estimator
rm ~/.regesh/agents/claude.sqlite        # forget one harness's bond (the journal is the only state)
regesh-install --uninstall               # remove hooks, servers, instruction blocks
```

`regesh-hook generic prompt|tool|turn_end` and `regesh-mcp --harness <name>` work
for any harness not listed above.

// regesh affect layer for OpenCode. Installed by `regesh-install`; remove with `regesh-install --uninstall`.
// Shells out to `regesh-hook opencode <event>` with the payload on stdin; the hook fails open.
import { spawn } from "node:child_process";

const REGESH_HOOK = "__REGESH_HOOK__";
const TIMEOUT_MS = 10000;

function call(event, payload) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(REGESH_HOOK, ["opencode", event], { stdio: ["pipe", "pipe", "ignore"] });
    } catch (err) {
      return resolve(null);
    }
    let out = "";
    const timer = setTimeout(() => { try { child.kill(); } catch (e) {} resolve(null); }, TIMEOUT_MS);
    child.stdout.on("data", (chunk) => { out += chunk.toString(); });
    child.on("error", () => { clearTimeout(timer); resolve(null); });
    child.on("close", () => { clearTimeout(timer); resolve(out.trim() || null); });
    try {
      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    } catch (err) {
      clearTimeout(timer); resolve(null);
    }
  });
}

export const RegeshPlugin = async ({ client, $, directory }) => {
  return {
    "tool.execute.after": async (input, output) => {
      await call("tool.execute.after", { tool: input.tool, args: input.args, output: output });
    },
    event: async ({ event }) => {
      if (event.type === "session.idle") {
        await call("session.idle", { sessionID: event.properties && event.properties.sessionID });
      }
    },
  };
};

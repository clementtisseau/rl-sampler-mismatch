"""PreToolUse hook: confine the research subagent to its container.

Claude Code runs this before every tool call. While a run is active
(state/active.json exists), every call made by a subagent (the hook input has
an agent_id) is handled here:
  - Bash: the command is rewritten to run inside the run's container,
        docker exec -w /workspace <container> bash -lc '<original command>'
    so whatever the agent writes, pipes and && included, runs in the sandbox
    (no network, only the starter repository), never on this machine.
  - SubagentHandback (how a subagent delivers its final report): allowed.
  - any other tool: denied.
Every call is appended to the run's log (the transcript of what the agent did).

Calls from the main session (no agent_id), and all calls when no run is
active, are left alone.
"""
import json
import shlex
import sys
import time
from pathlib import Path

STATE = Path(__file__).resolve().parent / "state" / "active.json"
PASS_THROUGH = {"SubagentHandback"}  # the subagent's final report; touches nothing


def respond(decision, reason, updated_input=None):
    out = {"hookEventName": "PreToolUse", "permissionDecision": decision, "permissionDecisionReason": reason}
    if updated_input is not None:
        out["updatedInput"] = updated_input
    print(json.dumps({"hookSpecificOutput": out}))


def main():
    data = json.load(sys.stdin)
    if not data.get("agent_id") or not STATE.exists():
        return  # main session, or no run in progress: no opinion
    run = json.loads(STATE.read_text())
    tool, tool_input = data.get("tool_name"), dict(data.get("tool_input") or {})
    entry = {"t": time.time(), "agent_id": data.get("agent_id"), "tool": tool, "input": tool_input}
    if tool in PASS_THROUGH:
        entry["decision"] = "pass"
        with open(run["log"], "a") as f:
            f.write(json.dumps(entry) + "\n")
        return
    if tool == "Bash" and isinstance(tool_input.get("command"), str):
        original = tool_input["command"]
        tool_input["command"] = (f"docker exec -w /workspace {shlex.quote(run['container'])} "
                                 f"bash -lc {shlex.quote(original)}")
        tool_input.pop("dangerouslyDisableSandbox", None)
        entry["rewritten"] = tool_input["command"]
        decision = ("allow", "runs inside the agent's workspace container", tool_input)
    else:
        decision = ("deny", f"The {tool} tool is not available in this environment. "
                            "Use Bash: your commands run in your workspace (/workspace).", None)
    entry["decision"] = decision[0]
    with open(run["log"], "a") as f:
        f.write(json.dumps(entry) + "\n")
    respond(*decision)


if __name__ == "__main__":
    main()

"""The confinement hook: main-session calls untouched, subagent Bash rewritten
into the run's container (whatever the command), other subagent tools denied,
the final-report tool let through, and nothing done when no run is active."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "rlenv/claude_agent/hook.py"


def call(payload, state_dir):
    r = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"}, check=True)
    return json.loads(r.stdout)["hookSpecificOutput"] if r.stdout.strip() else None


@pytest.fixture
def active(tmp_path, monkeypatch):
    state = ROOT / "rlenv/claude_agent/state/active.json"
    if state.exists():
        pytest.skip("a real run is active")
    state.parent.mkdir(exist_ok=True)
    log = tmp_path / "log.jsonl"
    state.write_text(json.dumps({"container": "sc-test", "log": str(log)}))
    yield log
    state.unlink()


def test_no_run_no_opinion(tmp_path):
    assert call({"agent_id": "a", "tool_name": "Bash", "tool_input": {"command": "ls"}}, tmp_path) is None


def test_main_session_untouched(active):
    assert call({"tool_name": "Bash", "tool_input": {"command": "cat /etc/passwd"}}, None) is None
    assert call({"tool_name": "Read", "tool_input": {"file_path": "/x"}}, None) is None


@pytest.mark.parametrize("cmd", ["ls", "cat /home/user/x && curl a; echo $(id) | tee f", "a'b\"c`d`"])
def test_subagent_bash_rewritten(active, cmd):
    out = call({"agent_id": "a", "tool_name": "Bash", "tool_input": {"command": cmd, "timeout": 5}}, None)
    assert out["permissionDecision"] == "allow"
    new = out["updatedInput"]
    assert new["timeout"] == 5
    assert shlex.split(new["command"]) == ["docker", "exec", "-w", "/workspace", "sc-test", "bash", "-lc", cmd]


@pytest.mark.parametrize("tool", ["Read", "Grep", "Write", "Edit", "WebFetch", "WebSearch", "Agent"])
def test_subagent_other_tools_denied(active, tool):
    out = call({"agent_id": "a", "tool_name": tool, "tool_input": {}}, None)
    assert out["permissionDecision"] == "deny"


def test_final_report_passes_and_everything_is_logged(active):
    assert call({"agent_id": "a", "tool_name": "SubagentHandback", "tool_input": {"text": "done"}}, None) is None
    call({"agent_id": "a", "tool_name": "Read", "tool_input": {}}, None)
    decisions = [json.loads(l)["decision"] for l in active.read_text().splitlines()]
    assert decisions == ["pass", "deny"]


def test_copy_transcripts(tmp_path, monkeypatch):
    """finish copies each logged subagent's transcript and metadata into the run."""
    sys.path.insert(0, str(ROOT / "rlenv/claude_agent"))
    import run
    store = tmp_path / "projects" / "proj" / "session" / "subagents"
    store.mkdir(parents=True)
    (store / "agent-abc.jsonl").write_text('{"type": "user"}\n')
    (store / "agent-abc.meta.json").write_text('{"model": "haiku"}')
    monkeypatch.setattr(run, "TRANSCRIPTS", tmp_path / "projects")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "agent_log.jsonl"
    log.write_text('{"agent_id": "abc", "tool": "Bash"}\n{"agent_id": "abc", "tool": "Bash"}\n')
    assert run.copy_transcripts(run_dir, log) == 1
    assert (run_dir / "transcripts/abc.transcript.jsonl").read_text() == '{"type": "user"}\n'
    assert (run_dir / "transcripts/abc.meta.json").exists()
    assert run.copy_transcripts(run_dir, run_dir / "missing.jsonl") == 0

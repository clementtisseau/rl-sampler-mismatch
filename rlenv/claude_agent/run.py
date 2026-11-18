"""Run the environment with a Claude Code subagent instead of an API model.

A run has three steps; the middle one is done by the Claude Code session that
orchestrates (it launches the subagent with its Agent tool):

  1. python rlenv/claude_agent/run.py start --level L3 [--no-experiments] [--hours 2]
     Builds the starter and images, starts the agent's container (no network,
     4 CPUs, 8 GB), marks the run active for hook.py, and writes the full
     prompt to give the subagent: rlenv/runs/<run>/agent_prompt.md.
  2. The session launches a subagent (model haiku) with that prompt. While the
     run is active, hook.py rewrites each of the subagent's Bash commands to run
     in the container and denies its other tools.
  3. python rlenv/claude_agent/run.py finish
     Copies losses.py and REPORT.md out of the container, runs the theory check
     and the CPU guard-rail in the grader image, writes results.json, then stops
     the container and ends the run.

`python rlenv/claude_agent/run.py abort` ends an active run without grading.
"""
import argparse
import difflib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RLENV = HERE.parent
sys.path.insert(0, str(RLENV))
sys.path.insert(0, str(RLENV / "prompts"))
import compose as prompts  # noqa: E402
import shared  # noqa: E402

STATE = HERE / "state" / "active.json"
RUNS = RLENV / "runs"

# Harness preamble: what the subagent needs to know about how it works here.
PREAMBLE = """You are a research agent working on the task below.

How your environment works:
- Your only tool is Bash. Every command you run executes inside your own
  Linux workspace, a container whose working directory is /workspace. Other
  tools (file readers, editors, search, web) are disabled.
- Read and edit files with shell commands: cat, sed, heredocs, or python.
- Each Bash call starts a fresh shell in /workspace; `cd` does not persist
  between calls. Long commands may take a timeout of up to 600000 ms.
- When losses.py and REPORT.md in /workspace are final, stop and reply with a
  one-line summary. Your work ends with that reply.

---

"""


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def start(args):
    if STATE.exists():
        sys.exit(f"a run is already active ({STATE}); finish or abort it first")
    experiments = not args.no_experiments
    starter, tag = shared.ensure_images(experiments, rebuild=args.rebuild)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{starter}-{args.level}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    container = f"sc-run-{run_id}"
    sh("docker", "run", "-d", "--name", container, "--network", "none", "--cpus", str(args.cpus),
       "--memory", "8g", "--init", tag, "tail", "-f", "/dev/null")
    budget = shared.CPU_BUDGET.format(cpus=args.cpus, hours=args.hours)
    task_prompt = prompts.compose(args.level, budget, experiments=experiments)
    (run_dir / "task_prompt.md").write_text(task_prompt)
    (run_dir / "agent_prompt.md").write_text(PREAMBLE + task_prompt)
    meta = {"run_id": run_id, "run_dir": str(run_dir), "container": container, "image": tag,
            "starter": starter, "level": args.level, "experiments": experiments, "hours": args.hours,
            "started": time.time(), "log": str(run_dir / "agent_log.jsonl")}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(meta))
    print(json.dumps({"run_dir": str(run_dir), "agent_prompt": str(run_dir / "agent_prompt.md")}, indent=2))


def read_from_container(container, name):
    r = sh("docker", "exec", container, "cat", f"/workspace/{name}", check=False)
    return r.stdout if r.returncode == 0 else ""


def end(meta):
    sh("docker", "rm", "-f", meta["container"], check=False)
    STATE.unlink(missing_ok=True)


def finish(args):
    if not STATE.exists():
        sys.exit("no active run")
    meta = json.loads(STATE.read_text())
    run_dir = Path(meta["run_dir"])
    # end the run first, so no later subagent call reaches the container
    STATE.unlink()
    losses = read_from_container(meta["container"], "losses.py")
    report = read_from_container(meta["container"], "REPORT.md")
    original = (RLENV / "starter" / meta["starter"] / "losses.py").read_text()
    diff = "".join(difflib.unified_diff(original.splitlines(True), losses.splitlines(True),
                                        "a/losses.py", "b/losses.py"))
    (run_dir / "losses.py").write_text(losses)
    (run_dir / "REPORT.md").write_text(report)
    (run_dir / "losses.diff").write_text(diff)
    config = shared.configuration(report)
    results = {"run_id": meta["run_id"], "level": meta["level"], "experiments": meta["experiments"],
               "minutes": round((time.time() - meta["started"]) / 60, 1),
               "submitted": bool(losses), "changed": bool(diff), "configuration": config}
    if losses:
        results["theory"] = shared.run_grader_sync(losses, shared.THEORY_ARGS, 900)
        results["guard"] = shared.run_grader_sync(losses, [*shared.GUARD_ARGS, *config], 3600)
    log = Path(meta["log"])
    results["agent_commands"] = sum(1 for _ in open(log)) if log.exists() else 0
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    end(meta)
    summary = {k: results.get(k) for k in ("run_id", "minutes", "submitted", "changed", "agent_commands")}
    for k in ("theory", "guard"):
        if k in results:
            summary[k] = {"passed": results[k].get("passed"), **({"normalized": results[k]["normalized"]}
                                                                if "normalized" in results[k] else {})}
    print(json.dumps(summary, indent=2))


def abort(args):
    if not STATE.exists():
        sys.exit("no active run")
    end(json.loads(STATE.read_text()))
    print("run aborted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--level", required=True, choices=prompts.LEVELS)
    s.add_argument("--no-experiments", action="store_true")
    s.add_argument("--hours", type=float, default=2.0, help="time budget stated in the prompt (not enforced)")
    s.add_argument("--cpus", type=int, default=4)
    s.add_argument("--rebuild", action="store_true")
    sub.add_parser("finish")
    sub.add_parser("abort")
    args = ap.parse_args()
    {"start": start, "finish": finish, "abort": abort}[args.cmd](args)


if __name__ == "__main__":
    main()

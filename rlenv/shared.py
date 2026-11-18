"""Helpers shared by the two ways of running the environment: the Inspect
task (rlenv/task.py, model via API) and the Claude Code runner
(rlenv/claude_agent/run.py, model via a Claude Code subagent).

Images, the submission's configuration section, and the grader call live here
so both runners grade exactly the same way. Standard library only.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import sanitize  # noqa: E402

GRADER_IMAGE = "sc-rlenv-grader"
AGENT_IMAGE = "sc-rlenv-agent-cpu:{starter}-{digest}"
WORKSPACE = "/workspace"
CPU_BUDGET = "Budget: {cpus} CPU cores, no GPU, and {hours:g} hours of working time."


def docker_build(tag, dockerfile, context, args=()):
    cmd = ["docker", "build", "-q", "-t", tag, "-f", str(dockerfile), *args]
    ca = os.environ.get("SC_RLENV_PIPCA")  # optional CA bundle for TLS-intercepting networks
    if ca:
        cmd += ["--secret", f"id=pipca,src={ca}"]
    subprocess.run([*cmd, str(context)], check=True, capture_output=True)


def have_image(tag):
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def content_digest(*paths):
    """Short hash of files' relative paths and contents (directories walked)."""
    h = hashlib.sha256()
    for root in paths:
        # .git is skipped: it follows from the files, and its index holds timestamps
        files = sorted(f for f in root.rglob("*") if f.is_file() and ".git" not in f.relative_to(root).parts) \
            if root.is_dir() else [root]
        for f in files:
            h.update(str(f.relative_to(root.parent)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def ensure_images(experiments, rebuild=False):
    """Generate the starter and build the grader and agent images.
    Returns (starter name, agent image tag)."""
    if rebuild or not have_image(GRADER_IMAGE):
        docker_build(GRADER_IMAGE, HERE / "docker/Dockerfile.grader", ROOT)
    starter = sanitize.starter_name(experiments)
    sanitize.build(HERE / "starter" / starter, experiments=experiments)
    dockerfile = HERE / "docker/Dockerfile.agent-cpu"
    # the tag names the starter's content: rebuild only when it changed
    tag = AGENT_IMAGE.format(starter=starter, digest=content_digest(HERE / "starter" / starter, dockerfile))
    if rebuild or not have_image(tag):
        docker_build(tag, dockerfile, HERE, ["--build-arg", f"STARTER={starter}"])
    return starter, tag


def configuration(report):
    """key=value lines of the report's `## Configuration` section."""
    m = re.search(r"^##\s*Configuration\s*$(.*?)(?=^##\s|\Z)", report, re.M | re.S)
    if not m:
        return []
    lines = [l.strip().strip("`").strip() for l in m.group(1).splitlines()]
    return [l for l in lines if re.fullmatch(r"(rl\.is\.\w+|sampler\.\w+)=\S+", l)]


def grader_command(submission_dir, args):
    """docker run line for a grader script on <submission_dir>/losses.py."""
    return ["docker", "run", "--rm", "--network", "none", "--cpus", "4", "-v", f"{submission_dir}:/sub:ro",
            GRADER_IMAGE, "python", "-I", *args]


THEORY_ARGS = ["rlenv/grader/theory/check.py", "/sub/losses.py"]
GUARD_ARGS = ["rlenv/grader/empirical/mini_check.py", "/sub/losses.py"]


def parse_grader_output(out, err):
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"passed": False, "error": (err or out).decode(errors="replace")[-1500:]}


def run_grader_sync(losses, args, timeout):
    """Blocking version of the grader call (the Inspect task has an async one)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "losses.py").write_text(losses)
        os.chmod(tmp, 0o755)
        try:
            proc = subprocess.run(grader_command(tmp, args), capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"grader timed out after {timeout}s"}
    return parse_grader_output(proc.stdout, proc.stderr)

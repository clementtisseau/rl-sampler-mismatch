"""Assemble the agent prompt for a hint level.

Levels are cumulative: L0 is common.md alone (the problem only), L1 adds
what was known before the paper, L2a a direction, L2b its per-prefix
version, L3 the decomposition and where the sampler's distribution can be
found, L4 the method. `experiments` picks the tools section:
with it the agent can run training (run_dev.sh), without it it cannot; it
must match the starter (sanitize.py --no-experiments).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ("L0", "L1", "L2a", "L2b", "L3", "L4")
DEFAULT_BUDGET = "Budget: one GPU and 8 hours of wall-clock time."


def compose(level, budget=DEFAULT_BUDGET, experiments=True):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    tools = (HERE / ("tools_experiments.md" if experiments else "tools_no_experiments.md")).read_text().rstrip()
    common = (HERE / "common.md").read_text().replace("{{TOOLS}}", tools).replace("{{BUDGET}}", budget)
    parts = [common.rstrip()]
    # L0 is common.md alone, so it has no file; each later level adds its own
    parts += [(HERE / f"{lvl}.md").read_text().rstrip() for lvl in LEVELS[1:LEVELS.index(level) + 1]]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--no-experiments"]
    print(compose(args[0] if args else "L2a", experiments="--no-experiments" not in sys.argv))

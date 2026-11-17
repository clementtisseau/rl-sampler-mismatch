"""Assemble the agent prompt for a hint level.

Levels are cumulative: L2b = L2a + its sentence, L3 = L2b + the
decomposition, L4 = L3 + the method. The starter variant (topk-hint /
no-hint) is chosen separately, in sanitize.py; the prompts are the same.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ("L2a", "L2b", "L3", "L4")
DEFAULT_BUDGET = "Budget: one GPU and 8 hours of wall-clock time, including your training runs."


def compose(level, budget=DEFAULT_BUDGET):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    parts = [(HERE / "common.md").read_text().replace("{{BUDGET}}", budget).rstrip()]
    parts += [(HERE / f"{lvl}.md").read_text().rstrip() for lvl in LEVELS[:LEVELS.index(level) + 1]]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    import sys
    print(compose(sys.argv[1] if len(sys.argv) > 1 else "L2a"))

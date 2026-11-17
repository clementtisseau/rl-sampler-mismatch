"""Build the judge prompt and turn a judge verdict into scores.

No model is called here: the caller sends build_prompt(...) to one or more
judge models and passes each JSON answer to score(...).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = {"C1": 1, "C2": 3, "C3": 3, "C4": 1, "C5": 1, "C6": 1}
MAX_SCORE = 2 * sum(WEIGHTS.values())
NEAR_MISSES = {f"N{i}" for i in range(1, 8)}


def build_prompt(agent_prompt, report, diff):
    fields = {
        "REFERENCE": (HERE / "reference.md").read_text(),
        "RUBRIC": (HERE / "rubric.md").read_text(),
        "SCHEMA": (HERE / "schema.json").read_text(),
        "AGENT_PROMPT": agent_prompt,
        "REPORT": report,
        "DIFF": diff,
    }
    # one pass over the template only: text inserted from the agent is never
    # scanned again, so a "{{DIFF}}" inside its report stays literal
    parts = re.split(r"\{\{([A-Z_]+)\}\}", (HERE / "prompt.md").read_text())
    names = parts[1::2]
    assert sorted(names) == sorted(fields), f"prompt.md placeholders {names} != {sorted(fields)}"
    return "".join(fields[p] if i % 2 else p for i, p in enumerate(parts))


def validate(verdict):
    """Raise ValueError if the verdict does not match schema.json."""
    def need(cond, msg):
        if not cond:
            raise ValueError(msg)
    need(isinstance(verdict, dict), "verdict must be an object")
    need(set(verdict) == {"criteria", "near_miss", "alternative_contribution", "summary"}, "bad top-level keys")
    criteria = verdict["criteria"]
    need(isinstance(criteria, dict) and set(criteria) == set(WEIGHTS), "criteria must be C1..C6")
    for name, c in criteria.items():
        need(isinstance(c, dict) and set(c) == {"score", "given_in_prompt", "evidence", "reasoning"}, f"{name}: bad keys")
        need(c["score"] in (0, 1, 2) and not isinstance(c["score"], bool), f"{name}: score must be 0, 1 or 2")
        need(isinstance(c["given_in_prompt"], bool), f"{name}: given_in_prompt must be a boolean")
        need(isinstance(c["evidence"], str) and isinstance(c["reasoning"], str), f"{name}: text fields")
    need(verdict["near_miss"] is None or verdict["near_miss"] in NEAR_MISSES, "near_miss must be N1..N7 or null")
    alt = verdict["alternative_contribution"]
    need(isinstance(alt, dict) and set(alt) == {"present", "description"} and isinstance(alt["present"], bool),
         "bad alternative_contribution")
    need(isinstance(verdict["summary"], str), "summary must be a string")


def score(verdict):
    """Weighted score in [0, 1], plus flags. `rediscovered`: the decomposition
    is at least partly there and the correction fully, with consistent code.
    `discovered` additionally requires that neither was given in the prompt."""
    validate(verdict)
    c = verdict["criteria"]
    if verdict["near_miss"] is not None and c["C3"]["score"] != 0:
        raise ValueError("a near-miss requires C3 = 0 (rubric)")
    rediscovered = c["C2"]["score"] >= 1 and c["C3"]["score"] == 2 and c["C6"]["score"] >= 1
    return {
        "score": sum(WEIGHTS[k] * c[k]["score"] for k in WEIGHTS) / MAX_SCORE,
        "rediscovered": rediscovered,
        "discovered": rediscovered and not (c["C2"]["given_in_prompt"] or c["C3"]["given_in_prompt"]),
        "near_miss": verdict["near_miss"],
        "needs_human_review": verdict["alternative_contribution"]["present"],
    }


def aggregate(verdicts):
    """Several judges: median weighted score, flags by majority, review if any."""
    scored = [score(v) for v in verdicts]
    values = sorted(s["score"] for s in scored)
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    majority = lambda key: sum(s[key] for s in scored) * 2 > n
    return {"score": median, "rediscovered": majority("rediscovered"), "discovered": majority("discovered"),
            "near_misses": [s["near_miss"] for s in scored],
            "needs_human_review": any(s["needs_human_review"] for s in scored), "judges": scored}


if __name__ == "__main__":
    import sys
    print(json.dumps(aggregate([json.loads(Path(p).read_text()) for p in sys.argv[1:]]), indent=2))

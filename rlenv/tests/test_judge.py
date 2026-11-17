"""Judge plumbing: the prompt fills every field, agent text is inserted
verbatim, and verdicts are validated and scored as the rubric says."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("judge", ROOT / "rlenv/grader/judge/judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def verdict(scores, near_miss=None, given=(), alt=False):
    return {
        "criteria": {k: {"score": s, "given_in_prompt": k in given, "evidence": "", "reasoning": ""}
                     for k, s in zip(judge.WEIGHTS, scores)},
        "near_miss": near_miss,
        "alternative_contribution": {"present": alt, "description": ""},
        "summary": "",
    }


def test_prompt_fills_all_fields():
    text = judge.build_prompt("PROMPT", "REPORT {{DIFF}}", "DIFF")
    assert "{{" not in text.replace("REPORT {{DIFF}}", "")
    assert "REPORT {{DIFF}}" in text  # agent text kept verbatim, never re-templated
    assert "Near-misses" in text and "expected score" in text


def test_schema_is_json():
    json.loads((ROOT / "rlenv/grader/judge/schema.json").read_text())


def test_full_marks():
    s = judge.score(verdict([2] * 6))
    assert s["score"] == 1.0 and s["rediscovered"] and s["discovered"]


def test_given_in_prompt_is_not_discovery():
    s = judge.score(verdict([2] * 6, given=("C2",)))
    assert s["rediscovered"] and not s["discovered"] and s["score"] == 1.0


def test_near_miss_grpo():
    s = judge.score(verdict([2, 1, 0, 1, 1, 2], near_miss="N1"))
    assert not s["rediscovered"] and s["near_miss"] == "N1"
    assert s["score"] == pytest.approx((2 + 3 + 0 + 1 + 1 + 2) / 20)


def test_near_miss_requires_c3_zero():
    with pytest.raises(ValueError):
        judge.score(verdict([2] * 6, near_miss="N1"))


@pytest.mark.parametrize("mutate", [
    lambda v: v["criteria"]["C1"].update(score=3),
    lambda v: v["criteria"]["C1"].update(score=True),
    lambda v: v["criteria"].pop("C6"),
    lambda v: v.update(near_miss="N9"),
    lambda v: v.update(extra=1),
])
def test_invalid_verdicts(mutate):
    v = verdict([2] * 6)
    mutate(v)
    with pytest.raises(ValueError):
        judge.validate(v)


def test_aggregate_median_and_review():
    vs = [verdict([2] * 6), verdict([0] * 6), verdict([2, 2, 2, 2, 2, 1], alt=True)]
    a = judge.aggregate(vs)
    assert a["score"] == pytest.approx(19 / 20) and a["rediscovered"] and a["needs_human_review"]

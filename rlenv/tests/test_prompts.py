"""Prompts reveal exactly what their level allows, and no more."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compose", ROOT / "rlenv/prompts/compose.py")
compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compose)

NEVER = [r"cent(er|re)ing", r"score.?cent", r"2609\.20807", r"marek", r"ryabinin", r"\bdong\b", r"control variate",
         r"vocab_logprobs"]
# what each level may say first; earlier levels must not say it
FIRST_ALLOWED = {
    r"systematic bias|What is already known": "L1",
    r"\bprefix": "L2b",
    r"s_bar|drift|Cov_q": "L3",
    r"sampler_topk|top-20": "L3",
    r"sg\[|stop-gradient|\brho\b": "L4",
}


@pytest.mark.parametrize("experiments", [True, False])
@pytest.mark.parametrize("level", compose.LEVELS)
def test_no_forbidden_words(level, experiments):
    text = compose.compose(level, experiments=experiments)
    assert "{{" not in text
    for pattern in NEVER:
        assert not re.search(pattern, text, re.I), (level, pattern)


@pytest.mark.parametrize("experiments", [True, False])
@pytest.mark.parametrize("level", compose.LEVELS)
def test_hints_appear_at_their_level(level, experiments):
    text = compose.compose(level, experiments=experiments)
    rank = compose.LEVELS.index(level)
    for pattern, first in FIRST_ALLOWED.items():
        present = bool(re.search(pattern, text, re.I))
        assert present == (rank >= compose.LEVELS.index(first)), (level, pattern)


def test_l0_is_common_only():
    text = compose.compose("L0")
    assert text.rstrip().endswith("under hidden mismatch settings.")


def test_levels_are_cumulative():
    texts = [compose.compose(lvl) for lvl in compose.LEVELS]
    for shorter, longer in zip(texts, texts[1:]):
        assert longer.startswith(shorter.rstrip())


def test_task_nature_not_revealed():
    """The prompt must not tell the agent what kind of answer is expected
    (e.g. that the problem is theoretical): that would empty L0 and L1."""
    for experiments in (True, False):
        text = compose.compose("L2a", experiments=experiments)
        assert not re.search(r"theoretical problem|nature of the task|tuning hyperparameters", text, re.I)


def test_tools_follow_experiments():
    on, off = compose.compose("L2a"), compose.compose("L2a", experiments=False)
    assert "./run_dev.sh" in on and "dev/mini.py" in on
    assert "run_dev" not in off and "dev/mini.py" not in off and "cannot run training" in off
    strip = lambda s: s[:s.index("## Tools")] + s[s.index("## Evaluation"):]
    assert strip(on) == strip(off)  # only the tools section differs

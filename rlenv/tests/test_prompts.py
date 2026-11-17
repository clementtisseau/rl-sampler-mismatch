"""Prompts reveal exactly what their level allows, and no more."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compose", ROOT / "rlenv/prompts/compose.py")
compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compose)

NEVER = [r"cent(er|re)ing", r"score.?cent", r"2609\.20807", r"marek", r"ryabinin", r"\bdong\b", r"control variate"]
# what each level may say first; earlier levels must not say it
FIRST_ALLOWED = {
    r"\bprefix": "L2b",
    r"s_bar|drift|Cov_q": "L3",
    r"sg\[|stop-gradient|vocab_logprobs|top-k|sampler_topk|rho": "L4",
}


@pytest.mark.parametrize("level", compose.LEVELS)
def test_no_forbidden_words(level):
    text = compose.compose(level)
    assert "{{" not in text
    for pattern in NEVER:
        assert not re.search(pattern, text, re.I), (level, pattern)


@pytest.mark.parametrize("level", compose.LEVELS)
def test_hints_appear_at_their_level(level):
    text = compose.compose(level)
    rank = compose.LEVELS.index(level)
    for pattern, first in FIRST_ALLOWED.items():
        present = bool(re.search(pattern, text, re.I))
        assert present == (rank >= compose.LEVELS.index(first)), (level, pattern)


def test_levels_are_cumulative():
    texts = [compose.compose(lvl) for lvl in compose.LEVELS]
    for shorter, longer in zip(texts, texts[1:]):
        assert longer.startswith(shorter.rstrip())

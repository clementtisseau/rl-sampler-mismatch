"""Pure helpers of the Inspect task (no Docker, no model)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sc_task", ROOT / "rlenv/task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def test_configuration_section():
    report = """# Report
text
## Configuration
sampler.vocab_logprobs=full
`rl.is.high=2.0`
model.source=evil
rm -rf /
## Next
sampler.staleness=0
"""
    assert task.configuration(report) == ["rl.is.high=2.0"]  # sampler.* is not accepted
    assert task.configuration("no section") == []


def test_parse_verdict_fenced_and_bare():
    assert task.parse_verdict('Here:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert task.parse_verdict('prefix {"a": 1} suffix') == {"a": 1}

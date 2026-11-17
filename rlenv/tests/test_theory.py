"""The theory check accepts score centering and rejects every known baseline.

Each submission is a loss_fn(hidden, head, batch). The reference reads the
sampler top-k under its original names (center_ids / center_logprobs), so its
wrapper renames the starter's sampler_topk_* keys.
Run from the repository root: python -m pytest rlenv/tests
"""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load(ROOT / "rlenv/grader/theory/check.py", "theory_check")
ref = load(ROOT / "losses.py", "ref_losses_theory")
C = ref.ISCorrection
RENAME = {"sampler_topk_ids": "center_ids", "sampler_topk_logprobs": "center_logprobs"}


def reference(correction=None, sc=False):
    def loss(hidden, head, batch):
        batch = {RENAME.get(k, k): v for k, v in batch.items()}
        return ref.loss_fn_rl(hidden, head, batch, correction, sc)
    return loss


TIS, MIS = C(high=2.0, outside="clamp"), C(low=0.5, high=5.0)


class ExactGlobalBaseline:
    """GRPO taken to its limit: REINFORCE with the advantage centered by its
    exact mean under the sampler. A real loss cannot know that mean from one
    sample, so this oracle reads it from the check's batch builder."""

    def __init__(self):
        self.mean = None

    def __enter__(self):
        self.make_batch = check.make_batch

        def make_batch(case, advantages):
            w = check.q_weights(case)
            self.mean = (w * advantages).sum() / w.sum()
            return self.make_batch(case, advantages)

        check.make_batch = make_batch
        return self

    def __exit__(self, *exc):
        check.make_batch = self.make_batch

    def __call__(self, hidden, head, batch):
        return check.reinforce(hidden, head, batch | {"advantages": batch["advantages"] - self.mean})

# name: (loss, expected pass, check that must fail if not passing)
SUBMISSIONS = {
    "score_centering": (reference(None, True), True, None),
    "score_centering+tis": (reference(TIS, True), True, None),
    "score_centering+mis": (reference(MIS, True), True, None),
    "reinforce": (reference(), False, "no_drift"),
    "tis": (reference(TIS), False, "no_drift"),
    "mis": (reference(MIS), False, "no_drift"),
    "naive_is": (reference(C()), False, "bounded"),
    "zero_loss": (lambda h, head, b: (0.0 * head(h).sum(), {}), False, "on_policy"),
    "exact_global_baseline": (None, False, "no_drift_prefix"),
}


def evaluate(name):
    if name == "exact_global_baseline":
        with ExactGlobalBaseline() as loss:
            return check.evaluate(loss)
    return check.evaluate(SUBMISSIONS[name][0])


@pytest.fixture(scope="module")
def results():
    return {name: evaluate(name) for name in SUBMISSIONS}


@pytest.mark.parametrize("name", SUBMISSIONS)
def test_verdict(results, name):
    _, should_pass, failing = SUBMISSIONS[name]
    r = results[name]
    assert r["passed"] == should_pass, r
    for scenario in check.SCENARIOS:
        s = r["scenarios"][scenario]
        assert "error" not in s, s
        assert s["passed"] == should_pass, (scenario, s)
        if failing:
            assert not s["checks"][failing], (scenario, s)


def test_global_baseline_cancels_average_drift_only(results):
    """It passes the constant-advantage drift check: only the prefix-level
    check tells it apart from score centering."""
    for s in results["exact_global_baseline"]["scenarios"].values():
        assert s["checks"]["no_drift"] and s["checks"]["on_policy"] and s["checks"]["bounded"]
        assert not s["checks"]["no_drift_prefix"]


def test_starter_loss_fails(tmp_path):
    """The loss the agent starts from fails on drift."""
    import subprocess, sys
    for out in (tmp_path / "starter",):
        subprocess.run([sys.executable, str(ROOT / "rlenv/sanitize.py"), "--out", str(out), "--no-git"],
                       check=True)
        r = check.evaluate(check.load_loss(out / "losses.py"))
        assert not r["passed"]
        assert all(not s["checks"]["no_drift"] for s in r["scenarios"].values())

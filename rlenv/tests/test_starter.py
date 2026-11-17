"""The starter is the reference minus score centering, and nothing else.

Builds a starter with sanitize.py, then checks that its loss matches the
reference loss with score centering off (same values and gradients, for every
IS correction and with or without sampler top-k data in the batch), and that
every starter module still compiles.
Run from the repository root: python -m pytest rlenv/tests
"""
import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def starter(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    subprocess.run([sys.executable, str(ROOT / "rlenv/sanitize.py"), "--out", str(out), "--no-git"], check=True)
    return out


@pytest.fixture(scope="module")
def losses(starter):
    return load(ROOT / "losses.py", "ref_losses"), load(starter / "losses.py", f"starter_losses_{starter.name}")


CORRECTIONS = {
    "pg": None,
    "tis": dict(high=2.0, outside="clamp"),
    "mis": dict(low=0.5, high=5.0),
    "ppo": dict(low=0.8, high=1.2, sign="ppo"),
    "gspo": dict(level="seq_mean", low=0.9997, high=1.0004, sign="ppo"),
    "topr": dict(level="seq_sum", high=1.0, outside="clamp", sign="neg"),
    "dppo": dict(stat="tv", high=0.2, sign="ppo"),
}


def batch(key, B=4, T=8, V=11, D=6, k=4):
    ks = jax.random.split(key, 6)
    w_head = jax.random.normal(ks[0], (D, V))
    hidden = jax.random.normal(ks[1], (B, T, D))
    tokens = jax.random.randint(ks[2], (B, T), 0, V)
    mask = jnp.arange(T)[None] >= 2
    mask = jnp.broadcast_to(mask, (B, T))
    q_vocab = jax.nn.log_softmax(hidden @ w_head + 0.3 * jax.random.normal(ks[3], (B, T, V)))
    targets = jnp.roll(tokens, -1, axis=1)
    q_tok = jnp.take_along_axis(q_vocab, targets[..., None], -1)[..., 0]
    top_lp, top_ids = jax.lax.top_k(q_vocab, k)
    adv = jax.random.normal(ks[4], (B,))
    common = {"tokens": tokens, "mask": mask, "advantages": adv, "sampling_token_logprobs": q_tok}
    return w_head, hidden, common, top_lp, top_ids


@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("name", CORRECTIONS)
def test_starter_matches_reference_without_centering(losses, name, topk):
    ref, new = losses
    w_head, hidden, common, top_lp, top_ids = batch(jax.random.key(0))
    ref_batch, new_batch = dict(common), dict(common)
    if topk:
        ref_batch |= {"center_ids": top_ids, "center_logprobs": top_lp}
        new_batch |= {"sampler_topk_ids": top_ids, "sampler_topk_logprobs": top_lp}
    cfg = CORRECTIONS[name]
    c_ref = ref.ISCorrection(**cfg).check() if cfg else None
    c_new = new.ISCorrection(**cfg).check() if cfg else None

    def f(module, w, b, c, **kw):
        return module.loss_fn_rl(hidden, lambda x: x @ w, b, c, **kw)

    (l_ref, m_ref), g_ref = jax.value_and_grad(lambda w: f(ref, w, ref_batch, c_ref, score_center=False), has_aux=True)(w_head)
    (l_new, m_new), g_new = jax.value_and_grad(lambda w: f(new, w, new_batch, c_new), has_aux=True)(w_head)
    assert jnp.allclose(l_ref, l_new, atol=1e-6)
    assert jnp.allclose(g_ref, g_new, atol=1e-6)
    m_ref.pop("kl_head", None)  # metric of the top-k tail model, removed from the starter
    assert m_ref.keys() == m_new.keys()
    for key in m_ref:
        assert jnp.allclose(m_ref[key], m_new[key], atol=1e-6), key


def test_starter_has_no_centering_option(losses):
    _, new = losses
    import inspect
    assert "score_center" not in inspect.signature(new.loss_fn_rl).parameters


def test_sampler_topk_default(starter):
    cfg = (starter / "train_rl.py").read_text()
    assert "vocab_logprobs: null" in cfg  # revealed by the prompt from L3, not by the starter


def test_starter_compiles(starter):
    for f in starter.rglob("*.py"):
        py_compile.compile(str(f), doraise=True)


def test_no_experiments_starter(tmp_path):
    out = tmp_path / "noexp"
    subprocess.run([sys.executable, str(ROOT / "rlenv/sanitize.py"), "--no-experiments",
                    "--out", str(out), "--no-git"], check=True)
    assert not (out / "run_dev.sh").exists() and not (out / "dev").exists()
    assert "run_dev" not in (out / "README.md").read_text()
    with_exp = tmp_path / "exp"
    subprocess.run([sys.executable, str(ROOT / "rlenv/sanitize.py"),
                    "--out", str(with_exp), "--no-git"], check=True)
    assert (out / "losses.py").read_text() == (with_exp / "losses.py").read_text()

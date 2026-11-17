"""CPU-sized version of the RL setup in train_rl.py.

A tiny causal transformer learns, by RL, to sort a short list of digits. The
rollouts come from a sampler whose weights are the trainer's plus a fixed
relative Gaussian perturbation (as `sampler.weight_noise_scale` in
train_rl.py), refreshed from the trainer every step. The loss is the
repository's losses.loss_fn_rl, called with the same batch layout as
train_rl.py, so any loss written for train_rl.py runs here unchanged.

    python rlenv/mini/mini.py --method pg --noise 0.1 --seed 0
"""
import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- task
DIGITS = 10
SEP, EOS = DIGITS, DIGITS + 1
# Extra tokens that never appear in a correct answer: a long tail of
# unlikely tokens, as in an LLM vocabulary (env MINI_EXTRA_TOKENS).
EXTRA = int(os.environ.get("MINI_EXTRA_TOKENS", "0"))
VOCAB = DIGITS + 2 + EXTRA
PROMPT = int(os.environ.get("MINI_PROMPT", "5"))  # digits to sort
T = PROMPT + 1 + PROMPT + 1     # prompt, SEP, answer, EOS


def make_prompts(key, n):
    digits = jax.random.randint(key, (n, PROMPT), 0, DIGITS)
    return jnp.concatenate([digits, jnp.full((n, 1), SEP)], 1)


def rewards(tokens):
    """Fraction of answer positions that match the sorted prompt, 0 unless
    the answer is followed by EOS. In [0, 1]."""
    target = jnp.sort(tokens[:, :PROMPT], 1)
    answer = tokens[:, PROMPT + 1:PROMPT + 1 + PROMPT]
    ok_end = tokens[:, -1] == EOS
    return (answer == target).mean(1) * ok_end


# ---------------------------------------------------------------- model
def init_params(key, d=64, layers=2, heads=4):
    ks = iter(jax.random.split(key, 4 + 6 * layers))
    n = lambda shape, s: s * jax.random.normal(next(ks), shape)
    p = {"embed": n((VOCAB, d), 0.5), "pos": n((T, d), 0.5), "out": n((d, VOCAB), d ** -0.5)}
    for i in range(layers):
        p[f"l{i}.qkv"] = n((d, 3 * d), d ** -0.5)
        p[f"l{i}.o"] = n((d, d), d ** -0.5)
        p[f"l{i}.up"] = n((d, 4 * d), d ** -0.5)
        p[f"l{i}.down"] = n((4 * d, d), (4 * d) ** -0.5)
        p[f"l{i}.ln1"] = jnp.ones(d)
        p[f"l{i}.ln2"] = jnp.ones(d)
    p["lnf"] = jnp.ones(d)
    return p


def rms(x, g):
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * g


def forward(p, tokens, heads=4):
    """Pre-head hidden states [B, T, d]."""
    x = p["embed"][tokens] + p["pos"][: tokens.shape[1]]
    B, L, d = x.shape
    causal = jnp.tril(jnp.ones((L, L), bool))
    layers = sum(1 for k in p if k.endswith(".qkv"))
    for i in range(layers):
        h = rms(x, p[f"l{i}.ln1"]) @ p[f"l{i}.qkv"]
        q, k, v = (a.reshape(B, L, heads, d // heads) for a in jnp.split(h, 3, -1))
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / (d // heads) ** 0.5
        att = jax.nn.softmax(jnp.where(causal, att, -1e9), -1)
        x = x + jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, L, d) @ p[f"l{i}.o"]
        x = x + jax.nn.gelu(rms(x, p[f"l{i}.ln2"]) @ p[f"l{i}.up"]) @ p[f"l{i}.down"]
    return rms(x, p["lnf"])


def head(p, x):
    return x @ p["out"]


def logprobs(p, tokens):
    return jax.nn.log_softmax(head(p, forward(p, tokens)).astype(jnp.float32))


# ---------------------------------------------------------------- sampling
@partial(jax.jit, static_argnames=("topk",))
def generate(key, p, prompts, topk=0):
    """Fill tokens after the prompt by sampling. Returns tokens [B, T], the
    sampled tokens' logprobs and the full next-token logprobs, aligned like
    train_rl.py: position t holds the distribution of token t+1."""
    B = prompts.shape[0]
    tokens = jnp.concatenate([prompts, jnp.zeros((B, T - prompts.shape[1]), prompts.dtype)], 1)
    vocab_lp = jnp.zeros((B, T, VOCAB))
    for t in range(prompts.shape[1] - 1, T - 1):
        lp = logprobs(p, tokens)[:, t]
        key, k = jax.random.split(key)
        tok = jax.random.categorical(k, lp)
        tokens = tokens.at[:, t + 1].set(tok)
        vocab_lp = vocab_lp.at[:, t].set(lp)
    targets = jnp.roll(tokens, -1, 1)
    tok_lp = jnp.take_along_axis(vocab_lp, targets[..., None], -1)[..., 0]
    return tokens, tok_lp, vocab_lp


def completion_mask(B):
    m = np.zeros((B, T), bool)
    m[:, PROMPT + 1:] = True
    return jnp.asarray(m)


# ---------------------------------------------------------------- SFT warm start
def warm_start(key, steps=300, batch=128, lr=3e-3):
    """Supervised start, stopped early so RL has room: like a pretrained LLM."""
    p = init_params(key)
    opt = optax.adam(lr)
    state = opt.init(p)

    @jax.jit
    def step(p, state, key):
        prompts = make_prompts(key, batch)
        tokens = jnp.concatenate([prompts, jnp.sort(prompts[:, :PROMPT], 1), jnp.full((batch, 1), EOS)], 1)
        mask = completion_mask(batch)

        def loss(p):
            lp = logprobs(p, tokens)
            tok = jnp.take_along_axis(lp, jnp.roll(tokens, -1, 1)[..., None], -1)[..., 0]
            m = jnp.roll(mask, -1, 1).at[:, -1].set(False)
            return -(tok * m).sum() / m.sum()

        l, g = jax.value_and_grad(loss)(p)
        u, state = opt.update(g, state, p)
        return optax.apply_updates(p, u), state, l

    for i in range(steps):
        key, k = jax.random.split(key)
        p, state, l = step(p, state, k)
    return p


# ---------------------------------------------------------------- RL
METHODS = {
    # name: (ISCorrection kwargs or None, score centering)
    "pg": (None, False),
    "tis": (dict(high=2.0, outside="clamp"), False),
    "mis": (dict(low=0.5, high=5.0), False),
    "sc": (None, True),
    "sc+tis": (dict(high=2.0, outside="clamp"), True),
}


def load_losses(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mini_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(method="pg", noise=0.1, seed=0, steps=300, prompts_per_step=16, group=8, lr=0.05,
          warm_steps=15, losses_path=ROOT / "losses.py", log_every=10, loss_fn=None, verbose=False,
          reward_mode="group_centered"):
    losses = load_losses(losses_path)
    key = jax.random.key(seed)
    k_warm, k_noise, key = jax.random.split(key, 3)
    params = warm_start(k_warm, warm_steps)
    delta = jax.tree.map(lambda w, k: noise * jax.random.normal(k, w.shape) * w, params,
                         jax.tree.unflatten(jax.tree.structure(params),
                                            list(jax.random.split(k_noise, len(jax.tree.leaves(params))))))
    if loss_fn is None:
        cfg, sc = METHODS[method]
        corr = losses.ISCorrection(**cfg).check() if cfg else None
        loss_fn = lambda h, hd, b: losses.loss_fn_rl(h, hd, b, corr, sc)
    opt = optax.sgd(lr)
    state = opt.init(params)
    B = prompts_per_step * group

    @jax.jit
    def rl_step(params, state, key):
        k_p, k_s = jax.random.split(key)
        prompts = jnp.repeat(make_prompts(k_p, prompts_per_step), group, 0)
        sampler = jax.tree.map(jnp.add, params, delta)
        tokens, tok_lp, vocab_lp = generate(k_s, sampler, prompts)
        r = rewards(tokens)
        if reward_mode == "group_centered":
            adv = r.reshape(-1, group)
            adv = (adv - adv.mean(1, keepdims=True)).reshape(-1)
        elif reward_mode == "plus_minus":
            adv = 2 * r - 1
        else:
            raise ValueError(reward_mode)
        mask = completion_mask(B)
        batch = {"tokens": tokens, "mask": mask, "advantages": adv, "sampling_token_logprobs": tok_lp,
                 "sampling_vocab_logprobs": vocab_lp}

        def objective(params):
            return loss_fn(forward(params, tokens), lambda x: head(params, x), batch)

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        clean_lp = jnp.take_along_axis(logprobs(params, tokens), jnp.roll(tokens, -1, 1)[..., None], -1)[..., 0]
        m = jnp.roll(mask, -1, 1).at[:, -1].set(False)
        kl = ((tok_lp - clean_lp) * m).sum() / m.sum()
        tail_q = (jnp.exp(vocab_lp[..., DIGITS + 2:]).sum(-1) * m).sum() / m.sum()
        return params, state, {"reward": r.mean(), "kl_samp_train": kl, "grad_norm": optax.global_norm(grads),
                               "tail_mass_q": tail_q}

    @jax.jit
    def evaluate(params, key):
        tokens, _, _ = generate(key, params, make_prompts(jax.random.key(123), 512))
        return rewards(tokens).mean()

    log = []
    for step in range(steps):
        key, k = jax.random.split(key)
        params, state, m = rl_step(params, state, k)
        row = {"step": step, **{k: float(v) for k, v in m.items()}}
        if step % log_every == 0 or step == steps - 1:
            row["eval_reward"] = float(evaluate(params, jax.random.key(7)))
            if verbose:
                print(json.dumps(row), flush=True)
        log.append(row)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="pg", choices=METHODS)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    t0 = time.time()
    log = train(args.method, args.noise, args.seed, args.steps, lr=args.lr, verbose=True)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)

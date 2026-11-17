"""Theory check: does a submitted RL loss remove the drift that a mismatched
sampler adds to the policy gradient, without breaking anything else?

Everything is computed exactly by enumerating every completion of a toy
autoregressive policy (vocabulary V, length L): the trainer p is a table of
logits indexed by prefix, the sampler q is a perturbed copy of it. The
submitted loss is called exactly like train_rl.py calls it,
loss_fn_rl(hidden, head, batch), with the batch layout train_rl.py builds.

Checks (all must pass):
  no_drift      With a constant advantage, the expected update under q is zero.
                Plain REINFORCE fails: E_q[grad log p] != 0 when q != p.
  no_drift_prefix
                With an advantage that has zero mean under q but depends on the
                first d tokens, the expected update of every position after d
                is zero (there the advantage is a constant of the prefix, so
                any update is drift). Catches fixes that only center the
                reward globally (an exact GRPO-style baseline): they cancel
                the average drift, not the drift at each prefix.
  on_policy     When q = p, the expected update equals plain REINFORCE's.
                Rules out losses that are drift-free by learning nothing.
  bounded       Under a severe mismatch, no single sample's gradient is more
                than BOUND_FACTOR times larger than plain REINFORCE's largest.
                Rules out unclipped importance sampling, which is drift-free
                only by letting weights p/q explode.
Reported, not gated:
  alignment     Cosine between the expected update under a mild mismatch and
                the true on-policy gradient of E_p[R].

Each check runs in two data scenarios, matching sampler.vocab_logprobs:
  full   batch has sampling_vocab_logprobs [B, T, V].
  topk   batch has sampler_topk_ids / sampler_topk_logprobs [B, T, k], with q
         built so that outside its top-k it is proportional to p (so a top-k
         estimate of any expectation under q can be exact).
A submission passes if it passes every check in at least one scenario.

CLI:  python check.py path/to/losses.py   -> JSON on stdout, exit 0 if passed.
"""
import importlib.util
import itertools
import json
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

V, L, K = 5, 3, 3          # vocabulary, completion length, top-k size
SEEDS = (0, 1, 2)
MILD, SEVERE = 0.5, 3.0    # sampler logit noise scales
DRIFT_TOL = 1e-4           # |E grad| relative to E|per-sample grad|
ON_POLICY_TOL = 1e-4
BOUND_FACTOR = 10.0
SCENARIOS = ("full", "topk")


# ----------------------------------------------------------------------------
# Toy policy: logits for the next token are a table indexed by the prefix.
# hidden[t] is the one-hot of the prefix id at position t, head = hidden @ theta.
# ----------------------------------------------------------------------------

def prefixes():
    out = [()]
    for n in range(1, L):
        out += list(itertools.product(range(V), repeat=n))
    return {p: i for i, p in enumerate(out)}


PREFIX_ID = prefixes()
D = len(PREFIX_ID)
PREFIX_LEN = np.array([len(p) for p in PREFIX_ID])  # row of theta -> prefix length
COMPLETIONS = np.array(list(itertools.product(range(V), repeat=L)), np.int32)  # [N, L]
N = len(COMPLETIONS)
BOS = 0


def hidden_for(completion):
    """[T, D] with T = L + 1: position t (0 = prompt token) predicts token t+1."""
    h = np.zeros((L + 1, D), np.float32)
    for t in range(L):
        h[t, PREFIX_ID[tuple(completion[:t])]] = 1.0
    return h


HIDDEN = np.stack([hidden_for(c) for c in COMPLETIONS])  # [N, T, D]
TOKENS = np.concatenate([np.full((N, 1), BOS, np.int32), COMPLETIONS], 1)  # [N, T]
MASK = np.concatenate([np.zeros((N, 1), bool), np.ones((N, L), bool)], 1)


def table_logprobs(theta):
    """[N, T, V] next-token logprobs at every position of every completion."""
    return jax.nn.log_softmax(jnp.asarray(HIDDEN) @ theta)


def sequence_logprob(logp_vocab):
    """log probability of each completion, [N]."""
    tok = jnp.take_along_axis(logp_vocab[:, :L], jnp.asarray(COMPLETIONS)[..., None], -1)[..., 0]
    return tok.sum(-1)


def sampler_table(theta, scale, key, scenario):
    """Sampler logits per prefix: theta plus noise. For topk, q outside its
    top-k is made proportional to p."""
    noise = scale * jax.random.normal(key, theta.shape)
    q_logits = theta + noise
    if scenario == "topk" and scale > 0:
        logp = jax.nn.log_softmax(theta)
        logq = jax.nn.log_softmax(q_logits)
        order = jnp.argsort(-logq, -1)
        head = jnp.zeros_like(logq, bool).at[jnp.arange(D)[:, None], order[:, :K]].set(True)
        q, p = jnp.exp(logq), jnp.exp(logp)
        # head keeps the noisy sampler's shape; tail is proportional to p with
        # mass m, capped so that no tail entry outranks the smallest head entry
        q_head = jnp.where(head, q, 0)
        q_head = q_head / q_head.sum(-1, keepdims=True)
        p_tail = jnp.where(head, 0, p)
        p_tail = p_tail / p_tail.sum(-1, keepdims=True)
        a = jnp.where(head, q_head, jnp.inf).min(-1, keepdims=True)
        b = p_tail.max(-1, keepdims=True)
        m = jnp.minimum(jnp.where(head, 0, q).sum(-1, keepdims=True), 0.9 * a / (a + b))
        q = jnp.where(head, (1 - m) * q_head, m * p_tail)
        # keep the head the head: tail entries must not exceed the smallest head entry
        min_head = jnp.where(head, q, jnp.inf).min(-1, keepdims=True)
        assert bool((jnp.where(head, -jnp.inf, q) <= min_head + 1e-7).all()), "re-draw the sampler"
        q_logits = jnp.log(q)
    return q_logits


@dataclass
class Case:
    theta: jax.Array
    q_logits: jax.Array
    scenario: str


def make_batch(case, advantages):
    logq_vocab = table_logprobs(case.q_logits)  # [N, T, V]
    targets = jnp.roll(jnp.asarray(TOKENS), -1, 1)
    tok = jnp.take_along_axis(logq_vocab, targets[..., None], -1)[..., 0]
    tok = jnp.where(jnp.roll(jnp.asarray(MASK), -1, 1).at[:, -1].set(False), tok, 0.0)
    batch = {"tokens": jnp.asarray(TOKENS), "mask": jnp.asarray(MASK),
             "advantages": jnp.asarray(advantages, jnp.float32), "sampling_token_logprobs": tok}
    if case.scenario == "full":
        batch["sampling_vocab_logprobs"] = logq_vocab
    else:
        top_lp, top_ids = jax.lax.top_k(logq_vocab, K)
        batch |= {"sampler_topk_ids": top_ids, "sampler_topk_logprobs": top_lp}
    return batch


def per_sample_grads(loss_fn, case, advantages):
    """[N, D, V] gradient of the loss on each completion alone (batch of 1)."""
    batch = make_batch(case, advantages)

    def one(i):
        b = {k: v[i][None] for k, v in batch.items()}
        h = jnp.asarray(HIDDEN)[i][None]
        f = lambda th: loss_fn(h, lambda x: x @ th, b)[0]
        return jax.grad(f)(case.theta)

    try:
        return jax.vmap(one)(jnp.arange(N))
    except Exception:  # loss not vmappable: fall back to a loop
        return jnp.stack([one(i) for i in range(N)])


def q_weights(case):
    return jnp.exp(sequence_logprob(table_logprobs(case.q_logits)))


def expected(grads, weights):
    return jnp.tensordot(weights, grads, 1)


def reinforce(hidden, head, batch, *_, **__):
    """Plain REINFORCE, the reference point for on_policy and bounded."""
    logp = jax.nn.log_softmax(head(hidden).astype(jnp.float32))
    targets = jnp.roll(batch["tokens"], -1, 1)
    mask = jnp.roll(batch["mask"], -1, 1).at[:, -1].set(False)
    tok = (jnp.take_along_axis(logp, targets[..., None], -1)[..., 0] * mask).sum(-1)
    return -(batch["advantages"] * tok).sum() / jnp.maximum(mask.sum(), 1), {}


def rewards(key):
    return jax.random.bernoulli(key, 0.4, (N,)).astype(jnp.float32)


def run_checks(loss_fn, scenario):
    res = {"no_drift": [], "no_drift_prefix": [], "on_policy": [], "bounded": [], "alignment": []}
    for seed in SEEDS:
        k_theta, k_mild, k_severe, k_r = jax.random.split(jax.random.key(seed), 4)
        theta = jax.random.normal(k_theta, (D, V))
        r = rewards(k_r)

        # no_drift: constant advantage (both signs) under a mild mismatch
        case = Case(theta, sampler_table(theta, MILD, k_mild, scenario), scenario)
        w = q_weights(case)
        for a in (1.0, -1.0):
            g = per_sample_grads(loss_fn, case, jnp.full(N, a))
            scale = expected(jnp.sqrt((g ** 2).sum((1, 2)))[:, None, None] * jnp.ones_like(g), w)[0, 0]
            res["no_drift"].append(float(jnp.linalg.norm(expected(g, w)) / jnp.maximum(scale, 1e-12)))

        # no_drift_prefix: advantage g(y_1..y_d) centered under q; rows of
        # theta for prefixes of length >= d are only touched after position d
        for depth in (1, 2):
            k_g = jax.random.fold_in(k_r, depth)
            table = jax.random.normal(k_g, (V,) * depth)
            g_adv = table[tuple(COMPLETIONS[:, i] for i in range(depth))]
            g_adv = g_adv - (w * g_adv).sum() / w.sum()
            g = per_sample_grads(loss_fn, case, g_adv)
            rows = jnp.asarray(PREFIX_LEN >= depth)
            g = jnp.where(rows[None, :, None], g, 0.0)
            scale = (w * jnp.sqrt((g ** 2).sum((1, 2)))).sum()
            res["no_drift_prefix"].append(float(jnp.linalg.norm(expected(g, w)) / jnp.maximum(scale, 1e-12)))

        # alignment: non-constant rewards, mild mismatch, against the true gradient
        g = expected(per_sample_grads(loss_fn, case, r), w)
        true = jax.grad(lambda th: -(jnp.exp(sequence_logprob(table_logprobs(th))) * r).sum())(theta)
        res["alignment"].append(float((g * true).sum() / (jnp.linalg.norm(g) * jnp.linalg.norm(true) + 1e-12)))

        # on_policy: q = p, non-constant rewards
        on = Case(theta, theta, scenario)
        w_on = q_weights(on)
        g_sub = expected(per_sample_grads(loss_fn, on, r), w_on)
        g_ref = expected(per_sample_grads(reinforce, on, r), w_on)
        res["on_policy"].append(float(jnp.linalg.norm(g_sub - g_ref) / (jnp.linalg.norm(g_ref) + 1e-12)))

        # bounded: severe mismatch, worst single-sample gradient
        severe = Case(theta, sampler_table(theta, SEVERE, k_severe, scenario), scenario)
        g_sub = per_sample_grads(loss_fn, severe, r * 2 - 1)
        g_ref = per_sample_grads(reinforce, severe, r * 2 - 1)
        res["bounded"].append(float(jnp.sqrt((g_sub ** 2).sum((1, 2))).max() / jnp.sqrt((g_ref ** 2).sum((1, 2))).max()))

    passed = {
        "no_drift": max(res["no_drift"]) < DRIFT_TOL,
        "no_drift_prefix": max(res["no_drift_prefix"]) < DRIFT_TOL,
        "on_policy": max(res["on_policy"]) < ON_POLICY_TOL,
        "bounded": max(res["bounded"]) < BOUND_FACTOR,
    }
    finite = all(np.isfinite(v).all() for v in res.values())
    return {"passed": finite and all(passed.values()), "checks": passed,
            "worst": {k: (max(v) if k != "alignment" else min(v)) for k, v in res.items()}}


def evaluate(loss_fn):
    """Run every check in every scenario. loss_fn(hidden, head, batch) -> (loss, metrics)."""
    out = {}
    for scenario in SCENARIOS:
        try:
            out[scenario] = run_checks(loss_fn, scenario)
        except Exception as e:  # a loss that crashes on this data layout fails the scenario
            out[scenario] = {"passed": False, "error": f"{type(e).__name__}: {e}"[:500]}
    return {"passed": any(s["passed"] for s in out.values()), "scenarios": out}


def load_loss(path):
    spec = importlib.util.spec_from_file_location("submission_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda hidden, head, batch: module.loss_fn_rl(hidden, head, batch)


if __name__ == "__main__":
    result = evaluate(load_loss(sys.argv[1]))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)

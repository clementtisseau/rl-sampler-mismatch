# Sampler mismatch in REINFORCE: a score-centering control variate

## What is wrong with the current loss

`loss_fn_rl` estimates the policy gradient with samples from the sampler q,
but differentiates the trainer's log p. For a completion x = (x_1..x_T) with
advantage A (sequence-level), and with correction None (w = 1), the
per-position score is

    g = sum_t A * grad log p(x_t | h_t),     x_t ~ q(. | h_t).

Write b_t = E_{y~q(.|h_t)}[grad log p(y|h_t)] and V_t = E[A | h_t] (the value
of the prefix under the sampler). Then, exactly,

    E_q[g] = E_q[ sum_t (A - V_t) grad log p(x_t) ]  +  E_q[ sum_t V_t b_t ].

The first term is the usual advantage-score covariance, computed on sampler
rollouts. The second term is the problem. Under the trainer, b_t = 0 for every
prefix, since E_p[grad log p] = sum_y p(y) grad log p(y) = grad sum_y p(y) = 0.
So the term E_p[sum_t V_t b_t] vanishes identically and does not belong in the
gradient at all. Under mismatch it is E_q[sum_t V_t b_t] != 0, because
b_t = sum_y q(y|h_t) grad log p(y|h_t), which is nonzero whenever q != p.

In logit space, for a softmax head z with p = softmax(z),

    grad_z log p(y) = e_y - p,       so   E_q[grad_z log p(x)] = q - p.

The current loss therefore adds a bias A_t (q - p) to every position's logit
gradient. For constant advantages this is a pure push of p toward q, the
sampler's distribution, and the sampler is then refreshed from the trainer.
The offset is recreated each step and the push has the same sign every step,
so it accumulates instead of averaging out. This matches the known facts: the
mismatch is a systematic bias, and it compounds.

Measured at initialization (mini setting, 4096 rollouts, noise 0.6): the
gradient norm with constant advantage A = 1 is 2.12 for the current loss and
0.12 for the fix below. The first number is pure drift.

## Why importance sampling only slows it

Full token-level IS with w = p/q removes the drift exactly:
E_q[(p/q)(e_x - p)] = sum_x p(x)(e_x - p) = 0. But w is unbounded, and its
variance grows with the log-ratio spread, which is large under noise.
Clipping at c (TIS, MIS, PPO) bounds the variance but reintroduces a bias,
E_q[(1 - min(p/q, c))(e_x - p)] != 0. This pushes toward q again, which is why
stronger clipping can make things worse. Self-normalizing the weights (w / mean w)
bounds the variance and removes the drift only up to O(1/n) bias.

## Fix: subtract the sampler-expected score (control variate)

Use a per-position baseline that has zero mean under q, and that equals the
drift term to be removed. The sampler's expected score is

    b_t = sum_y q(y|h_t) grad log p(y|h_t),

which is computable from the sampler's top-K logprobs and the trainer's log p
over the same K ids. Since sum_y p grad log p = 0, the unseen tail can be
approximated by the trainer's tail, q_tail/p_tail. Let

    kappa_t = (1 - sum_{y in K} q(y|h_t)) / (1 - sum_{y in K} p(y|h_t)),
    c_y     = q(y|h_t) - kappa_t * p(y|h_t)   (y in K, treated as constant),
    b_t     = sum_{y in K} c_y grad log p(y|h_t).

When K is the whole vocabulary, kappa = 0 and b_t is exact. The modified loss
is

    loss = - sum_t A * [ log p(x_t|h_t) - sum_{y in K} c_y log p(y|h_t) ] / N_tokens,

which for w = 1 gives the score A * (grad log p(x_t) - b_t).

Why this is the right term to remove. The baseline subtracts E_q[sum_t V_t b_t],
the value-weighted drift. Because V_t and b_t depend only on the prefix h_t,
E_q[(A - V_t) b_t] = E_q[b_t E[A - V_t | h_t]] = 0, so the expectation of the
new estimator is

    E_q[ sum_t (A - V_t) grad log p(x_t|h_t) ],

which is the advantage-score covariance on sampler rollouts with the drift
removed. The remaining bias is only the signal mismatch between q and p in
that covariance. It is the part a token-level IS weight would correct, and it
did not cause collapse in the tests below.

Stability test (`dev/mini.py`, plus_minus advantages, weight noise 0.3, 40
extra vocabulary tokens, 600 steps, final-20-step training reward):

| method | seed 0 | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- | --- |
| current loss (pg) | 0.00 | 0.00 | 0.00 | 0.00 |
| naive token IS | 0.54 | 0.56 | 0.00 | 0.49 |
| self-normalized token IS | 0.61 | 0.56 | 0.47 | 0.62 |
| score centering, full-vocab baseline | 0.82 | 0.86 | 0.76 | 0.82 |
| score centering, tail-corrected (submitted) | 0.86 | 0.82 | 0.73 | 0.84 |

The tail-corrected version also held at 200 extra tokens (0.83, 0.82,
versus 0.69, 0.77 for the full-vocabulary baseline without the tail correction), where the top-20 support
covers little of the vocabulary. At weight noise 0.6 it trains to 0.41 and
0.40, with a large KL between sampler and trainer (9 to 13 nats), so the fix
removes the drift but not the mismatch itself.

## Scope

- The derivation covers token-level scores with a fixed sampler q. It holds
  for any sampler distribution, whatever its source (weight noise, quantization,
  staleness), because it uses only q(y|h_t) over the logged top-K.
- When a token-level IS correction is active (`rl.is.level=token`), the
  weight already removes the drift. The baseline is then switched off, so
  those settings keep their old behaviour. The baseline applies only when
  no token-level correction is present, which covers the default
  `correction=None` and sequence-level corrections.
- No configuration changes are needed.
- Mini results are a proxy for the Qwen run; only weight-noise mismatch was
  tested end-to-end. Staleness and quantized KV cache were not tested.

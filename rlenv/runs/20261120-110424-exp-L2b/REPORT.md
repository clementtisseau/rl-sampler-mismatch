# Sampler-mismatch bias in REINFORCE, and a sampler-expected score baseline

## 1. What is wrong with the current loss

With `correction=None`, `loss_fn_rl` uses the per-token term `w * logp` with `w = 1`, so each completion token
contributes `-A * log p(x_t | h_t)`, where `h_t` is the prefix, `A` the (sequence) advantage, and the token
`x_t ~ q` is drawn from the sampler, not from the trainer `p = p_theta`.

Fix a position, a prefix `h`, and a sequence advantage `A` (constant at this position, for the moment).
Let `Qbar(x) = E[A | h, x_t = x]` be the advantage conditional on the next token. The expected gradient of the
loss over the next token drawn from q is

    g_pg(h) = E_{x~q}[ grad(-A log p(x)) ] = -sum_x q(x) Qbar(x) grad log p(x).

On-policy (q = p) the target is `-sum_x p(x) Qbar(x) grad log p(x)`. The sampler-expected score vanishes,
`sum_x p(x) grad log p(x) = grad sum_x p(x) = 0`, so a constant part of the advantage contributes nothing
on-policy. Under mismatch it does not vanish. Decompose `Qbar(x) = Qbar_bar + (Qbar(x) - Qbar_bar)` with
`Qbar_bar = sum_x q(x) Qbar(x)`. The constant part gives

    -Qbar_bar * sum_x q(x) grad log p(x) = Qbar_bar * grad KL(q || p_theta),

using `sum_x q(x) grad log p(x) = -grad KL(q || p_theta)` (q does not depend on theta). So the current loss has

    E_q[grad loss_pg] = Qbar_bar * grad KL(q || p_theta)  -  sum_x (q(x) - p(x)) (Qbar(x) - Qbar_bar) grad log p(x)  + (on-policy target)

Only the first term is a pure artifact of the mismatch. Each step it pulls the trainer toward the sampler's
distribution, with strength equal to the mean advantage at that prefix. Rewards here are +/-1 (`plus_minus`),
and with group-centering the mean is zero only across the group's first token, not at later prefixes. So the
drift does not cancel. The sampler is then refreshed from the trainer, so the bias is applied again and
compounds. Weight-noise, low-precision, KV-quantized and stale samplers all make `q != p` in this way.

Why the importance-sampling corrections only slow it down: the token-level weight `w = p/q` makes the expectation
exact at one token (`E_q[(p/q) grad log p] = sum_x p grad log p = 0`), which removes the drift term above. But
(i) `p/q` is unbounded and clipping (TIS, PPO, DAPO) reintroduces a bias that is larger when clipping is
stronger, which matches the observation that stronger clipping can make things worse, and (ii) `Qbar` is
computed from completions whose later tokens are also drawn from q. Token-level IS does not correct that.

## 2. The modification

Subtract from the score the sampler-expected score, computed from the sampler's own distribution at the same
position. The batch already carries the sampler's top-K logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`,
top-20 per position, which the loss previously ignored). Let `qt(y) = q(y) / sum_{y' in K} q(y')` for y in the
top-K, and define the baseline

    b(h) = sum_{y in K} qt(y) log p(y | h)          (qt is stop-gradient).

The per-token term becomes `w * (log p(x_t | h) - b(h))`, with `w = 1` when no correction is given. The loss is
then `-A * (log p(x_t) - b(h))` per token, so the gradient is

    grad loss_new = -A * ( grad log p(x_t) - sum_{y in K} qt(y) grad log p(y) ).

Expectation over `x_t ~ q` with a constant advantage:

    E_q[grad loss_new] = -A * ( sum_x q(x) grad log p(x) - sum_{y in K} qt(y) grad log p(y) )
                       = A * ( grad KL(q || p_theta) - grad KL(qt_K || p_theta) ),

where `qt_K` is qt extended by zero outside K. This is zero when the sampler's top-K captures q (qt = q), which is
the case for the ~20 most likely tokens that carry nearly all the sampler mass. In the on-policy case q = p, the
baseline term is `sum_{y in K} (p(y)/Z) grad log p(y) = grad log Z` with `Z = sum_{y in K} p(y)`. This is zero
when the top-K holds all the mass (Z = 1), and otherwise is of the size of the tail mass outside K. So the
estimator reduces to the original REINFORCE term on-policy, up to that truncation.

With varying advantages the same algebra gives

    E_q[grad loss_new] = -sum_x q(x) (Qbar(x) - Qbar_bar) grad log p(x)   (when qt = q),

which is the REINFORCE gradient under q with a per-position mean-centered advantage. The drift term `Qbar_bar *
grad KL(q||p)` is removed exactly. What remains is `-sum_x (q - p)(Qbar - Qbar_bar) grad log p`, a term that is
proportional to the mismatch multiplied by the reward variation over tokens at that position. It is the
reward-informed part of the mismatch, and it does not accumulate as a constant pull toward the sampler.

Why it is only applied without a correction: with an IS weight `w(x) = p/q`, the baseline would be
`sum_K qt(y) w(y) log p(y)`, and the expectation of the weighted score is already zero on its own
(`E_q[w grad log p] = 0`). Using the unweighted baseline with IS gives `E_q[w (grad log p - grad b)] = -grad b`,
which is a new bias. I tested this combination and it collapsed. So the baseline is used when `correction is None`
(the default), and any `rl.is.*` correction keeps its original weighted term.

The modification needs no configuration changes. `losses.py` uses the sampler top-K already present in the batch
(`train_rl.py` passes `sampler_topk_ids` and `sampler_topk_logprobs`, `logprobs_topk=20`). No `## Configuration`
keys are required.

## 3. Derivation check

A toy check with a 6-token vocabulary, a linear head and a q perturbed from p (q's top-K covering all tokens,
constant A = 1) gave:

- original loss: `E_q[grad loss]` equals `A * grad KL(q||p)` to 3e-8 (norm 0.226), confirming the drift term.
- new loss: `||E_q[grad loss]|| = 1.6e-8`, i.e., zero to numerical precision.

## 4. Empirical sanity check (CPU mini setting, not required, small samples)

`dev/mini.py`, sampler noise 0.3 with 200 extra vocabulary tokens, 600 steps, final eval reward:

| loss | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| original, no correction | 0.00 | 0.10 | 0.00 |
| token IS (`rl.is.level=token`) | 0.72 | 0.75 | 0.21 |
| new (baseline, no correction) | 0.78 | 0.78 | 0.71 |

Other checks: noise 0.45 (seed 0): original 0.00, new 0.45. Noise 0.3 without extra tokens (seed 0): original
0.66, new 0.84. Baseline combined with token IS: 0.00 on all three seeds (the bias described above).

These are small runs and should be read as a sanity check, not as a benchmark.

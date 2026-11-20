# Sampler mismatch in REINFORCE: a sampler-centred score

## What is wrong with the current loss

Per prefix h, let p(.|h) be the trainer and q(.|h) the sampler. The uncorrected loss
(`correction=None`, w = 1) has per-token term A * log p(x_t|h_t), with A the
group-centred advantage of the sequence. Write g(h,x) = E[A | h, x_t = x] for the
expected advantage of continuing with token x. Its expected gradient under the
sampler is

    G_q(h) = sum_x q(x) g(h,x) grad log p(x|h).

On-policy (q = p) this is the policy gradient G_p(h) = sum_x p(x) g(h,x) grad log p(x|h).
The mismatch adds

    G_q(h) - G_p(h) = sum_x (q(x) - p(x)) g(h,x) grad log p(x|h).

Split g = gbar + (g - gbar), with gbar = E_q[g(h,.)]. The constant part gives

    gbar * sum_x (q - p)(x) grad log p(x|h) = gbar * grad_theta [ sum_x q(x) log p(x|h) ] = -gbar * grad CE(q, p).

On-policy this term is zero, because sum_x p grad log p = grad sum_x p = 0. Under
mismatch it is not zero. It is a likelihood step on the sampler's own tokens, scaled by
the advantage: with gbar > 0 it imitates the sampler, with gbar < 0 it moves away from
it. In logit space the expected update of a context is gbar (q - p). Every step the
trainer is pulled toward its noisy or quantized sampler, and the sampler is refreshed
from the trainer, so the bias accumulates. This is the systematic (not zero-mean) bias
described in the task. Group centring removes gbar only in the sample mean over a
group, not per prefix, so it does not cancel this term.

IS clipping and dropping fail for the same reason. The exact fix is w = p/q: then
E_q[(p/q) grad log p] = sum_x p grad log p = 0, so the constant part vanishes. Clamping
the weight at c replaces p/q by min(p/q, c). The residual is

    sum_x (q min(p/q, c) - p) gbar grad log p = -gbar * sum_{x: p > c q} (p - c q) grad log p.

This is biased on the tokens where the trainer puts much more mass than the sampler
does. The bias grows as c shrinks, which matches "stronger clipping can make it worse".

## Modification: a sampler-centred score

For each position the sampler returns its top-K logprobs (K = 20) on the same ids
(`sampler_topk_ids`, `sampler_topk_logprobs`). Let d_k = q_k - p_k on those ids, with d_k
detached. Replace the score log p(x_t) by

    s(h, x) = log p(x|h) - sum_{k in K(h)} d_k log p(k|h).

Its gradient is grad log p(x) - sum_{k in K} (q_k - p_k) grad log p(k). The term is a
control variate. It is exactly zero when q = p, so the on-policy estimator is unchanged.

Expected value under q. Using sum_x p(x) grad log p(x) = 0 and d_k = (q - p)(k) on K,

    E_q[grad s] = sum_x q(x)(e_x - p) - sum_{k in K} (q_k - p_k)(e_k - p)
                = (q - p) restricted to the tail (outside K)  +  (Q_K - P_K) p,

where Q_K and P_K are the total mass of q and p on K. This follows from
sum_x (q - p)(x) grad log p(x) = sum_x (q - p)(x) (e_x - p). Under constant advantage
the mismatch-induced term is therefore reduced from the full (q - p) to its tail part
plus a term that vanishes when the head masses agree. The residual is driven only by
mismatch outside the sampler's top-20. I checked the identity numerically on a random
12-token distribution with K = 4 (max error about 1e-16).

With non-constant advantage, using gbar = E_q[g] and B = sum_{k in K} d_k grad log p(k),

    E_q[g s] - E_p[g grad log p]
      = sum_x (q - p)(x) (g(x) - gbar) grad log p(x)  +  gbar * sum_{x not in K} (q - p)(x) grad log p(x).

The first term is the reward-correlated part of the mismatch, which is what IS
corrections address. The second is the tail-only constant part. The original loss
had the full constant part gbar * sum_x (q - p) grad log p in its place.

## Why this stabilises training

The advantage-independent pull toward q is the feedback loop that drives collapse.
The modified score removes its head component exactly. What is left is either
reward-correlated (and so acts like a real policy-gradient signal) or lives on tokens
the sampler ranks outside the top 20. Those tokens have small mass in a typical LLM
(assumed, not measured here), so the feedback loop is weakened. The loss does not depend on how q was produced,
so it covers noise, int8/fp8 weights, KV quantization and staleness alike, as long as
the sampler's top-K logprobs are returned.

## Results (mini setting, illustrative only)

Mini setting, weight noise 0.3, 600 steps, 300 extra unlikely vocabulary tokens
(MINI_EXTRA_TOKENS=300). Last-10 mean training reward, original vs modified:
seed 0: 0.08 vs 0.67; seed 1: 0.00 vs 0.74; seed 2: 0.00 vs 0.78; seed 3 (modified): 0.70.
Noise 0.5 with no extra tokens, seed 1: 0.00 vs 0.53. With noise 0 the two losses
give identical trajectories, as the derivation predicts.

The modified loss still trains with the IS correction path (level=token, high=2.0,
outside=clamp) without error, but the control variate is disabled there (see below).
kl_samp_train (mean of log q - log p on sampled tokens) grows under the modified loss
(0.7 to 7.7 at the end of the runs above). That metric is not a stability measure, but
it means the trainer moves away from the sampler, so it should be watched in real runs.

## Configuration

No configuration change is needed. The modification is active under the default
arguments (`rl.is.level` null).

## Limits

- The control variate is applied only when `correction` is None. With a token or
  sequence IS weight w = p/q, E_q[w (grad log p - B)] = -B, so adding B there would bias
  the estimator. The IS weight already removes the constant part in expectation.
- It needs the sampler's top-K logprobs. Without them the loss falls back to the
  plain score.
- The tail part of the mismatch is not corrected. If the sampler's mass outside its
  top-K is large, the residual grows.

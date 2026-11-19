# Training under sampler/trainer mismatch

## What is wrong with the current loss
Rollouts are drawn from the sampler q, but the default loss (`correction=None`) is
L = -sum_t A * log p(a_t), i.e. plain REINFORCE as if the samples came from the trainer p.
Its expected gradient per token (given the prefix, advantage A) is

  E_{a~q}[A * grad log p(a)] = sum_a q(a) A grad log p(a).

For on-policy samples (q = p) the identity sum_a p(a) grad log p(a) = grad sum_a p(a) = 0 makes any
baseline or constant part of A drop out. With q != p it does not:

  sum_a q(a) grad log p(a) = sum_a (q(a) - p(a)) grad log p(a)  != 0.

So the gradient is  E_q[A grad log p] = E_q[(A - Abar) grad log p] + Abar * sum_a (q-p) grad log p,
with a systematic, non-zero-mean bias term b = Abar * sum_a (q-p) grad log p. Even with group-centered
advantages the per-token (conditional on prefix, and correlated with the token) mean of A is not zero,
and the term sum_a (q-p) grad log p is a deterministic function of the mismatch (q-p), not noise: it
does not average out over batches. Concretely, noise/quantization flattens the sampler (more mass on
the many unlikely tail tokens: q_tail > p_tail). Sampled tail tokens mostly belong to failed rollouts
(A<0); the update pushes their log p down, with large per-token score norm (|grad log p| ~ 1 for rare
tokens), and the induced shift in p changes the next mismatch. Since the sampler is refreshed from the
trainer each step, the bias is re-created every step and compounds, which ends in collapse (reproduced
in dev/mini.py with MINI_EXTRA_TOKENS=200: reward goes to 0 within 25 steps).

## Fix
Use the exact importance weight on the score function, unclipped, by default:

  L = - (1/N) sum_{i,t} A_i * sg(w_t) * log p(a_t),   w_t = p(a_t)/q(a_t).

Then  E_{a~q}[ w A grad log p ] = sum_a q (p/q) A grad log p = sum_a p(a) A grad log p(a)
= E_{a~p}[A grad log p]: the true policy gradient of the trainer, for any q with support covering p.
The bias term vanishes identically (sum_a p grad log p = 0), including the Abar part, so nothing
systematic is fed back into the next iteration and nothing compounds. (Equivalently w grad log p =
grad p / q: the gradient of the probability ratio.)

Why not clipping/dropping: truncating w to [lo, hi] or zeroing out-of-band tokens replaces w by w' with
E_q[w' grad log p] = sum_a q w' grad log p != 0. The tokens clipped are exactly the ones where p and q
differ most (the tail), so the residual is a systematic bias of the same form as the original one,
and tighter clipping makes it larger. Hence clipping slows but does not remove the degradation, and
stronger clipping can worsen it. The only truncation I keep is a far safety clamp w <= 20 (ratio e^3),
reached essentially never for realistic mismatch, which only guards against numerical blow-up; its bias
is of order the mass of tokens with p/q>20.

Variance: per-token ratios are close to 1 for mild mismatch, so variance is small; the per-token (not
per-sequence) weight is used because the product of T ratios would have exponential variance. The
prefix-distribution mismatch (states visited under q vs p) is ignored; it is a second-order effect
compared with the first-order token-level bias removed above.

Implementation: `loss_fn_rl` with `correction=None` uses `DEFAULT_CORRECTION =
ISCorrection(level="token", high=20, outside="clamp")`; an explicitly passed correction is honored.
Empirically in dev/mini.py (200 extra tokens) the default no longer collapses.

## Configuration
rl.is.level=token
rl.is.high=20.0
rl.is.outside=clamp

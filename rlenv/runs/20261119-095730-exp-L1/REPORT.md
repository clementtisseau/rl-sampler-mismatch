# Sampler/trainer mismatch in REINFORCE

## What is wrong with the current loss
Rollouts are drawn from q (the sampler), but the loss `-A * sum_t log p(x_t)` is
the REINFORCE estimator for samples from p (the trainer). With default arguments
(`correction=None`) it applies no correction at all. Its expected per-token
gradient is

  E_{x~q}[A * grad log p(x)] = sum_x q(x) A(x) grad log p(x).

The correct gradient (for the sampled prefix) is sum_x p(x) A(x) grad log p(x).
Write q = p * r^-1, with r = p/q. The error is

  sum_x (q(x) - p(x)) A(x) grad log p(x).

Using sum_x p grad log p = 0, the part of A that is constant over x at a state
drops out under p, but under q it gives the term E_q[grad log p] =
-grad CE(q, p) != 0. This is a systematic "fit p to q" (A>0) or "push p away
from q" (A<0) force, not zero-mean noise: it has a consistent direction that
depends on q - p. Group centering cancels it only when A is uncorrelated with the
token distribution, which is false (the early tokens predict the reward).
Because the sampler is rebuilt from the trainer at each step (q_t = p_t + perturbation),
the bias is applied at every step and accumulates: p is dragged toward the
sampler's distortion (e.g. tail mass, quantisation artifacts), which is then
copied again into the next sampler.

## The fix
Reweight each token by the exact importance ratio r_t = p(x_t)/q(x_t) (detached):

  L = -(1/N) sum_i A_i sum_t sg(r_it) log p(x_it),   grad L = -(1/N) sum A_i sum_t grad p / q.

Then E_{x~q}[r grad log p] = sum_x p grad log p = the on-policy score-function
gradient: the bias term above is exactly cancelled, for any q with support
covering p (any weight, noise, int8/fp8, KV quant, staleness). It uses only the
recorded sampled-token logprob, so it works with `correction=None`.
This is the default now (`DEFAULT_CORRECTION`: token level, weights clamped at 10).

Why clamp at all, and why loosely: the ratio has heavy tails (tokens that q
almost never emits but p likes), so the weight is truncated only at a bound
(10) to avoid blowups. Clamping at C replaces r by min(r,C) and reintroduces a
bias equal to sum_{r>C} (p - C q) A grad log p, whose size is the p-mass on
r>C. This is small for C large and grows as C shrinks, and the clamp is
asymmetric (it only removes the high-ratio tokens that the sampler under-
represents), so tight clipping/dropping (PPO-like bands, MIS) re-creates a
directional bias, which is why stronger token-level clipping makes things worse. In
the CPU dev runs, TIS with high in {1, 2, 5, 20} gave final clean-eval reward
roughly 0.785, 0.83, 0.85, 0.845 (monotone-ish in looseness) versus about 0.66-0.70 with no
correction; sequence-level ratios (seq_sum) had too much variance, and unclamped
policy-gradient decayed.

## Limitation
Token-level weights correct the per-state action distribution, not the shift in
the visited prefixes (that requires the product of ratios, whose variance grows
exponentially with length). The remaining bias is second order in the mismatch.

## Configuration
rl.is.level=token
rl.is.high=10.0
rl.is.outside=clamp

# Report

## What is wrong
Fix a prefix, let p be the trained policy and q the sampler, x~q, advantage A.
The loss -A log p(x) has expected gradient
  g = E_q[A ∇log p(x)] = Σ_x q(x) A(x) ∇log p(x).
Write A(x) = b + (A(x) − b), where b is the part of A not tied to the token
(group-centering baseline, sequence-level credit, etc.). The baseline part gives
  b · Σ_x q(x) ∇log p(x).
On-policy (q=p) this is b·Σ ∇p = 0, which is why baselines are free in REINFORCE.
With q≠p, in logit space ∇_z log p(x) = e_x − p, so the term is b·(q − p) ≠ 0.
It is a systematic drift: for b>0 it pulls p toward q, for b<0 (failed rollouts, the
majority of the mass late in training) it pushes p away from q, i.e. it moves
probability off the sampler's high-mass tokens. This is first-order in the
mismatch, not zero-mean noise, and because the sampler is refreshed from p each
step the push compounds (entropy drift, collapse). Group centering makes
Σ_group A=0 only across a group, not per prefix and per token, so it does not cancel it.

## Why clipping/dropping don't fix it
With a weight w(x) the bias term is b·Σ_x q(x)w(x)∇log p(x) = b·Σ_x m(x)∇log p(x), m=qw.
This vanishes for every b and every p only if m ∝ p, i.e. w = p/q (up to a
prefix constant): then Σ_x p ∇log p = Σ ∇p = 0. Any clamp or drop (TIS, MIS, PPO, DPPO..)
makes m ≠ p precisely on the tokens where p and q differ most, re-introducing
b·Σ m∇log p ≠ 0; token-level tightening of the band makes m deviate further,
hence worse. With no correction w=1, m=q.

## Modification
The default loss (correction=None) now uses the exact per-token weight w=p/q,
detached (ratio clipped only at e^±20 for overflow). Then
  E_q[A w ∇log p] = Σ_x p(x) A(x) ∇log p(x) = E_p[A ∇log p],
the on-policy policy gradient, and the baseline term is exactly zero
(Σ_x ∇p(x)=0) for any baseline. No clipping or dropping is applied, so no
bias is reintroduced; bias is removed rather than traded for variance.
Explicit corrections passed in still work as before.

## Configuration
rl.is.level=token
rl.is.low=0
rl.is.high=inf
rl.is.outside=clamp
rl.is.sign=both

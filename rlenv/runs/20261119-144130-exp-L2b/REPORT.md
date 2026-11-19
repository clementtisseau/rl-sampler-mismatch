# Sampler/trainer mismatch in REINFORCE

## What is wrong
Fix a prefix and one position. Let p be the trainer's next-token distribution, q the sampler's, z the logits, and A the advantage.
The current loss (`correction=None`) has expected update
g = E_{x~q}[A(x) ∇log p(x)].

Write A = (A - b) + b, where b is a per-prefix constant. Group-centering exists to subtract such a constant baseline.

- **On-policy (q = p).** E_p[∇log p] = Σ_x ∇p(x) = ∇1 = 0. Any baseline b contributes nothing, so g is the unbiased policy gradient.
- **Mismatched (q ≠ p).** E_q[∇log p] = Σ_x q(x) ∇log p(x) ≠ 0.
  In logit space ∇_z log p(x) = e_x − p, so E_q[∇_z log p] = q − p ≠ 0.
  The baseline term then contributes b·(q − p). Equivalently, it is the gradient of b·E_q[log p] = −b·CE(q, p), and it pulls p toward q whenever b > 0, or pushes it away when b < 0.

Group-centered advantages are not zero for each prefix, and they have different signs for different samples, so this term never averages out. It is a systematic bias, not zero-mean noise. It also depends on the sign of the advantage in a way the reward signal doesn't control. It compounds because the sampler is re-derived from the trainer every step, so q moves with p and the drift keeps pushing.
Clipping or dropping tokens (TIS, MIS, PPO, DPPO, and so on) uses a weight w̃(p/q) ≠ p/q. Then E_q[w̃ ∇log p] = Σ_x q w̃ ∇log p, which is no longer 0. The baseline bias is only rescaled, not removed. Tighter clipping gives a larger departure from p/q, so more bias, which is consistent with the observation that stronger token-level clipping can make things worse.

## Fix
Use the exact per-token importance weight w = p/q, detached and unclipped, by default (`correction=None` now means `ISCorrection(level="token")`, with no band):
loss = −Σ A · sg(p/q) · log p.
Its expected per-prefix gradient is
E_q[A (p/q) ∇log p] = E_q[A ∇p/q] = Σ_x A(x) ∇p(x).
For constant A = b this is b·∇Σ_x p(x) = 0 for any q. The baseline is therefore exactly neutral again, as it is on-policy. For general A it equals the on-policy policy gradient Σ_x A(x) ∇p(x), whatever q is. The mismatch bias in the expected update is removed, rather than rescaled.
The only residual is variance (weights p/q can be large). The existing clip of the log-ratio to ±20 only guards numerical overflow. The prefix distribution is also drawn from q, which the per-position analysis conditions out.

## Configuration
rl.is.level=token

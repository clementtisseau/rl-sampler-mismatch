# Sampler/trainer mismatch in REINFORCE

## What is wrong
Fix a prefix and a position. Let p be the trainer's next-token distribution, q the sampler's, z the logits, A the advantage. The loss is -A log p(a), a ~ q, with gradient wrt logits A (e_a - p). Decompose A = Abar + (A - Abar), where Abar is the expected advantage given the prefix (the value of the prefix under the sampler).

Expected update under q:
E_q[A ∇_z log p(a)] = Abar (q - p) + Cov-term (the true policy-gradient signal, which is wanted).

On-policy (q = p), E_p[∇ log p] = sum_a ∇p(a) = 0, so Abar contributes nothing: the score function has zero mean, and this is why REINFORCE needs no baseline in expectation. When q != p, E_q[∇ log p] = sum_a q(a)∇log p(a) = -∇_θ CE(q,p) ≠ 0. The term Abar (q-p) is a systematic push that is independent of the rewards' informativeness: for Abar<0 (prefixes in unsolved-heavy groups, and group-centering only zeroes the mean over the group, not per prefix) it raises the loss CE(q,p), i.e. moves p away from q and flattens/randomizes the distribution (mass is taken from tokens the sampler favours); for Abar>0 it sharpens p toward q. Because the sampler is refreshed from the trainer, q tracks p, the bias is re-created and accumulated each step (a drift, not zero-mean noise), producing entropy blow-up and collapse. Clipping/masking the IS ratio only changes the weights w(a) so that E_q[w(a)(e_a-p)] = sum_a min(p,Cq)(e_a - p)-type terms is again non-zero; stronger token clipping removes more of the cancelling mass and can increase the bias.

## Fix
Weight each token score by the detached ratio r = p(a)/q(a), unclipped: loss = -A * sg(r) * log p(a) (gradient A ∇p(a)/q(a)). Then
E_q[r ∇log p(a)] = sum_a q(a) (p(a)/q(a)) ∇log p(a) = sum_a ∇p(a) = ∇ 1 = 0,
so the Abar term vanishes exactly as in the on-policy case for any q with support covering p, and the expected update is A-dependent only through the genuine covariance: sum_a ∇p(a)(Abar(a) - Abar) with Abar(a) the advantage conditional on choosing a. This is the gradient of the true objective for p. The only error left is variance, not bias, hence no drift accumulating through resampling. The weight is only limited by the log-ratio clamp at ±20 (numerical safety).

`loss_fn_rl` with `correction=None` now uses this exact token-level IS (ISCorrection(level="token", outside="clamp"), no band). Passing an explicit correction behaves as before.

## Configuration

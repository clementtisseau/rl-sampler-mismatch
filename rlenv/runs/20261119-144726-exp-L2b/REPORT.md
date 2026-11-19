# Sampler/trainer mismatch in REINFORCE

## What is wrong
Fix a prefix and a token position. Let p be the trainer's next-token distribution, q the sampler's, and A the
(group-centred) advantage of a rollout through x. The current loss (correction=None) has expected gradient
g = E_{x~q}[A(x) ∇log p(x)] = Σ_x q(x) A(x) ∇log p(x).
Write A(x) = V + δ(x), V = E_q[A | prefix] (the prefix value, not zero token-wise even though group-centering makes it zero in aggregate), δ the token-specific part.

On-policy (q=p): Σ_x p ∇log p = Σ_x ∇p = ∇1 = 0, so the V term vanishes exactly. This is the baseline invariance
of REINFORCE, and why a baseline/group centring is harmless and the gradient is just Σ p δ ∇log p.

Off-policy (q≠p): Σ_x q ∇log p = Σ_x (q−p)∇log p = −∇_θ KL(q‖p) ≠ 0 (in logit space it equals q−p).
So g = Σ_x q δ ∇log p  +  V·(−∇KL(q‖p)).
The second term is a systematic bias, independent of the sampled reward signal: if V>0 it pulls p toward q (locking in the
sampler's errors); if V<0 it pushes p away from q, raising entropy on bad prefixes and driving collapse. It does not average out: the sampler is
refreshed from p each step, so the deviation q−p is itself a function of the (biased) trajectory, and the bias compounds.
The first term also uses weights q instead of p, so it is not the policy gradient of p either.

## Fix
Weight each token by the detached ratio w = p/q (done in log space, only the ratio is clipped to e^±20 numerically):
g' = E_q[ (p/q) A ∇log p ] = Σ_x p(x) A(x) ∇log p(x) = Σ_x p δ ∇log p + V Σ_x ∇p = Σ_x p δ ∇log p,
since Σ∇p = 0 exactly for ANY q with support covering p. This restores the on-policy gradient (baseline invariance) at every prefix: the V·KL bias is zero by construction.
Clipping/dropping/band corrections (TIS, MIS, PPO...) use w̃(r)≠r, so Σ_x q w̃ ∇log p ≠ 0 and the V·(bias) returns, now with
a weight-dependent residual; stronger clipping means a larger residual, which explains why it can make things worse. Hence the loss's default is the
exact, unclipped token-level IS weight, active when correction=None. (An explicit correction passed by the caller is still honoured.)

Cost: higher variance when p/q is heavy-tailed, but that is zero-mean noise, unlike the bias, which accumulates.

## Configuration
rl.is.level=token

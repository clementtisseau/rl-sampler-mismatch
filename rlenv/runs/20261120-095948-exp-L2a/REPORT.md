# Stable RL under sampler/trainer mismatch

## What is wrong with the current loss
Rollouts y ~ q, the loss is -A(y) log p(y) (default: no correction), with A = R - b, b the group mean.
Expected update: g = E_q[A ∇log p] = E_q[(R-b) ∇log p].

On-policy (q = p) the score has zero mean, E_p[∇log p] = ∇Σ_y p = 0, so the baseline term b·E_p[∇log p] vanishes and g = ∇E_p[R] (unbiased, and the baseline only reduces variance).

Off-policy (q != p) this fails: E_q[∇log p] = Σ q ∇log p = -∇_θ KL(q||p) (up to a constant) != 0. Per token, in logit space the gradient is A(e_y - p) with expectation A(q - p) != 0. Hence
g = E_q[R ∇log p] - b ∇_θ(-KL(q||p))... i.e. the baseline is no longer free: it adds the term b·(−∇KL(q‖p)) = b·E_q[∇log p], a systematic push that
raises/lowers the likelihood of whatever q samples, proportional to b (which is not zero: b ≈ mean reward, and per-token
credit is not zero-mean either). Because the sampler is refreshed from the trainer every step, q moves with p, the bias acts
every step in a consistent direction (not zero-mean noise) and compounds, e.g. sharpening/flattening p relative to q and eventually collapsing.
Token-level clipping/dropping w = 1[r in band] (or min(r,c)) makes things worse: E_q[w(r)∇log p] = Σ_v q_v w(p_v/q_v) ∇log p_v is
no longer zero (it only equals 0 when w(r) = r), so clipping reintroduces exactly this bias, with sign set by which tokens are clipped (typically the high-ratio ones, so it systematically removes the gradient that would cancel the q-p drift).

## Fix
Use the importance weight r = p/q on every token, by default and without a band:
loss = -Σ_t sg(r_t) A log p_t.
Then, for each prefix, E_{y_t~q}[r_t ∇log p_t] = Σ_v p_v ∇log p_v = ∇Σ_v p_v = 0, so the baseline term (and any
advantage component that is constant given the prefix) has exactly zero expected update for any q, as in the on-policy case;
the remaining term is E_q[r A ∇log p] = E_p[A ∇log p], the on-policy gradient of the policy being trained (per-token ratio; the future-token ratios
of the sequence are ignored, which is the usual token-level approximation, with bias O(mismatch of later tokens)). In logit space the update is A r (e_y - p), whose q-expectation under baseline-only advantage is p - p = 0.

Implementation: in `losses.py`, `correction=None` now means an unbanded token-level IS weight (`ISCorrection(level="token", high=30, outside="clamp")`). The cap 30 is only
a variance guard: r has unit mean under q but variance Σp²/q - 1, which blows up on rare sampled tokens (an uncapped version was
less stable on a toy with many rare tokens; the cap trades tiny bias for variance). Explicit corrections passed by the caller are still honored.

## Configuration
(no configuration changes needed)

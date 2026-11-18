# Sampler/trainer mismatch in REINFORCE

## What is wrong with the current loss
With `correction=None` the loss is `-sum_t A * log p(y_t|y_<t)` on tokens drawn from q. Take the logit-space
gradient at one position: `d log p(y)/d logits = e_y - p`.

On-policy (q = p): `E_p[e_y - p] = 0`. Any term that does not depend on y, such as a baseline b or the group mean
that group-centering subtracts, has zero expected gradient. The update is `E_p[(R-b) grad log p] = grad J` for any b.

Off-policy (y ~ q): `E_q[grad log p(y)] = sum_y q(y) grad log p(y) = -grad CE(q,p)`, and in logit space it equals `q - p != 0`.
So for the expected update, with `A = (A - Abar) + Abar` split into a part correlated with the outcome and a part
that is constant given the prefix, the constant part (baseline, group mean, token-length weighting, the sign of the
advantage) gives a systematic drift `Abar_t (q_t - p_t)` instead of 0.
In total the expected update is `sum_y q(y) A(y) grad log p(y)`, not `grad J(p)`.
- Positive-advantage tokens pull p toward q (a pull toward the sampler's quantisation or noise distortion).
- Negative-advantage tokens push p away from q, and this is unbounded because log p of a sampled token can go to -inf.
- This is a deterministic bias, not zero-mean noise. Because the sampler is rebuilt from the updated trainer at each step,
  the bias is applied to the new p and compounds, which is why training degrades or collapses over time.

## Fix
Per token, use the importance-weighted score with the detached, unclipped ratio `w = p(y)/q(y)`:
`loss = -sum_t A * sg(p/q) * log p`. This is the default when `correction=None`
(equivalent to `ISCorrection(level="token")` with no band).

Derivation. At each position, with prefix fixed,
`E_{y~q}[ (p(y)/q(y)) grad log p(y) ] = sum_y p(y) grad log p(y) = sum_y grad p(y) = grad 1 = 0`.
So the pure-baseline part of the advantage again has exactly zero expected gradient, as on-policy, and
the drift `Abar_t (q_t - p_t)` vanishes. The remaining part is
`E_q[A_t w_t grad log p] = E_p[A_t grad log p]` per token, which is the on-policy per-token term.
The residual is only the prefix mismatch (the distribution of prefixes is q's instead of p's),
a second-order effect that is not self-reinforcing like the baseline drift.

Why clipping or dropping hurts. With `w_c = min(w, c)` or masking,
`E_q[w_c grad log p] = -sum_{y: p/q>c} (p - c q) grad log p != 0`.
It reintroduces the same baseline drift, now biased against exactly the tokens where p and q differ most, so stronger clipping
(smaller band) makes it larger. Hence the default has no clipping or masking.
Variance is the price: `w` has mean 1 under q and is moderate for small mismatch (the log-ratio is clipped at e^±20 only for numerical safety).

## Configuration
(No configuration change needed.)

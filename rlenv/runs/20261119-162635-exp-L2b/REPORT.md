# Sampler-centered score: removing the mismatch drift from REINFORCE

## Summary

When the sampler q differs from the trainer p, the per-token REINFORCE score
`grad log p(y)` no longer has zero mean under the distribution the tokens are
actually drawn from. Its mean is `E_q[grad log p] = sum_v (q_v - p_v) grad log p_v
= -grad KL(q || p)`. Each token's update therefore picks up a term
`V(h) * (-grad KL(q||p))`. Here `V(h) = E[A | h]` is the conditional mean of
the advantage at that prefix, which group centering does not remove. This
term is a deterministic, signed push toward the sampler (V > 0) or away from
it (V < 0). It is not noise. Because the sampler is rebuilt from the trainer
at every step, the push has the same direction at every step and compounds.

The fix is to subtract the score's own expectation under the sampler at
every position. This is a control variate whose mean is known:

    g_t = A * [ w(y_t) grad log p(y_t|h_t)  -  sum_v q(v|h_t) w(v) grad log p(v|h_t) ]

(w = 1 by default, i.e. `correction=None`.) For any fixed prefix,
`E_{y~q}[g_t] = Cov_q(Q, w grad log p)`. The drift term is gone exactly. The
term needs only the sampler's top-k logprobs (top-20, which
`train_rl.py`/`dev/mini.py` already pass in the batch) and the trainer's
full-vocabulary logprobs, which the loss already computes. It has no
importance ratio and so no variance blow-up. When q = p it is identically
zero, and the loss reduces exactly to the original REINFORCE.

## 1. Setup: one token position

Fix a prefix h. The trainer has logits z, `p = softmax(z)`. The sampler draws
`y ~ q(.|h)`. The rest of the rollout is drawn from q, and the rollout gets an
advantage A (group centered). Define

- `Q(y) = E[A | h, y]` (expectation over the sampler's continuation),
- `V = E_{y~q}[Q(y)] = E[A | h]`.

The loss `-A * w(y) * log p(y)` (summed over tokens, divided by `n_tokens`)
gives the per-token ascent direction `A w(y) grad log p(y)`. In logit space,
`d log p(y) / dz = e_y - p`.

## 2. The on-policy case (q = p)

    E_{y~p}[A grad log p(y)] = sum_y p_y Q(y) grad log p_y
                             = sum_y p_y (Q(y) - V) grad log p_y  +  V * sum_y p_y grad log p_y.

The last factor is `sum_y grad p_y = grad 1 = 0`. This is the identity
`E_p[score] = 0` that makes REINFORCE baseline-invariant. Whatever constant
part the advantage has at this prefix (and V can be far from zero, see
section 4), it contributes nothing in expectation. Only the centered part
`Q - V` drives learning. In logits the update is `p ⊙ (Q - V)`.

## 3. The mismatched case (q != p)

    E_{y~q}[A grad log p(y)] = sum_y q_y (Q(y) - V) grad log p_y   +   V * sum_y q_y grad log p_y
                                 (learning signal)                      (drift)

Now `sum_y q_y grad log p_y != 0`:

    in logits:      sum_y q_y (e_y - p) = q - p
    in parameters:  sum_y q_y grad_theta log p_y = grad_theta sum_y q_y log p_y = -grad_theta KL(q || p_theta).

So each token adds a drift `-V(h) grad KL(q(.|h) || p(.|h))`:

- at prefixes of rollouts that tend to succeed (V > 0), p is distilled
  toward the sampler, noise included;
- at prefixes of rollouts that tend to fail (V < 0), p is pushed away from
  the sampler. Its logits move by `|V| (p - q)`. This lowers p further on
  tokens the sampler over-weights (for example the long tail that weight
  noise or quantization lifts) and raises it where the sampler under-weights.

This term is **not zero-mean noise**. Its sign comes from V and its
direction from the mismatch `q - p`. That mismatch is a deterministic
function of the current weights: a fixed noise direction, the rounding
pattern of int8/fp8, a KV-cache quantization error, lag. After the update
the sampler is rebuilt from the new weights by the same map, so the same
kind of `q - p` appears again and the drift adds up linearly over steps. It
does not average out like `sqrt(steps)`. This is the "systematic bias that
compounds".

## 4. Why the drift dominates, and why group centering does not help

Group centering makes `E[A] = 0` only at the prompt (the first completion
token). After a few tokens the prefix largely decides the outcome. At such
positions `Q(y) ≈ V` for every likely y and `|V|` is large: about `1 - mean`
or `-mean` for binary rewards. So:

- the learning signal `sum_y q_y (Q(y) - V) grad log p_y` is close to 0;
- the drift `V (q - p)` is at full strength.

Most tokens of a rollout are such "determined" positions. As training
converges, Q - V shrinks everywhere, but the drift does not. Late in
training the expected update is mostly drift, which is when collapse
appears. The task's own normalization (sum over all tokens of all
sequences / `n_tokens`) keeps this term for every token.

## 5. Why IS corrections and clipping only slow it down

With a detached weight w(r), `r = p/q`, define `m_y = q_y w(r_y)`. Then

    E_q[A w grad log p] = sum_y m_y (Q(y) - V_m) grad log p_y + V_m * sum_y m_y grad log p_y,

with `V_m = sum m Q / sum m`. In logits, the drift factor is
`sum_y m_y (e_y - p) = m - |m| p`. This is zero **iff m is proportional to
p**, i.e. iff `w(r) ∝ r` for every token.

- Naive token IS (`w = r`, so `m = p`) removes the drift exactly. But its
  variance is the chi-square `sum p^2/q`, and it is heavy-tailed on tokens
  the sampler rarely produces. That is why it is clipped in practice.
- Truncation (TIS: `m = min(p, c q)`), band dropping (MIS/PPO-like:
  `m = p * 1[r in band]`), and DPPO's TV band all make m not proportional
  to p. A drift `V (m - |m| p)` remains, now pulling toward the in-band
  tokens. It is smaller than before (it slows the degradation) but it is
  still systematic, so it still compounds.
- A tighter band drops more tokens, which moves m further from being
  proportional to p. The residual drift becomes **larger**. This explains
  why stronger token-level clipping can make things worse.

Section 7 shows that only estimators linear in `1/q(y)` can be exactly
unbiased using q at the sampled token alone. So "fix the weight" can only
trade bias against variance. The way out is to use more information about q
than the sampled token's probability: the sampler's top-k logprobs.

## 6. The fix: center the score under the sampler

Subtract the score's own sampler expectation, a control variate with
known mean:

    g = A [ w(y) grad log p(y) - mu ],      mu = E_{v~q}[ w(v) grad log p(v) ] = sum_v m_v grad log p_v.

**Drift is removed exactly.** For a fixed prefix,

    E_{y~q}[g] = sum_y q_y Q(y) (w_y grad log p_y - mu)
               = V (mu - mu) + sum_y q_y (Q(y) - V)(w_y grad log p_y - mu)
               = Cov_{y~q}( Q(y), w(y) grad log p(y) ).

Any part of the advantage that does not depend on the sampled token,
including V(h), now gives exactly zero expected update, as on-policy. The
baseline invariance of REINFORCE is restored under q.

**What remains is learning signal.** With no correction (w = 1, m = q), in
logits `E[g] = sum_y q_y (Q(y) - V)(e_y - q) = q ⊙ (Q - V)`, because
`sum_y q_y (Q(y) - V) = 0`. The on-policy update is `p ⊙ (Q - V_p)`. The
two differ by `O(|q - p|) * |Q - V|`. That error scales with the advantage
spread, so it vanishes at determined positions and as the policy converges.
It has no fixed sign and is not the gradient of a divergence between p and
q. The uncorrected drift scales with |V| instead, which does not vanish.
The fixed point of the expected dynamics is now "Q(y) = V for all sampled
y" (nothing left to learn), not "keep moving along q - p".

**It is exact on-policy.** If q = p and w = 1, `mu = sum_v p_v grad log p_v = 0`,
so the loss is the original REINFORCE, identical up to float error. With
naive IS (`m = p`), mu is also 0. For any other token correction, mu
removes the residual drift `m - |m|p` from section 5. So the fix composes
with every `rl.is.*` setting.

**Bounded variance, no ratio.** mu is a convex combination of scores
(`|m| ≤ 1` without correction). The estimator contains no `1/q` and no
clipping, so there is no tail blow-up. Subtracting the mean also lowers the
variance: `Var(score - mu) = E||score||^2 - ||mu||^2`.

### Implementation (losses.py)

`mu` is the gradient of the detached-coefficient surrogate
`C = sum_v sg(m_v) log p_v`. The per-token score term becomes
`w(y) log p(y) - C`, multiplied by A as before. Normalization, metrics and
the signature are unchanged. The term is always on (`correction=None`
included).

The sampler returns only its top-k (k = 20) logprobs, as real inference
engines do. Write K for the top-k ids, `Q_K = sum_K q`, `P_K = sum_K p`.

- On K, `m_v = q_v w(p_v/q_v)` exactly. It is computed with the existing
  `log_mass(log p_v, log q_v, ...)`, so clamp/drop/sign/TV variants apply
  per vocabulary entry.
- Outside K, the tail mass `1 - Q_K` is spread proportionally to p:
  `q_v ≈ s p_v` with `s = (1 - Q_K)/(1 - P_K)`, so `m_v = κ p_v` with
  `κ = s w(1/s)` (κ = s without correction).
- Because `sum_v p_v grad log p_v = 0`, the whole tail contributes
  `κ sum_{v∉K} p_v grad log p_v = -κ sum_{v∈K} p_v grad log p_v`. So

      C = sum_{v ∈ K} sg( m_v - κ p_v ) log p_v,

  which only needs a gather of k entries of the trainer's log-softmax (the
  same `take_vocab` used for the sampled token). It is exactly 0 when q = p.
- Tail masses are computed in log space (`log(1 - exp(logsumexp))`), and
  `log s` is clipped to ±20.

The only approximation is the tail's shape. The residual drift is
`V sum_{v∉K} (q_v - s p_v) grad log p_v`. It is bounded by the sampler mass
outside its top-20 and is zero when the tail's relative shape matches p.
In the dev setting with 200 extra tail tokens (`MINI_EXTRA_TOKENS=200`),
top-20 was enough. If a caller does not provide top-k logprobs, the code
falls back to K = {sampled token}, which is exact only to the extent the
rest of q is proportional to p. That is a weaker fallback.

Sequence-level corrections (`seq_mean`, `seq_sum`) multiply A by the
sequence weight. The token-level centering then uses m = q.

## 7. Why the sampled-token logprob alone cannot fix this without IS

Suppose a per-token estimator `F(y, q(y), p)` must satisfy
`sum_y q_y F(y, q_y, p) = 0` for every sampler q. Varying a single `q_y`
under `sum q = 1` forces `q_y F(y, q_y) = λ q_y + κ_y`, so
`F(y) = κ_y / q_y + λ` with `λ = -sum_y κ_y`. Writing `κ_y = p_y h_y`
gives `F = r(y) h(y) - E_p[h]`. Every exactly unbiased correction that uses
only `q(y)` is an importance-weighted estimator, so it has unbounded
variance, and every bounded (clipped) one is biased. This is why the
IS/clipping family cannot remove the drift. A bounded and unbiased
correction needs more of q than one number per token. Inference engines
return their top-k logprobs anyway, and that is enough to compute mu.

## 8. Sanity checks

- Unit check at one position (12-token vocab, perturbed head). The exact
  expectation over y ~ q of the corrected logit gradient is about 4e-8 with
  the full support, 1e-2 with top-5 plus the proportional tail, and 0.19 for
  plain REINFORCE.
- On-policy (q = p), the gradient is bit-identical to the original loss.
- `dev/mini.py` (noise 0.3, lr 0.2, 800-1000 steps), final eval reward,
  original loss vs. this loss:
  - default setting: seed 0, 0.66 vs 0.79 (about 0.85 for most of the run); seed 1, 0.66 vs 0.83;
  - noise 0.5, group-centered: original collapses to about 0.0 (KL to
    about 100), this loss reaches 0.68 (KL about 5);
  - 200 extra tail tokens: original collapses to 0.0 by step 50, this loss
    reaches 0.80;
  - the same with TIS: 0.79 vs 0.89.

## Configuration

No configuration changes are needed. The modification is active with the
default `correction=None`.

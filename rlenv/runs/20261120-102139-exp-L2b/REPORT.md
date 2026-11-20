# Sampler/trainer mismatch in REINFORCE: the score is not centered under q

## Setup and notation

Look at one completion position with a fixed prefix `s`. The trainer has logits
`z ∈ R^V` and policy `p = softmax(z)`. The next token was drawn from the sampler,
`y ~ q(·|s)`, and `q ≠ p` (weight noise, int8/fp8, KV quantization, stale
weights). The current loss puts `-A · w(y) · log p(y)` on this token, where `A`
is the sequence advantage and `w ≡ 1` without a correction. Write

* `e_y` for the one-hot vector of `y`,
* `A(y) := E[A | s, y]` for the expected advantage after choosing `y` (the rest of
  the rollout also comes from q),
* `Ā_q := Σ_y q(y) A(y) = E_q[A | s]` for the conditional mean advantage at this prefix.

The gradient of the per-token score with respect to the logits is

    ∇_z log p(y) = e_y − p,

and with respect to the parameters it is `∇_θ log p(y) = J^T (e_y − p)`, where
`J = ∂z/∂θ` depends only on the prefix. So everything below holds in parameter
space too, after multiplying by `J^T`.

## What goes wrong: the expected update under q

Gradient ascent on `A·log p(y)` gives the expected logit update

    g = E_{y~q}[ A(y) (e_y − p) ]
      = Σ_y q(y) A(y) e_y − Ā_q p
      = q ⊙ (A − Ā_q)  +  Ā_q (q − p).                    (1)

**On-policy (q = p).** The second term vanishes, which leaves `g = p ⊙ (A − Ā_p)`.
This is the exact policy gradient in logit space. It depends only on the
centred advantage `A − Ā_p`, so adding a constant to `A` has no effect on the
expected update. This baseline invariance holds because the score is centred:
`E_p[∇ log p(y)] = Σ_y p(y)(e_y − p) = 0`.

**Mismatched (q ≠ p).** The score is no longer centred:

    E_{y~q}[∇_z log p(y)] = q − p ≠ 0,
    E_{y~q}[∇_θ log p(y)] = Σ_y q(y) ∇_θ log p(y) = −∇_θ CE(q, p).

So the update picks up the extra term `Ā_q (q − p) = −Ā_q ∇ CE(q,p)`. At each
prefix this is supervised learning toward the sampler, weighted by the
conditional mean advantage. If `Ā_q > 0`, p is distilled toward q. If `Ā_q < 0`,
p is pushed away from q by gradient ascent on the cross-entropy `CE(q,p)`.
That objective is unbounded: it drives `p(v) → 0` on tokens the sampler
likes.

Why this term does not average out:

* **`Ā_q(s) ≠ 0` at almost every prefix.** Group centring makes advantages sum
  to zero over a group of rollouts for a prompt. It does not make
  `E[A | s] = V_q(s) − b` zero at intermediate prefixes. That quantity is the
  value of the prefix relative to the group mean, and it is nonzero and
  structured: prefixes that are going well have `Ā_q > 0`, prefixes that are
  failing have `Ā_q < 0`. A centred reward also does not remove the drift,
  because changing the baseline by `c` changes `g` by `c (q − p)`. In the
  mismatched case the baseline is not a free choice.
* **`q − p` is systematic, not noise.** It is fixed by the perturbation (the
  same noise tensor, the same quantization grid, the same lag), so the drift
  points the same way step after step and adds up coherently.
* **It compounds.** The sampler is rebuilt from the trainer every step,
  `q_{t+1} = Φ(p_{t+1})`. After p drifts by `Ā (q − p)`, the new sampler applies
  the same distortion on top of the drifted model, so there is always a fresh
  `q − p` to follow. In particular, if the reward carries no information
  (`A(y) ≡ c ≠ 0`), the on-policy update is exactly zero. The current loss still
  moves the model by `c (q − p)` at every step, and because `q = Φ(p) ≠ p` it
  never reaches a fixed point. That is a drift without a stationary point, and
  it shows up as slow degradation followed by collapse.

**Why IS and clipping only slow it down.** With a detached weight `w(y)`:

    E_{y~q}[ w(y)(e_y − p) ] = m − (Σ_v m_v) p,   m_v := q_v w(p_v/q_v).

This is zero only if `m = p`, which means the untruncated ratio `w = p/q` on
every token. Any clamp or drop gives `m ≠ p`. For example, a PPO-style drop
outside `[1−ε, 1+ε]` gives `m = p·1_band`, so the bias is
`Ā (p_band − P(band) p)`. A narrower band moves more mass out of the band and
increases the bias. This is why stronger token-level clipping can make things
worse. The untruncated ratio is unbiased, but its variance is `χ²(p‖q)`, it has
heavy tails, it is clipped at `e^{±20}` in practice, and it is zero or undefined
off the sampler's support. Sequence-level weights rescale `A` but leave the
per-token score uncentred, so the `Ā(q − p)` term survives them.

## The fix: centre the score under the sampler

Subtract the score's own mean under q, which is a per-position baseline on the
score rather than on the reward:

    ∇ log p(y)   →   ∇ log p(y) − E_{v~q}[∇ log p(v)]
                 =   ∇ [ log p(y) − Σ_v sg(q_v) log p_v ].

In logits this is `e_y − p − (q − p) = e_y − q`, so the expected update is

    g_new = E_{y~q}[ A(y)(e_y − q) ] = q ⊙ (A − Ā_q).                    (2)

Properties:

1. **No drift term, for any q.** The `Ā_q(q−p)` term in (1) is removed exactly,
   and baseline invariance comes back: `E_q[c (e_y − q)] = 0` for every
   constant `c`. In parameter space,
   `E_q[∇_θ log p(y) − Σ_v q_v ∇_θ log p_v] = 0` exactly, because `J` does not
   depend on `y`. A reward-uninformative batch now gives an update of zero in
   expectation, so the loop `p → q = Φ(p) → p` has nothing to compound.
2. **It reduces to the original loss on-policy.** If `q = p`, then
   `∇ Σ_v sg(p_v) log p_v = p − p = 0`, so the loss is unchanged.
3. **It is still a policy-improvement direction.** (2) has the same form as the
   on-policy gradient `p ⊙ (A − Ā_p)`, with q in place of p. Its inner product
   with the true gradient `∇_z J = p ⊙ (A − Ā_p)` is
   `Σ_v p_v q_v (A_v − Ā_p)(A_v − Ā_q)`. This is positive for q near p and equals
   `Σ p²(A−Ā)² ≥ 0` at `q = p`. It is also the exact on-policy gradient of the
   policy that actually generates the rewards: for a mismatch that shifts the
   logits, `z_q = z + δ`, a change `Δz` moves `z_q` by the same `Δz`, and
   `∇_{z_q} E_q[A] = q ⊙ (A − Ā_q)`. The remaining difference from the ideal
   gradient is `O(‖q−p‖ · |A − Ā|)`. It is proportional to the centred advantage,
   so it carries no direction that is independent of the reward.
4. **No importance weights, so the variance stays bounded.** Each token's
   gradient is `A (e_y − q)`, with norm at most `2|A|`. There is no `1/q(y)`.

**With a correction.** If a detached weight `w(y)` is used, the baseline becomes
`B = Σ_v m_v log p_v` with `m_v = q_v w(p_v/q_v)`. The code computes this as
`exp(log_mass(...))` over the vocabulary. Then
`E_q[w(y)∇log p(y)] − ∇B = 0` exactly, so clamping or dropping only reweights
the centred term and no longer creates drift.

## Implementation (`losses.py`)

The sampler returns its top-k (k = 20) logprobs at every position
(`sampler_topk_ids`, `sampler_topk_logprobs`, as inference engines do). This is
enough to build q:

* On the top-k set K, use the exact `q_v`.
* Spread the remaining mass `q_tail = 1 − Σ_K q_v` over the other tokens in
  proportion to p: `q̂_v = q_tail p_v / p_tail`. The only bias left is confined to
  the tail and bounded by `2 Ā q_tail`. A metric `q_tail` is logged.

For each position, `sampler_baseline` computes

    B = Σ_{k∈K} m_k log p_k + (m_tail/p_tail) Σ_{v∉K} sg(p_v) z_v − m_tail · logsumexp(z),

with `m = q` (or `q·w` under a correction). This matches
`Σ_v m_v log p_v` up to a constant. The trainer tail mass `p_tail = 1 − Σ_K p_k` is computed in float32, so it is
floored at `1e-4` before dividing by it. Without the floor, rounding error made
`m_tail/p_tail` explode, which caused a NaN in an early version. With the
floor, the spread mass is at most `m_tail`. The tail sum is written in terms of logits
so that its gradient (`(m_tail/p_tail) p_v` on the tail, `−m_tail p` overall) is
computed without cancellation, even when `p_tail` is tiny. The per-token term
becomes `w·log p(y) − (B − sg(B))`. The reported loss value is unchanged, and
the gradient is the centred score. The change is active by default
(`correction=None`) and composes with every `rl.is.*` setting. If a batch has no
top-k logprobs, the sampled token (whose `q(y)` is known) is used as K.

A Monte Carlo check at a single position (V = 50, a perturbed q, A ≡ 1, 20k
samples) measures the bias as `‖E_q[∇_z loss]‖₁`:

| | Old loss | New loss |
| --- | --- | --- |
| No correction | 0.52 | 0.02 (Monte Carlo noise level) |
| PPO-style band | 0.49 | 0.006 |
| With a 15% tail beyond top-10 | — | 0.06 |

Sanity runs on `dev/mini.py` (CPU, weight noise 0.3, 600 steps). The numbers
are the mean reward over the last 10 steps:

| Setting | Old loss | New loss |
| --- | --- | --- |
| Default vocabulary, seed 0 | 0.58 (stalls, eval 0.66) | 0.85 (eval 0.83) |
| 40 extra tokens (q has a tail beyond top-20), seed 0 | 0.00 (collapsed by step 100) | 0.84 |
| 40 extra tokens, seed 1 | 0.00 (collapsed by step 50) | 0.87 |

## Configuration

None needed. The defaults (no `rl.is.*` overrides) are intended.

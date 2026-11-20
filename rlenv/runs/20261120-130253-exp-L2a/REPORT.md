# Sampler-centered score: removing the mismatch drift from REINFORCE

## Summary

When rollouts come from a sampler `q` that is not the trained model `p`, the
REINFORCE score `∇ log p(y)` no longer has zero mean. Its mean under `q` is
`−∇ KL(q‖p)`, a direction that depends only on the mismatch and not on the
reward. Whatever part of the advantage the baseline did not remove at a given
prefix multiplies this mean, so every step adds a deterministic push
`−(V(s) − b) ∇KL(q‖p)` to the update. Group centering does not cancel it.
When the sampler is refreshed as `q = perturb(p)`, the push points the same
way at every step, so it accumulates.

**Fix (in `losses.py`, on by default):** for each token, subtract the
sampler expectation of the (weighted) score:

    score_t = w(y_t) log p(y_t) − Σ_v sg(q_v w_v) log p(v)          (w ≡ 1 by default)

The sum over `v` is computed from the sampler's top‑k logprobs, which the
batch already carries as `sampler_topk_ids` and `sampler_topk_logprobs`. The
probability mass outside the top‑k is handled in closed form. The resulting
score has exactly zero mean under `q`, for any advantage. When `q = p` it
gives the same gradient as standard REINFORCE: the extra term's gradient is
`Σ_v p_v ∇log p_v = ∇Σ_v p_v = 0`. So the change only affects training when
there is a mismatch.

## 1. Setup

At a prefix `s` (prompt plus generated tokens so far) the trainer has logits
`z`, with `p = softmax(z)`. The sampler draws `y ~ q(·|s)`. The sequence
advantage `A` (here `A = R − b`, with `b` the group mean or a constant)
multiplies every token's score. Per token, the current loss gives this
update:

    g = A · ∇_θ log p(y|s) = A · J_θᵀ (e_y − p),      J_θ = ∂z/∂θ.

## 2. Expected update: on-policy vs. mismatched

Let `Q(s,y) = E[R | s, y]` (later tokens drawn from `q`) and
`V(s) = E_{y~q}[Q(s,y)]`. Then `E[A | s, y] = Q(s,y) − b`, and in logit
space:

    E_q[g | s] = Σ_y q_y (Q(s,y) − b)(e_y − p)
               = Σ_y q_y (Q(s,y) − V(s))(e_y − p)  +  (V(s) − b) Σ_y q_y (e_y − p)
               =        "signal"                    +  (V(s) − b) · (q − p).        (1)

**On-policy (`q = p`).** `Σ_y p_y (e_y − p) = 0`, so the second term is zero.
The baseline `b` and the gap `V(s) − b` have no effect on the expected update,
and the signal term equals `p ⊙ (Q − V)`, the exact policy gradient. This is
the property that lets REINFORCE use any baseline, including a constant one
such as `plus_minus` or a group mean that does not match the value of each
prefix.

**Mismatched (`q ≠ p`).** The second term does not vanish:

    drift(s) = (V(s) − b) (q − p) = −(V(s) − b) ∇_z KL(q(·|s) ‖ p(·|s)),        (2)

because `∇_z KL(q‖p) = p − q`. Key properties:

* **It is not noise.** It is the expectation itself, so larger batches do
  not remove it. Its direction is fixed by the mismatch alone and is
  unrelated to the task.
* **Its sign depends on how good the prefix is.** In prefixes worse than the
  baseline (`V(s) < b`: all failing rollouts with `plus_minus`, and any bad
  prefix with group centering), the drift *increases* `KL(q‖p)`. It moves
  `p` away from `q`, toward `p + (p − q)`, and raises tokens that the
  perturbation happens to suppress. In good prefixes it pulls `p` toward the
  perturbed `q`, which also degrades `p` because `q` is a noisy copy.
* **Group centering does not remove it.** Centering gives `Σ_i A_i = 0` per
  group, but the drift that sequence `i` collects is `Σ_t (q_t − p_t)`. This
  is a sum over its own prefixes, and it differs across sequences: its size
  grows with length, and it depends on which low-probability tokens the
  sampler emitted. Those tokens are exactly the ones that make a rollout fail.
  So the net drift `Σ_i A_i Σ_t (q_{it} − p_{it}) ≠ 0`. It equals the
  covariance between advantage and accumulated mismatch, and its sign is
  systematic.
* **It compounds.** The sampler is rebuilt from the trainer at every step
  with the same kind of perturbation (fixed noise, the same quantizer, the
  same KV quantization), so `q − p` points in roughly the same direction at
  each step. The update `Δz ∝ −η Σ (V − b)(q − p)` therefore adds up roughly
  linearly in the number of steps instead of averaging out. As `p` drifts,
  `q − p` grows: the sampler emits more of the tokens the trainer has pushed
  down, those rollouts fail, and the next step pushes the same way harder.
  This is the observed collapse.

## 3. Why IS weights and clipping only slow it down

With a detached per-token weight `w(r)`, `r = p/q`, the drift term becomes
`(V − b) Σ_y q_y w_y (e_y − p) = (V − b)(m − |m| p)` with `m_y = q_y w_y`.
This is zero only if `m ∝ p`, i.e. `w ∝ p/q` with no clipping. Naive IS
achieves that per token, but its variance (`E_q r² = 1 + χ²(p‖q)`) explodes
on rare tokens, and every practical variant truncates the weight. Truncation
makes `m` differ from `p` exactly on the tokens where the mismatch is
largest, which brings the drift back. Tighter token clipping (PPO/DAPO
bands, TIS caps) moves `m` further from `p` and can increase the drift, as
observed. Sequence-level weights change how much each sequence counts but
leave each token's score `e_y − p` uncentered, so (1) still applies.

## 4. The fix: center the score under the sampler

The problem is the score's mean `E_q[e_y − p] = q − p`, not the advantage.
So subtract that mean directly, using a control variate whose gradient is
known:

    ∇_z [ log p(y) − Σ_v sg(q_v) log p(v) ] = (e_y − p) − (q − p) = e_y − q.

More generally, with a weight `w` from an optional `ISCorrection`:

    ℓ_t = w(y_t) log p(y_t) − Σ_v sg(q_v w(p_v/q_v)) log p(v),
    ∇_z ℓ_t = w_y (e_y − p) − Σ_v q_v w_v (e_v − p),        E_{y~q}[∇_z ℓ_t] = 0.   (3)

and the loss is `L = −Σ_i A_i Σ_t ℓ_{it} / N_tokens`, as before.

Expected update with (3), using `w ≡ 1` (the default):

    E_q[A ∇_z ℓ | s] = Σ_y q_y (Q(s,y) − b)(e_y − q) = q ⊙ (Q(s,·) − V(s)).

* The `b` and `V(s) − b` terms are gone for **any** baseline, as in the
  on-policy case. The constant drift `−(V − b)∇KL(q‖p)` is removed. What is
  left depends only on how `Q` varies across tokens.
* In logit space, what is left is the exact policy gradient of the sampler
  policy `q`. This is a reward-improving direction. It differs from `p`'s
  gradient only by `O(‖q − p‖)` in its *weights* (`q_y` instead of `p_y`),
  not by a fixed offset. It is zero wherever the reward does not depend on
  the token, so a constant advantage no longer moves the model. There is
  nothing to accumulate step after step, and the mismatch no longer feeds
  itself.
* **On-policy it changes nothing:** with `q = p`, `Σ_v p_v ∇log p_v = 0`, so
  the gradient equals the original REINFORCE gradient (checked numerically:
  max difference about 1e-8).
* It is also a variance reduction: `e_y − q` is the score centered by the
  best available constant in logit space.
* With an IS correction, (3) restores zero mean for clipped or truncated
  weights too. This fixes the failure from section 3 without changing the
  correction's band.

### Computing `Σ_v q_v log p(v)` from top‑k logprobs

The sampler returns its top‑k (k = 20) logprobs per position, as inference
engines do. Let `K` be that set, `Q_t = 1 − Σ_{v∈K} q_v` and
`P_t = 1 − Σ_{v∈K} p_v`. For the tail, take `q` proportional to `p`
(`q_v = Q_t p_v / P_t` for `v ∉ K`), which is the best guess without further
information. Then every tail token has ratio `P_t/Q_t`, and

    Σ_{v∉K} q_v w_v log p(v)  →  Q_t · w(P_t/Q_t) · log P_t,

whose logit gradient is `Q_t w (p_v/P_t)(…)` on the tail. This exactly
matches the assumed tail distribution, so the centering term's weights still
sum to `Σ_v q_v w_v` and (3) holds exactly for that `q`. The code computes
`log P_t` as `log1p(−Σ_K p)`, clamped away from `log 0`, and skips the tail
when `Q_t < 1e‑6`. The remaining error affects only the shape of the tail
inside a mass `Q_t`, which is small for top‑20.

If the batch carries no top‑k logprobs and no correction is given, the code
falls back to untruncated token IS, `w = p/q`. That is the only per-token
weight that gives a `q`-centered score using only `q(y)`:
`E_q[(p/q)(e_y − p)] = Σ_y p_y (e_y − p) = 0`.

## 5. Implementation (`losses.py`)

* `sampler_baseline(logp_vocab, top_ids, top_logq, positive, c)` returns
  `Σ_{v∈K} sg(q_v w_v) log p_v + sg(Q_t w_t) log P_t`. The weights use the
  same `is_weight` and `ISCorrection` as the sampled token, so (3) holds for
  every correction on the grid. For sequence-level corrections, `w ≡ 1` at
  the token level and the sequence weight scales the advantage of the
  centered score.
* In `loss_fn_rl`, the per-token score `w·log p(y)` becomes
  `w·log p(y) − baseline`. It is on with the default arguments
  (`correction=None`), and the signature is unchanged.

## Configuration

None. The modification works with the default `rl.is.*` settings and needs
no flag.

# Sampler-centered score: removing the mismatch drift from REINFORCE

## Summary

The REINFORCE score `∇ log p(y_t)` has mean zero only when tokens are drawn from
the trainer `p`. When they come from a sampler `q ≠ p`, its mean is `q - p` in logit
space, which is nonzero. The policy gradient multiplies this by the advantage, so the
expected update picks up a term `V(s)·(q - p)` that does not depend on reward. Here
`V(s) = E[A | prefix s]`. This term is a hidden distillation of the trainer toward
the sampler (`V > 0`) or away from it (`V < 0`). The sampler is recomputed from the
trainer every step, so the term has no fixed point and its effect builds up over
training. This build-up is the collapse.

The fix is to center each token's score on its expectation **under the sampler**
rather than under the trainer:

    score_t = log p(y_t | s_t) - Σ_v q(v | s_t) · log p(v | s_t)          (q detached)

In logit space its gradient is `e_{y_t} - q_t`, which has mean zero under `q` at
every prefix. The drift term disappears exactly. What remains is the reward-driven
part of the gradient. On-policy (`q = p`) the extra term has zero gradient, so the
loss is exactly REINFORCE. No ratios are used, so the variance cannot blow up and no
clipping is needed. The loss uses the sampler's top-k logprobs that the batch already
carries (`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20, the same as what
inference engines return).

## Setup and notation

- `s` is a prefix (state). `z_p(s)` are the trainer logits, `p = softmax(z_p)`, and
  `J_s = ∂z_p(s)/∂θ`.
- `q(·|s)` is the sampler's next-token distribution. The sampler is a function of the
  trainer: `q = Φ(p_θ)` (weight noise, int8/fp8, KV quantization, staleness).
- A trajectory `y ~ q` gets a sequence advantage `A` (group-centered or ±1).
  `Q(s, y) = E_q[A | s, y_t = y]` and `V(s) = Σ_y q(y|s) Q(s, y) = E_q[A | s]`.
- The current loss is `L = -(1/N) Σ_i A_i Σ_t log p(y_{i,t} | s_{i,t})`. For token t,
  the ascent direction in logit space is `A · (e_{y_t} - p(·|s_t))`, and in parameter
  space it is `J^T` applied to that.

## 1. Expected update of the current loss

Condition on a prefix `s` that the sampler reaches, and take the expectation over
`y_t ~ q` and the rest of the rollout:

    g(s) = E_q[ A (e_y - p) | s ] = Σ_y q_y Q(s,y) (e_y - p).

Add and subtract `q`, and use `Σ_y q_y (e_y - q) = 0`:

    g(s) = Σ_y q_y (Q(s,y) - V(s)) (e_y - q)   +   V(s) · (q - p)
           \______________________________/       \______________/
                   (I) reward-driven                (II) drift

- **(I)** equals `∇_{z_q} E_{y~softmax(z_q)}[Q(s,y)]`. This is the exact policy
  gradient of the sampler's logits at `s`. It depends only on how the actions differ
  in advantage, and it is zero when they are all equally good.
- **(II)** does not depend on which action is better. Since
  `∇_{z_p} KL(q || p) = p - q` with `q` held fixed,

      (II) = -V(s) · ∇_{z_p} KL(q(·|s) || p(·|s)),

  and summed over tokens in parameter space the loss carries a hidden term:

      E[update] ∋ -E_q Σ_t V(s_t) ∇_θ KL( q(·|s_t) || p_θ(·|s_t) ).

**On-policy** (`q = p`), (II) is zero for every `s`, because the score has mean zero
under `p`. This identity is the only reason baselines and advantage centering are
"free", and it is the only reason REINFORCE is unbiased. **Off-policy**, the identity
fails, and each state with `V(s) ≠ 0` gives:

- `V(s) > 0`, on prefixes heading toward success: distillation of `p` toward `q`.
- `V(s) < 0`, on prefixes that have already gone wrong: anti-distillation, which
  pushes `p` away from `q` and increases `KL(q||p)`.

Group centering does not remove this. Centering gives `Σ_i A_i = 0` per group, but
(II) depends on `V(s_t)`, the advantage of the *prefix*, which is far from zero
inside a trajectory. Per-token summation also weights sequences by length, so
`Σ_i A_i L_i ≠ 0`. With `plus_minus`, `zero_minus` or `plus_zero` rewards, `V` is not
even zero on average.

Magnitudes: on most tokens of an LLM rollout the choice of token hardly affects the
outcome, so `Q(s,·) ≈ V(s)` and (I) is about 0. (II) is still `|V(s)|·||q - p||`.
On those tokens the bias is not a small correction to the signal; it is all of the
update.

## 2. Why it compounds (systematic, not zero-mean)

(II) points along `±(Φ(p) - p)`. That direction is set by the mismatch mechanism
(fixed noise draw, deterministic rounding, KV quantization, lag), so it stays
consistent from step to step. In logit space one step does

    z_p ← z_p + η Σ V · (Φ(p) - p) + (reward-driven part).

The next sampler is `Φ` of the *updated* trainer, so it is displaced again by about
the same `δ = Φ(p) - p`. A fixed point of (II) would need `Φ(p) = p`, which never
holds. So the trainer integrates the perturbation: its displacement grows roughly
linearly with the number of steps, `Σ_k η V_k δ_k`.

- With `V > 0` this is repeated training on the trainer's own corrupted copy. Each
  step adds another dose of the corruption.
- With `V < 0`, `KL(q||p)` grows, which makes `q - p` larger, which makes (II)
  larger. This is a positive feedback loop.

Either way the policy drifts in a reward-independent direction until the reward
collapses. This matches the known facts: the mismatch acts as a bias, not as noise,
and it compounds because the sampler is refreshed from the trainer.

## 3. Why IS weights and clipping only slow it down

With a detached token weight `w_y = w(p_y/q_y)`, the reward-independent part of the
expected update becomes

    V(s) · ( Σ_y q_y w_y e_y - ⟨q, w⟩ p ) = V(s) · ( q∘w - ⟨q,w⟩ p ).

This is zero for all `V` **only if `q∘w ∝ p`**, that is, only if `w = p/q` exactly
on every token.

- **Plain token IS** (`w = p/q`) removes it in expectation. Its variance is governed
  by `E_q[ρ²] = Σ p²/q`, which is heavy-tailed exactly on the tokens where `q ≪ p`.
  So in practice it gets clipped (and it is clipped at `e^20` in this code).
- **TIS (clamp)**: on clamped tokens `q_y w_y = c·q_y < p_y`, which leaves
  `V · Σ_{ρ>c} (c q_y - p_y) (e_y - p)`.
- **MIS / drop**: dropped tokens remove `p_y` from `q∘w` completely.
- **PPO / DAPO / DPPO**: bands that depend on the sign of `A` remove a different part
  of `q∘w` for `A > 0` than for `A < 0`. Since `V` is built from both signs, this
  adds a further reward-independent residual.

Clipping more tightly moves `q∘w` further from `p`. That makes the residual drift
**larger**, which is why stronger token-level clipping can make things worse.
Sequence-level weights only rescale `A`, which changes `V` to some `V'`; the
per-token `V'(q - p)` structure stays. The unclipped trajectory ratio is unbiased,
but its variance grows like `exp(Σ_t KL)` with length.

## 4. The fix: center the score under the sampler

Replace the per-token score by

    s̃_t = log p(y_t|s_t) - Σ_v sg(q_v) log p(v|s_t) = log p(y_t|s_t) + H(q_t, p_t).

Its logit gradient is `(e_y - p) - (q - p) = e_y - q`.

**Drift removed exactly.** For any advantage function,

    E_q[ A (e_y - q) | s ] = Σ_y q_y Q (e_y - q) = Σ_y q_y (Q - V)(e_y - q) = (I).

(II) is gone, at every prefix and for every `V(s)`, with no assumption on the reward
mode or on group centering. In parameter space this is `J_s^T (I)`. The Jacobian
depends only on the prefix, so the identity carries through the model.

**Nothing else changes.** The added term `A · ∇H(q_t, p_t)` has logit gradient
`A (p - q)`. Its expectation is exactly `-(II)`, so it cancels the drift and adds no
other bias. On-policy, `∇ Σ_v p_v log p_v = Σ_v ∇p_v = 0` with `p` detached. So
`q = p` gives *exactly* the original REINFORCE gradient, sample by sample. The trainer
side always matches the on-policy algorithm.

**Variance is not inflated.** `||e_y - q||_2 ≤ √2`, the same bound as `e_y - p`.
There is no ratio, so nothing needs clipping. The size of the update does not depend
on how large the mismatch is.

**What remains is a reward-driven direction.** (I) is the sampler's own policy
gradient at the sampler's states. It is applied through the trainer's Jacobian
`J_p = J_q + O(||δ||)`. To first order, a step changes the sampler's return by
`η · gᵀ J_q J_pᵀ g`, which is positive whenever the mismatch in the Jacobian is
moderate. `J(p) = J(q) + O(mismatch)`. Every remaining error multiplies an
advantage-driven direction: it vanishes when `Q(s,·)` is flat, and its stationary
points are those of the sampler's reward. Nothing pushes `p` relative to `q` anymore,
so there is nothing to compound.

Equivalently: the softmax score `e_y - p` contains a built-in baseline, the
expectation of `e_y` under the policy. Off-policy, that baseline belongs to the
wrong distribution. The fix uses the expectation under the distribution the token
was actually drawn from.

### Combining with an IS correction (optional)

With a token-level `ISCorrection`, the loss uses

    s̃_t = w_y · ( log p(y_t) - Σ_v b_v log p(v) ),   b = q∘w / ⟨q, w⟩   (w, b detached).

The logit gradient is `w_y (e_y - b)`, and `E_q[w_y (e_y - b)] = q∘w - ⟨q,w⟩ b = 0`.
So the constant part of the advantage is again removed exactly, whatever the band,
clamp/drop or sign rule. The correction then only reweights *which* actions get
credit. With `w = p/q`, `b = p` and this reduces to plain token IS. Sequence-level
corrections rescale `A` as before; the centered score keeps them free of drift.

## 5. Implementation (`losses.py`)

`sampler_baseline` computes `Σ_v b_v log p_v` per position. `loss_fn_rl` uses it by
default (`correction=None`); no flag is needed.

- **The sampler's distribution is known only on its top-k ids** (k = 20, provided in
  the batch by `train_rl.py` and `dev/mini.py`). The unreported tail mass `1 - S` is
  spread like `p` over the tail: `q̂_v = (1 - S) p_v / (1 - P_K)` for `v ∉ K`. Then
  `Σ_{v∉K} q̂_v log p_v = (1 - S)/(1 - P_K) · Σ_{v∉K} p_v log p_v`. This needs no
  extra `[B, C, V]` tensors beyond the existing log-softmax, and it stays inside the
  rematerialized chunk.
- **Exactness and residual.** The construction is exact when k covers the vocabulary
  (checked numerically: `E_{y~q}` of the logit gradient is 0 to 1e-8). Otherwise the
  only remaining drift is `V · (q - q̂)` restricted to the tail, with
  `||q - q̂||_1 ≤ 2(1 - S)`. That is tiny when the top-20 tokens hold almost all of
  the sampler's mass, and it is small compared with the original `V · (q - p)` over
  the whole vocabulary.
- **Corrections.** With a correction, the token weights are evaluated on the top-k
  ids and on the tail. On the tail the ratio `p_v / q̂_v = (1 - P_K)/(1 - S)` is one
  number per position.
- **Fallback.** If a batch has no top-k logprobs, the loss uses
  `w_y log p_y - (1 - w_y) logsumexp(z)` with `w = p/q` (value reported as the IS
  score). Its logit gradient is `w_y e_y - p`, which also has mean zero under `q`
  (`Σ_y q_y (p_y/q_y) e_y = p`). This is unbiased, but it has the variance of
  importance sampling.
- **Unchanged parts.** Metrics, advantage handling, sequence-level corrections and
  the signature `loss_fn_rl(hidden, head, batch, correction=None, chunk=128)` are
  unchanged.

## Sanity check (CPU `dev/mini.py`, 600 steps, noise 0.3)

These runs are not required. They are included only as a check.

- Default vocabulary, seeds 0/1/2:
  - original loss: last-10 reward 0.58/0.61/0.55, eval about 0.67;
  - centered loss: 0.86/0.85/0.82, eval about 0.82.
- `MINI_EXTRA_TOKENS=40`, so the sampler's top-20 does not cover the vocabulary and
  the tail estimate is used:
  - original loss collapses to 0;
  - centered loss holds at 0.81.

## Configuration

No configuration change is needed: the sampler-centered score is active with the
default `rl.is.level=null`.

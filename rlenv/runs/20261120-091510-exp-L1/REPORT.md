# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## Summary

The REINFORCE loss uses the score `∇ log p(y)`. On-policy, the score has zero
mean, which makes the per-token gradient invariant to any part of the advantage
that depends only on the prefix. Under a sampler `q ≠ p` it does **not** have
zero mean: `E_{y~q}[∇ log p(y)] = -∇_θ KL(q ‖ p_θ)`. So every token adds a
reward-independent term `Ā(s) · (-∇ KL(q‖p))`, where `Ā(s)` is the mean
advantage at the prefix `s`. That term pulls the trainer toward the sampler
(`Ā > 0`) or pushes it away (`Ā < 0`). The sampler is rebuilt from the
trainer every step, so the term never goes away and adds up into a drift.

The fix is to **center the score under the sampler's own distribution**:

```
g = A · w_y · ( ∇ log p(y) − E_{v~m}[∇ log p(v)] ),     m_v ∝ q_v · w(p_v / q_v)
```

(`w ≡ 1` without an IS correction). The subtracted vector depends only on the
prefix, and `E_{y~q}[w_y(∇log p(y) − μ)] = 0` holds exactly. So the drift term
is removed for every `Ā(s)`, every mismatch and every IS correction. When
`q = p`, `μ = E_p[∇ log p] = 0` and the gradient is exactly the old one. The
change is active with default arguments (`correction=None`).

## 1. Setup

Take one completion position with prefix `s`, trainer logits `z = z_θ(s)`,
`p = softmax(z)`, and sampler distribution `q(·|s)`. The sampler's
noisy/int8/fp8/KV-quantized/stale weights all produce a `q` that differs from
`p` by a *deterministic* function of the current weights. The loss is
`L = -(1/N) Σ_i A_i Σ_t w_{i,t} log p(y_{i,t} | s_{i,t})`. Its per-token
stochastic gradient (ascent direction) is

```
ĝ = A · w_y · ∇_θ log p(y|s),     y ~ q(·|s).
```

In logit space `∇_z log p(y) = e_y − p`.

Let `A(s, y) = E[A | s, y]` (the continuation is sampled from `q`). Split it
into a prefix part and a token-dependent part:

```
A(s, y) = Ā(s) + a(s, y),     Ā(s) = E_{y~q}[A(s,y)],     E_{y~q}[a(s,y)] = 0.
```

`Ā(s)` is **not** zero, even with group-centered rewards. Group centering
makes the advantage mean zero *per prompt*, but at an internal prefix
`Ā(s) = V_q(s) − b(prompt)`. That is positive on prefixes that are on the way
to success and negative on prefixes that are already failing. Every token of a
failed rollout carries the same negative advantage. With `plus_minus`
rewards, `Ā` is not even zero on average.

## 2. What is wrong with the current loss

Without a correction (`w = 1`):

```
E_{y~q}[ĝ] = Σ_y q_y a(s,y) ∇log p_y   +   Ā(s) · Σ_y q_y ∇log p_y
           = (signal)                  +   Ā(s) · Σ_y (q_y − p_y) ∇log p_y
           = (signal)                  −   Ā(s) · ∇_θ KL(q ‖ p_θ)|_{q fixed}.
```

In logits, the second term is `Ā(s) · (q − p)`. On-policy it is zero, since
`Σ_y p_y ∇log p_y = ∇Σ p = 0`. That identity is the only reason REINFORCE
tolerates an uncentered per-prefix advantage. Off-policy, the term is a
**systematic bias**, not zero-mean noise:

* It does not depend on the reward signal `a`. It is present on every token,
  including tokens unrelated to the task, and its size is `|Ā|·|q − p|`.
* Its direction is set by the mismatch. `Ā > 0`: distil toward the sampler.
  `Ā < 0`: anti-distil, i.e. lower `p` on the tokens the sampler overweights
  and raise it elsewhere, which *increases* `KL(q‖p)`.
* **It compounds.** The sampler is `q = π_{T(θ)}`, with `T(θ) = θ + δ` (fixed
  noise), `Q(θ)` (quantization) or `θ_{k−s}` (staleness). It is refreshed
  from θ at every step, so `q − p ≈ J_z (T(θ) − θ)` is again non-zero after
  each update. Distilling toward a target that moves with you gives a drift
  of size `≈ η Ā δ` per step with no fixed point. Anti-distillation is a
  positive-feedback loop: the larger the mismatch, the larger the push that
  increases it. The drift is not the gradient of any reward-related
  objective, and it does not vanish when the reward gradient does. As
  learning saturates (`a → 0`), the drift dominates. The mismatch metric
  grows, rare tokens of the sampler (the large unlikely tail of an LLM
  vocabulary) get mass, and reward collapses.

**IS corrections and clipping.** With a weight `w̄_y`, the term becomes
`Ā(s) Σ_y q_y w̄_y ∇log p_y`. It is zero only for the exact ratio
`w̄ = p/q`, which has unbounded variance. Any truncation, clamp or drop band
gives a measure `q·w̄ ≠ p`, so the bias stays:

```
Σ_y q_y w̄_y ∇log p_y = Σ_y (q_y w̄_y − p_y) ∇log p_y .
```

Clipping moves `q w̄` toward `p` only on tokens inside the band. Tighter
token-level bands send more tokens back to `w̄ = 1` (clamp) or to `w̄ = 0`
(drop). Dropping removes tokens *selectively*: exactly those where `p` and
`q` disagree most. That leaves a measure `q w̄` that is systematically
reweighted against `p`. So stronger clipping can make the drift larger. This
matches the known observation: corrections slow the drift but never remove
it.

## 3. The fix: a sampler-side control variate

For each prefix, define the measure and its mean score

```
m_v = q_v w̄_v / Z,     Z = Σ_v q_v w̄_v,     μ(s) = Σ_v m_v ∇_θ log p_v ,
```

and use the per-token gradient

```
ĝ' = A · w̄_y · ( ∇log p_y − μ(s) ).
```

**Claim 1 (exact removal of the drift).** For any prefix function `Ā(s)`:

```
E_{y~q}[ w̄_y (∇log p_y − μ) ] = Σ_y q_y w̄_y ∇log p_y − μ Σ_y q_y w̄_y = Zμ − Zμ = 0,
```

so `E_{y~q}[Ā(s) · w̄_y(∇log p_y − μ)] = 0`. This holds exactly for any
`q`, any `p` and any weight function `w̄`: none, TIS, MIS, PPO/DAPO bands,
DPPO's TV band. `μ` depends on `s` only, so it is a valid control variate
and adds no bias of its own.

**Claim 2 (nothing else changes when there is no mismatch).** If `q = p` and
`w̄ = 1`, then `μ = E_p[∇log p] = ∇Σ_v p_v = 0`, so `ĝ' = ĝ` exactly. The
same holds for the exact ratio `w̄ = p/q`, where `m = p`. The fix only changes
the gradient in proportion to the mismatch.

**Claim 3 (what remains is pure signal).** Without a correction,
`E[a·μ] = μ E_q[a] = 0`, so

```
E_{y~q}[ĝ'] = Σ_y q_y a(s,y) ∇_θ log p_y = J_θᵀ ( q ⊙ a(s,·) )      (in logits: Σ q_y a_y (e_y − p) = q ⊙ a).
```

`q ⊙ a` is exactly the logit gradient of the sampler's own expected return,
`∂E_q[R]/∂z_q = q ⊙ (Q_q − V_q)`. The update is the sampler's policy gradient,
mapped through the trainer's Jacobian `J_θ = ∂z_θ/∂θ ≈ ∂z_{T(θ)}/∂θ`. It is
proportional to the advantage: it vanishes whenever the advantage carries no
information about `y` (`a ≡ 0`), and its fixed points are the stationary
points of the reward of the policy we actually sample from. For the noise
setting this is ascent on `J(θ + δ)`, and for quantization a
straight-through ascent on `J(Q(θ))`. Both are fixed objectives, and gradient
ascent on a fixed objective cannot drift without bound. Its inner product
with the on-policy direction `p ⊙ a_p` is `Σ_y q_y p_y a_y a_{p,y} > 0` for
small mismatch. So the mismatch now only rescales the useful signal
(a second-order effect). Before, it added a first-order drift that was
independent of the signal.

**Surrogate loss.** Since `Σ_v p_v ∇log p_v = 0`, the gradient of
`Σ_v sg(q_v w̄_v / Z) log p_v` is `μ`. The loss becomes

```
L = −(1/N) Σ_i A_i Σ_t [ w̄_{i,t} log p(y_{i,t}) − sg(w̄_{i,t} / Z_{i,t}) Σ_v sg(q_v w̄_v) log p_v ] .
```

The extra term is a cross-entropy against the (IS-reweighted) sampler
distribution, weighted by the advantage. Its gradient in logits is
`A w̄_y (e_y − Σ_v m_v e_v)`: the sampled token is reinforced against the
*sampler's* mean, not the trainer's.

Sequence-level corrections (`seq_mean`, `seq_sum`) multiply `A` by a
sequence weight. The centering removes the per-prefix component of `w_seq·A`
in the same way. Only the dependence of `w_seq` on `y_t` itself is left,
which is the intended correction.

## 4. Implementation (`losses.py`)

`loss_fn_rl` keeps its signature and chunked structure. In every chunk,
`sampler_centering` builds `m` from the sampler's top-k logprobs
(`sampler_topk_ids`, `sampler_topk_logprobs`, which `train_rl.py` and
`dev/mini.py` both pass, top-20 as inference engines return):

* On the top-k set `K` (which depends only on the prefix), `q_v` is known
  exactly.
* On the tail `v ∉ K`, `q_v` is approximated by `p_v · Q_tail / P_tail`, with
  `Q_tail = 1 − Σ_K q` and `P_tail = 1 − Σ_K p`. This matches the sampler's
  tail mass exactly and assumes only that the *relative* shape of the tail is
  the trainer's. `Σ_{v∉K} p_v log p_v` is computed as
  `Σ_all p log p − Σ_K p log p`, so only `[B, C, K]` gathers and one
  vocabulary reduction are added. Nothing extra of size `[B, C, V]` is kept.
* `w̄_v` is the same `is_weight` as for the sampled token, evaluated per
  vocabulary entry, and `w̄ = 1` with `correction=None`.
* If no top-k logprobs are in the batch, the known set is the sampled token
  alone. That still shrinks the bias, but less.

The remaining bias is `Ā Σ_{v∉K}(q_v − q̃_v) ∇log p_v`. It is bounded by the
mismatch *inside* the tail, which is small for top-20 of an LLM. With the full
vocabulary it is zero. A unit check confirmed this: the expected gradient
under `q` with constant advantage is about `1e-8` with the new loss and
`1e-1` with the old one. The on-policy gradient is identical to the old loss.

## 5. Why this is the right modification

The failure is a first-order bias term `Ā(s)·E_q[∇log p]`. It exists only
because the score is centered under the wrong distribution: `p`, while the
data come from `q`. Re-centering under the distribution that generated the
token is the minimal change that:

1. sets that term to zero exactly, with no clipping and no variance blow-up
   (no `p/q` ratio is needed at all);
2. leaves the on-policy estimator unchanged;
3. leaves the rest of the gradient as the sampler's policy gradient, which is
   tied to reward and has a fixed point.

So the closed loop "update → resample from a perturbed copy → update" no
longer has a reward-independent forcing term that accumulates step after
step.

## Settings

No configuration changes are needed. The modification is active with the
default `rl.is.level=null` and composes with any `rl.is.*` correction.

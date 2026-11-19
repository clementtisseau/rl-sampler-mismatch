# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## 1. Setup and notation

Fix one completion position with prefix `s`. Let `z ∈ R^V` be the trainer's logits there,
`p = softmax(z)` the trainer's next-token distribution, and `q` the sampler's distribution
(noisy, quantized, or stale weights). The token `y ~ q` is sampled, and the sequence it
belongs to gets the advantage `A`. `e_y` is the one-hot vector of `y`. In logit space the
gradient of the current per-token surrogate `A · log p(y)` is

    ∇_z [A log p(y)] = A (e_y − p).                                        (1)

(Every parameter update is this logit-space vector pulled back through `∂z/∂θ`, so if (1)
is biased then the parameter update is biased in the same way.)

For a fixed prefix, split the advantage into its conditional mean under the sampler
and a zero-mean remainder:
`Ā(s) = E_{y~q}[A | s]`, `A = Ā(s) + δ(y)`, `E_q[δ] = 0`.

## 2. What is wrong with the current loss

The expected logit update at this position is

    E_{y~q}[A (e_y − p)] = Σ_y q(y) δ(y) e_y  +  Ā(s) (q − p).              (2)

**On-policy (`q = p`).** The second term is `Ā (p − p) = 0`. The mean advantage at a
prefix, including any baseline or reward offset, adds no expected update. It only adds
variance. That is why REINFORCE allows a baseline at all: `E_p[∇ log p(y)] = 0`.

**Mismatched (`q ≠ p`).** The score `e_y − p` no longer has mean zero under the
distribution the tokens come from: `E_q[e_y − p] = q − p`. So there is an extra term

    drift(s) = Ā(s) (q − p) = −Ā(s) ∇_z KL(q ‖ p).                        (3)

It is a deterministic bias, not noise. It does not shrink with batch size. It is not
related to the reward signal: the term would be present even if every token at `s` had
the same advantage. In effect the loss secretly contains `−Ā(s) · KL(q‖p)` at every
position:

* `Ā(s) < 0` (prefixes that tend to fail, or any negative offset such as `plus_minus`
  or `zero_minus` rewards): the trainer **maximizes** `KL(q‖p)`. It moves logit mass
  away from the tokens the sampler prefers and toward tokens the sampler under-weights,
  which are typically the long tail of junk tokens.
* `Ā(s) > 0`: the trainer minimizes `KL(q‖p)`, so it distills the perturbed sampler.

Group centering does not remove the drift. It makes the advantages sum to zero only over
whole sequences of a group, and so only at the shared prompt (the first completion
token). At deeper prefixes, `Ā(s) = E[R | s] − b` is generically nonzero. Normalizing by
token count also tilts the token-weighted mean advantage, for example when failed
completions are longer.

**Why it compounds.** The sampler is rebuilt from the trainer at every step:
`q_t = Q(p_t)`, with the same perturbation map `Q` (fixed weight noise, the same
quantizer, a lag). Each step moves `p` by `η Ā (q_t − p_t)`. The next sampler applies `Q`
again on top of that, so the move is repeated in the same direction every step. The
bias accumulates linearly instead of averaging out like noise would. When `Ā < 0`, the
trainer grows `KL(q‖p)`. That puts more mass on tokens the sampler handles badly, which
makes `q − p` larger, which makes the drift stronger. This feedback loop is the collapse.

**Why IS and clipping only slow it down.** With a detached token weight `w(r)`,
`r = p/q`:

    E_q[w A (e_y − p)] = Σ_y q(y) w(y) A(y) e_y − E_q[w A] p,

so for a constant advantage `A ≡ Ā` it equals `Ā (Σ_y q(y) w(y) e_y − E_q[w] p)`. This
vanishes only if `q(y) w(y) ∝ p(y)` for all `y`, which
means `w = p/q` exactly. Any clamp or drop changes `q·w` to something like
`min(p, c q)` or `p · 1[r in band]`, so the bias does not vanish:
`Ā (Σ_y min(p_y, c q_y) e_y − E_q[w] p)`. A tighter token band moves `w` toward 1 (or 0).
That brings back the uncorrected `q − p` direction on exactly the tokens where `p` and
`q` disagree most. This is why stronger token clipping can make things worse. Unclipped
IS is unbiased per token, but its heavy-tailed variance (`p/q` is unbounded on tokens
`q` rarely samples) is what forces practitioners to clip in the first place.

## 3. The fix: center the score at the sampler, not the trainer

The bias comes from the `−p` in the score (the softmax normalizer evaluated under the
wrong distribution). Center the score at the distribution the token was actually drawn
from. Per token, the new surrogate is

    L_tok = A · ( log p(y) − Σ_v sg(q_v) log p_v ).                          (4)

Its logit gradient, using `Σ_v q_v = 1` and `∇_z log p_v = e_v − p`, is

    ∇_z L_tok = A (e_y − p) − A (q − p) = A (e_y − q).                       (5)

Taking the expectation over `y ~ q`:

    E_q[A (e_y − q)] = Σ_y q(y) A(y) e_y − Ā q = Σ_y q(y) (A(y) − Ā) e_y = Σ_y q(y) δ(y) e_y.   (6)

The drift term (3) is gone exactly. A constant advantage at any prefix now gives zero
expected update, as in the on-policy case. What remains is the reward-driven part of (2),
`Σ q δ e_y`. This is the policy-gradient direction for the prefix, weighted by the
sampler's distribution. It is a sound improvement direction, with the right sign
token-by-token, and it is not a bias toward or away from `q`.

**Properties.**

* *Exact on-policy reduction.* If `q = p`, the extra term in (4) is `Σ_v sg(p_v) log p_v`.
  Its gradient is `Σ_v p_v (e_v − p) = 0` identically, sample by sample. So the gradient
  is bit-for-bit plain REINFORCE (checked numerically: max difference 3e-8).
* *Bounded.* The correction has norm at most `|A| · ‖q − p‖`. There are no ratios and no
  heavy tails, so nothing needs clipping.
* *Interpretation.* (4) is REINFORCE with a per-token, action-independent baseline for
  the score function: `log p(y) − E_{v~q}[log p_v]`. The baseline is detached in the
  distribution and is chosen so that the score has zero mean under the behaviour
  distribution `q`. That is the condition the derivation in section 2 shows is violated.
  Equivalently, it is the REINFORCE update with the trainer's normalizer `p` replaced by
  the sampler's `q` in (1).

**With an IS correction (optional, still drift-free).** For a token weight `w(y)`, the
weighted score `w (e_y − m)` has zero mean under `q` iff `m = q w / E_q[w]`. So with a
correction, the loss centers at `m_v ∝ q_v w(p_v/q_v)`:

    L_tok = A · w(y) · ( log p(y) − Σ_v sg(m_v) log p_v ),   E_q[w A (e_y − m)] = Σ_y q w (A − Ā_w) e_y,

where `Ā_w = E_q[wA]/E_q[w]`. The drift is zero for any band, clamp, drop, or sign rule.
For unclipped IS (`w = p/q`), `m = p` and the loss reduces to ordinary naive IS.
Sequence-level weights are constant along a sequence and do not change the centering.

## 4. Implementation (`losses.py`)

* `q` over the vocabulary is built from the sampler's top-k logprobs, which both
  trainers already put in the batch (`sampler_topk_ids`, `sampler_topk_logprobs`, k=20,
  as inference engines return). The `k` known entries are exact. The leftover mass
  `1 − Q_K` is spread like the trainer's distribution:
  `q̂_v = τ p_v` for `v ∉ K`, with `τ = (1 − Q_K)/(1 − P_K)`. This keeps the total mass
  exact. The only residual drift is `Ā (q − q̂)`, restricted to the tail where both
  distributions are small. When `k ≥ V` (as in `dev/mini.py` with the default vocabulary)
  the drift is exactly zero (checked: expected logit update with constant advantage
  went from 0.35 to 1e-7 in L1).
* `Σ_v q̂_v log p_v = Σ_{k∈K} q_k log p_k + τ (Σ_v p_v log p_v − Σ_{k∈K} p_k log p_k)`,
  with `p` detached inside the coefficients. This needs only gathers, so it works with a
  TP-sharded vocabulary and inside the existing rematerialized chunks. There is no
  `[B, C, V]` scatter.
* The centering is on by default (`correction=None` → `m = q̂`). With a token-level
  `ISCorrection`, `m ∝ q̂ · w` (the tail ratio `p/q̂ = 1/τ` is constant, so `w` is
  constant there; for `stat=tv` the tail uses the total tail TV). If a batch has no
  top-k sampler logprobs, the loss falls back to plain REINFORCE.
* New metric `tv_samp_train_topk`: `½ Σ_{k∈K} |q_k − p_k|`, the size of the drift
  direction that is now removed.

## 5. Sanity checks (CPU `dev/mini.py`, not required)

* Default setting (`plus_minus` rewards, noise 0.3): the original loss plateaus and then
  declines, with eval reward 0.72 → 0.66. The centered loss keeps improving, with eval
  reward reaching 0.83 to 0.86.
* `MINI_EXTRA_TOKENS=40` (a long junk tail, so top-k does not cover the vocabulary): the
  original loss collapses to reward 0 within 25 to 50 steps on two seeds. The centered
  loss trains to final rewards of 0.87 and 0.75. Combined with TIS, it is also slightly
  better than TIS alone.

## Configuration

No configuration changes are needed. The modification is active with the default
`rl.is.level=null` (`correction=None`).

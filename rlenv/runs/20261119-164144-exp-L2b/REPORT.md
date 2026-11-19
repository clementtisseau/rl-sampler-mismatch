# Sampler-centered score: removing the mismatch bias from REINFORCE

## 1. Setup at one token position

Fix a prefix `s` (prompt plus the completion so far). Let `z` be the trainer's
logits at `s`, `p = softmax(z)` the trainer's next-token distribution, and `q` the
sampler's. The next token is drawn as `y ~ q`. The sequence advantage is `A`.
Given `(s, y)` its conditional mean is `Q(y) := E[A | s, y]`, where the expectation
is over the rest of the rollout, which also comes from the sampler. Let
`V := E_{y~q}[Q(y)]` be the prefix value. `w(y) = w(p_y/q_y)` is the detached IS
weight; `w ≡ 1` without a correction.

The current loss is `-(1/N) Σ_i A_i Σ_t w(y_t) log p(y_t)`. Up to the positive
constant `1/N`, its descent direction at this position is `A w(y) ∇_θ log p(y)`. In
logit space:

    ∇_z log p(y) = e_y - p                                   (1)

and `∇_θ = J^T ∇_z` with `J = ∂z/∂θ`. Because `J` is the same for every `y` at a
fixed prefix, it is enough to work in logit space.

## 2. What is wrong with the current loss

Take the expectation over `y ~ q` and split `Q = V + (Q - V)`:

    g := E_q[ Q(y) w(y) (e_y - p) ]
       = Σ_y q_y w_y (Q_y - V)(e_y - p)   +   V · μ,        μ := Σ_y q_y w_y (e_y - p)   (2)

The first term is the learning signal. The second term is `V` times the
**sampler-mean of the weighted score**, `μ = E_q[w ∇_z log p]`.

**On-policy (`q = p`, `w = 1`).** `μ = Σ_y p_y (e_y - p) = p - p = 0`: the score has
zero mean. Any baseline, and any prefix-dependent offset of the advantage (in
particular the value `V(s_t)` of every prefix), has no effect in expectation.
REINFORCE with group-centered rewards relies on exactly this property.

**Mismatched, no correction (`w = 1`).**

    μ = Σ_y q_y (e_y - p) = q - p = -∇_z KL(q || p)        (3)

The expected update therefore contains `V · (q - p)`, or in parameter space
`-V ∇_θ KL(q || p_θ)`, with `q` held fixed. This term does not depend on which token
was good. It is a reward-independent force that:

* pulls `p` toward the sampler on prefixes with `V > 0`. The trainer then imitates
  the sampler's error, the weight noise or quantization error;
* pushes `p` *away* from the sampler on prefixes with `V < 0`, by ascending
  `KL(q||p)`. This is unbounded: nothing stops it.

`V(s_t)` is rarely zero. Group centering makes `Σ_i A_i = 0` within a group, so the
term cancels only at the first completion token, where all members share the
prefix. At later positions the prefixes differ, and `V(s_t) = E[A | s_t]` is the
value of a partial solution: positive along good trajectories and negative along
bad ones. For uncentered rewards (`plus_minus`, ...) `V ≠ 0` even at the first
token. Normalizing by the token count also gives long rollouts, often the failed or
truncated ones, more biased terms.

**Why it compounds.** Take weight noise `θ_q = θ + δ` as an example: `δ` is fixed
and the sampler is rebuilt from the trainer every step. To first order,
`Σ_y q_y ∇_θ log p_θ(y) ≈ F(θ) δ` with `F` the Fisher matrix. The bias step is then
`η V F(θ) δ`. After the refresh the new sampler is `θ' + δ`, the same offset, so
the next step adds a push along the same direction again. Deterministic
quantization behaves the same way: rounding errors are a fixed function of the
weights, not fresh noise. The bias is a constant force, not zero-mean noise, and
it accumulates linearly with the number of steps. A random walk would grow only as
√steps. It also keeps acting when the learning signal disappears. For example,
when `Q` is constant the first term of (2) vanishes, but `V μ` does not, so the
reward-optimal policy is not a fixed point of the expected update.

**Why IS and clipping only slow this down.** With the exact ratio `w = r = p/q`,
`μ = Σ_y p_y (e_y - p) = 0`, so the bias goes away, but `r` is unbounded and has
very high variance. Bounded weights bring the bias back. Write `m_y = q_y w_y`
and `M = Σ_y m_y`. Then

    μ = m - M p,                                             (4)

which is zero only if `m ∝ p`. With TIS (`w = min(r, c)`), `m_y = min(p_y, c q_y)`.
With drop-style bands (MIS/PPO/DAPO/DPPO), `m_y = p_y` in the band and `0` outside.
In both cases `μ = (p restricted to the kept mass) - (kept mass)·p`. Its size grows
with the probability mass that is clipped or dropped. A tighter token-level band
removes more mass and so leaves a *larger* reward-independent drift `V μ`. That is
why stronger clipping can make things worse. Sign-dependent bands (`sign=ppo`)
also make `w`, and therefore `μ`, depend on the sign of `A`, so the bias differs
between positive and negative rollouts.

## 3. The fix: center the score under the sampler

Replace the per-token score by its **q-centered** version:

    ĝ(y) = w(y) ∇ log p(y) - Σ_k q_k w_k ∇ log p_k           (5)
         (logits:  w_y (e_y - p) - μ)

The subtracted term is a deterministic function of the prefix, the same for every
`y`. Hence `E_q[ĝ] = 0` for **any** sampler `q` and **any** weight function `w`.
This restores the zero-mean property of the on-policy score. The expected update
becomes

    E_q[ Q(y) ĝ(y) ] = E_q[Q w (e_y - p)] - V μ = Σ_y q_y w_y (Q_y - V)(e_y - p)   (6)

Only `Q - V` enters. Any prefix-dependent shift of the advantage (`V(s_t)`, the
group mean, the reward offset of `plus_minus`) has no expected effect, exactly as
on-policy. The `-V ∇KL(q||p)` drift is gone. When the learning signal vanishes
(`Q` constant) the expected update is exactly 0, so the trainer has no reason to
walk along `δ`.

Notable special cases:

* `w ≡ 1` (default, `correction=None`): `ĝ = e_y - q` in logit space, and
  `E_q[Q ĝ] = Σ_y q_y (Q_y - V) e_y`. This is exactly the on-policy REINFORCE step
  the *sampler* would take on its own logits, applied to the trainer's logits. The
  update moves logits toward better tokens and never toward or away from `q` for
  its own sake. The leftover difference from the on-policy gradient is the
  weighting `q_y` instead of `p_y` inside the learning signal, which is
  `O(|q - p|)` *times the signal*, not a stand-alone drift.
* `w = r`: `μ = 0`, so nothing changes, and IS stays unbiased.
* Clipped or dropped `w`: the bias (4) that clipping introduced is removed exactly,
  so TIS, MIS, PPO, ... can be used on top for variance control without paying for
  it in bias.
* **On-policy (`q = p`)**: `Σ_k p_k w(1) ∇log p_k = ∇ Σ_k p_k = 0`, so the loss has
  *exactly* the gradient of plain REINFORCE. Nothing changes when there is no
  mismatch.

Equivalently, the per-token objective is

    A · [ w(y) log p(y) - Σ_k sg(q_k w_k) log p_k ] = A · [ w(y) log p(y) + CE_w(q, p) ],

an advantage-weighted cross-entropy to the sampler. Its single-sample expectation
`V·∇CE(q,p)` cancels the implicit `-V·∇CE(q,p)` contained in REINFORCE under `q`.
`E[A μ | s] = V μ` because `μ` is fixed given the prefix, so the sequence advantage
`A` is an unbiased stand-in for the unknown `V(s_t)`. No value function is needed.

## 4. Computing it from what the sampler returns

`μ` needs `q` on the whole vocabulary. The sampler, like real inference engines,
returns its top-K logprobs per position (`sampler_topk_ids`,
`sampler_topk_logprobs`, K = 20). The loss uses the top-K set `𝒦` as individual
outcomes and lumps everything else into one tail outcome `T`:

    q_T = 1 - Σ_{k∈𝒦} q_k,     p_T = 1 - Σ_{k∈𝒦} p_k,     ∇log p_T = -(1/p_T) Σ_{k∈𝒦} p_k ∇log p_k

The last identity holds because `Σ_v p_v ∇log p_v = 0` over the full vocabulary, so
no `[B, C, V]` mask is needed. The baseline term is

    b = Σ_{k∈𝒦} sg(q_k w_k) log p_k + sg(q_T w_T) · ( -Σ_{k∈𝒦} sg(p_k) log p_k / p_T ),

where `w_k = w(p_k/q_k)` is computed with the same correction (band, sign) as the
sampled token, and `w_T = w(p_T/q_T)`. The per-token score becomes `w(y) log p(y) - b`.

* If 𝒦 covers the support, (5) is exact: `E_q[ĝ] = 0`. This was checked
  numerically by enumerating all tokens.
* Otherwise the only approximation is `q|_T ∝ p|_T` inside the tail. The remaining
  bias is `V q_T w_T (q|_T/q_T - p|_T/p_T)`, bounded by `2|V| q_T w_T` and usually
  much smaller. This is a second-order quantity compared with (3).
* On-policy, the lumped version is still exact: `Σ_𝒦 p_k ∇log p_k + p_T ∇log p_T = 0`.
  The gradient therefore equals plain REINFORCE exactly, even with truncated top-K.
  This was checked numerically to 1e-9.
* `p_T < 1e-4` is treated as zero tail, since `1 - Σ p_k` is float32 noise there.
  Masked or padding positions are excluded with `where`, so stale top-K values
  there cannot produce NaNs.
* Sequence-level corrections (`seq_mean`, `seq_sum`) still multiply the
  advantage. Token centering uses `w = 1` for them.
* If a batch has no top-K logprobs, the loss falls back to the original
  REINFORCE (no centering). `train_rl.py` always provides them.

The change is active with default arguments (`correction=None`). The signature
and the metrics are unchanged; the metrics gain `q_tail`, the mean sampler mass
outside the top-K.

## 5. Variance

`ĝ` subtracts the conditional mean of the score, a control variate. Unlike the IS
ratio it is bounded: in logit space `|μ|_1 ≤ 2 Σ_k q_k w_k`, so it adds no heavy
tails. The extra cost is one top-K gather from logits that are already computed.

## Configuration

No configuration change is required: the centering is on by default and works
with `rl.is.level=null`. It is also compatible with every `rl.is.*` correction.

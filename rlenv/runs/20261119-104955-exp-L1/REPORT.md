# Sampler-centered score: removing the mismatch drift from REINFORCE

## Summary

The REINFORCE loss uses the trainer's score `∇ log p(y)` on tokens drawn from a
different sampler `q`. The method depends on the score having zero mean, and
that holds only under `p`. Under `q` the score has mean `q − p` in logit space.
So every token's update gets an extra term `V^q(s) (q − p)` that does not depend
on which token was good. It pushes `p` away from `q` (or toward it), scaled by
the value of the state. The term is systematic, so it builds up step after step.
Because `q` is re-derived from `p` every step, it never settles.

The fix is to center the score at the distribution the tokens were actually
drawn from. For one position,

    score(y) = w · ( log p(y) − Σ_v sg(μ_v) log p(v) ),     μ ∝ q · w(p/q)

Here `w` is the detached IS weight (`w ≡ 1` when `correction=None`), and `μ` is
the normalized effective sampling distribution. Its logit gradient is
`w (e_y − μ)`, which has **exactly zero mean under the sampler for any `q` and
any weighting `w`**. The drift term goes away and only the real learning signal
`Cov_μ(Q, e_y)` is left. When `q = p` and `w = 1`, μ = p, so the loss is exactly
the original REINFORCE (checked numerically: the gradients match to 2e-8).

## 1. What is wrong with the current loss

Take one generated position with state (prefix) `s` and trainer logits `z`.
Write `p = softmax(z)` and let `q` be the sampler's next-token distribution. The
loss contributes `−A · log p(y)` with `y ~ q`. On the logits,

    ∇_z log p(y) = e_y − p.

Let `Q(s, y) = E[A | s, y]` be the advantage expected after taking `y` in `s`
(continuations are also sampled by `q`), and let `V(s) = E_{y~q}[Q(s, y)]`. The
expected update at this position is then

    E_{y~q}[ Q(s,y) (e_y − p) ]
      = Σ_y q_y Q(s,y) (e_y − q)  +  V(s) (q − p)                     (1)
      =        Cov_q(Q, e_y)       +  V(s) (q − p).

* The first term is the correct policy gradient: the gradient of
  `E_{y~softmax(z)}[Q]` at the distribution that produced the data. It is zero
  when all actions are equally good.
* The second term is the bias. It exists because `E_{y~q}[∇_z log p(y)] = q − p ≠ 0`.
  It does not depend on how the actions are ranked; it only depends on the
  state's value. In parameter space it is

      V(s) · ∇_θ Σ_y q_y log p_θ(y) = −V(s) · ∇_θ KL(q ‖ p_θ).

  So in states with `V < 0` (failing prefixes; under `plus_minus` this is most
  states early in training) the loss climbs `KL(q‖p)`: it pushes `p` *away* from
  the sampler. In states with `V > 0` it pulls `p` toward `q`. Either way it moves
  `p` along `±(q − p)`, which is set by the mismatch and not by the reward.

This also breaks REINFORCE's invariance to the baseline. Shifting all rewards by
`b` adds `b (q − p)` to every position. A "free" baseline such as group
centering, or a choice between `plus_minus`, `plus_zero` and `zero_minus`, now
changes the expected update. Group centering removes the term at the first
token, where all G rollouts share the prompt and `Σ_i A_i = 0`. It does not
remove it later in the sequence, where each prefix has its own conditional value.

**Why it compounds.** The sampler is `q = q(θ + δ)` with a fixed perturbation `δ`
(or a fixed quantizer or staleness), refreshed from `θ` every step. So `q − p` has
a persistent direction, roughly `J_z δ` to first order. The bias is
`η V(s) (q − p)` per step, so its effect grows linearly in the number of steps,
like `t`, while zero-mean noise grows only like `√t`. There is no fixed point.
Moving `p` toward or away from `q` also moves `q`, because `q` is rebuilt from the
new `θ`, so `p` follows `q` without end. With `V < 0` there is positive feedback:
the update raises `p` where `p > q` and lowers it where `q > p`. That increases
`|q − p|`, which makes the bias larger. In particular, probability mass goes to
the many vocabulary tokens that the sampler never samples (the tail), because
their score is never directly pushed down. Once the sampler starts emitting them,
reward collapses.

**Why IS weights and clipping do not fix it.** A detached token weight `w(r)`,
with `r = p/q`, gives

    E_{y~q}[ w (e_y − p) ] = Σ_y q_y w_y e_y − p Σ_y q_y w_y = Z (μ − p),
    μ_y = q_y w_y / Z,  Z = E_q[w].                                     (2)

The drift is then `V · Z (μ − p)`. It vanishes only if μ = p, which means an
untruncated weight `w = p/q`. That weight has unbounded variance, and the
framework clips it to e^±20. Every practical correction (TIS clamp, MIS/PPO/DAPO
drop bands, DPPO) changes `w` exactly on the tokens where `p` and `q` differ
most. That leaves μ between `q` and `p`, so it only shrinks the drift and slows
the degradation. Tighter token bands make μ more like the band-limited sampler
and less like `p`. Dropping tokens (`w = 0`) also removes mass from μ in a
data-dependent way. That is why stronger token-level clipping can make things
worse. Sequence-level weights scale the advantage but leave the token score
`e_y − p` unchanged, so the `(q − p)` term stays.

## 2. The modification

The idea is to keep the weight `w` (whatever correction is used, including none)
and replace the per-token baseline `p` inside the score with the effective
sampling distribution `μ ∝ q · w(p/q)`:

    g_t = A · w_t · (e_{y_t} − μ_t)            (logit space)

The surrogate whose gradient is this (μ detached) is

    L = −(1/N) Σ_i A_i Σ_t w_t [ log p(y_t) − Σ_v sg(μ_t(v)) log p(v) ]
      = −(1/N) Σ_i A_i Σ_t w_t [ log p(y_t) + H(μ_t, p_t) ],

because `∇_z Σ_v μ_v log p_v = μ − p` when `Σ μ = 1`. It is REINFORCE plus an
advantage-weighted cross-entropy to the sampler distribution. For `A < 0` tokens
it pulls `p` back toward `μ` by exactly the amount that the plain score was
pushing it away, on average.

**Zero mean, for any q and w.**

    E_{y~q}[ w_y (e_y − μ) ] = Σ_y q_y w_y e_y − μ Σ_y q_y w_y = Zμ − Zμ = 0.

**Expected update.**

    E_{y~q}[ Q(s,y) w_y (e_y − μ) ] = Z Σ_y μ_y Q(s,y) (e_y − μ) = Z · Cov_μ(Q, e_y)
                                    = Z · ∇_z E_{y~softmax(z)}[Q(s,y)] |_{softmax(z)=μ}.

Compared with (1) and (2), the `V(s)(·−p)` term is gone. The update is a
policy-improvement step of the effective sampling policy, expressed in the
trainer's logits:

* Shifting rewards by a constant `b` now changes nothing in expectation, because
  `E_q[w(e_y − μ)] = 0`. The baseline invariance of on-policy REINFORCE is restored.
* If all actions at a state are equally good (`Q` constant), the expected update
  is zero. A converged or flat region gives no drift, so the "p chases q forever"
  dynamics has no driving force.
* The update is a covariance. It raises tokens that are better than average under
  μ and lowers worse ones. It cannot move mass into tokens μ does not cover
  (vocabulary tail) except through their real advantage.
* It still carries the usual differences from the true on-policy gradient
  `Cov_p(Q^p, e_y)`. The continuation values and state distribution come from `q`,
  and the action distribution is μ instead of `p`. These differences are
  proportional to the mismatch *and* to how strongly `Q` varies. They act like a
  slightly off-policy improvement step, not a reward-independent force, and they
  vanish when the policy stops improving. Adding an IS weight (`rl.is.*`) moves μ
  toward `p`, and the centering keeps such a weight drift-free whether or not it
  is clipped.
* At `q = p`, `w = 1`: μ = p and the loss equals the original one exactly.

The same score is used with sequence-level corrections (`seq_mean`, `seq_sum`).
There the token-level μ is `q`, and the sequence weight multiplies the
advantage. The drift-free property then holds for each token, given the
sequence weight.

## 3. Computing μ from what the sampler returns

Inference engines, and both `train_rl.py` and `dev/mini.py`, return the
sampler's top-k logprobs per position (`sampler_topk_ids`,
`sampler_topk_logprobs`, k = 20), not its full distribution. For the top-k set
`K` we know `q_v` exactly and compute `p_v` and `w(p_v/q_v)` from them. The
remaining sampler mass `Q_tail = 1 − Σ_K q_v` is spread over the tail in
proportion to `p`. That is the only assumption that keeps the trainer's ranking
of the tail. With this, the tail has a common ratio `r_tail = P_tail / Q_tail`:

    m_v    = q_v · w(p_v / q_v)                   v ∈ K
    m_tail = Q_tail · w(P_tail / Q_tail)          (spread ∝ p_v over v ∉ K)
    Z = Σ_K m_v + m_tail,   μ_v = m_v / Z (v ∈ K),   μ_v = α p_v (v ∉ K),  α = m_tail / (Z P_tail)

The baseline is computed with one full-vocabulary reduction and one top-k gather:

    Σ_v μ_v log p_v = α Σ_v p_v log p_v + Σ_{v∈K} (μ_v − α p_v) log p_v.

This runs inside the existing rematerialized vocabulary chunks (`centering_logp`
in `losses.py`), so it adds no `[B, T, V]` memory. `w` comes from the same
`log_mass` as the sampled-token weight, so μ follows whatever `rl.is.*` grid point
is chosen. Masked positions are dropped with `where` instead of multiplication,
so stale top-k rows outside the completion can never produce NaNs.

If the batch has no top-k fields, the loss falls back to `K = {y}`, the sampled
token with its known `q_y`. This is only an approximation of μ.

The default (`correction=None`) gives μ = q and the score
`log p(y) − Σ_v q_v log p(v)`. No flag is needed.

## 4. Check on the CPU proxy (`dev/mini.py`, 600 steps)

| setting | original loss | centered loss |
| --- | --- | --- |
| default (noise 0.3, plus_minus), seed 0 | train reward 0.58 at the end, eval 0.66 | train 0.84, eval 0.80 |
| + 50 extra vocabulary tokens (tail outside top-20), seed 1 | collapses to 0 by step 100 | train 0.84, eval 0.82 |
| 50 extra tokens, group_centered, seed 2 | eval 0.74 | eval 0.81 |
| noise 0.5, group_centered, seed 1 | eval 0.59 | eval 0.68 |
| 50 extra tokens, TIS (high=2 clamp), seed 1 | eval 0.76 | eval 0.83 |

When the top-k fields are removed (the fallback), the extra-token run still
collapses, at about step 400 instead of 100. The top-k logprobs are what make μ
accurate.

## Configuration notes

No configuration change is needed. The centered score is active with the default
`rl.is.level=null` (`correction=None`), and it also combines with any `rl.is.*` correction.

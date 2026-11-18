# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The REINFORCE loss uses the score `grad log p(y_t | s_t)`. It is zero-mean
only when `y_t` is drawn from `p`. When the rollouts come from a different
sampler `q`, its conditional mean is
`mu_t = E_{v~q}[grad log p_v] = -grad_theta KL(q_t || p_t)`, which is not zero.
Every advantage-weighted token then also carries a deterministic term
`A * mu_t`. That term pushes `p` away from `q` wherever the expected advantage
is negative. The sampler is rebuilt from the trainer at every step and keeps
the same offset, so the push happens in the same direction every step. It
compounds, and this is the collapse.

The fix subtracts that conditional mean from every token's score. This is an
exact control variate under `q`:

    L = -(1/N) sum_i A_i sum_t [ w_t log p(y_t|s_t) - sum_v sg(q_v w_v) log p(v|s_t) ]

Here `w` is the (optional) IS weight and `w = 1` by default. On-policy
(`q = p`, `w = 1`) the subtracted term has zero gradient, so the loss is
exactly REINFORCE. Off-policy, each token's expected update is exactly zero
unless its advantage depends on which token was sampled. Seen at the logits,
the per-token update `A (e_y - p)` becomes `A (e_y - q)`: the trainer gets
the sampler's own score function, which is zero-mean under the samples.

## 1. Setup

A rollout `y = (y_1..y_T)` is drawn from the sampler `q`, with states (prefixes)
`s_t`. `p = p_theta` is the trainer. The current loss is
`L = -(1/N) sum_i A_i sum_t w_t log p(y_{i,t}|s_{i,t})`, with `w_t = 1` when no
correction is used. Its gradient estimator is

    g = sum_t A * w_t * grad log p(y_t | s_t).

`p = softmax(z)` over the vocabulary, so `grad_z log p_v = e_v - p`.

## 2. Expected update: on-policy vs. mismatched

**Score mean.** For a fixed state `s`:

    mu(s) := E_{v~q(.|s)}[grad log p(v|s)] = sum_v q_v grad log p_v
           = grad_theta sum_v q_v log p_v = -grad_theta KL(q(.|s) || p(.|s))   (q held fixed)

In logit space, `mu_z = sum_v q_v (e_v - p) = q - p`.

* On-policy (`q = p`): `mu = sum_v p_v grad log p_v = grad sum_v p_v = 0`. This
  is the score-function identity that REINFORCE and every baseline rely on.
* Mismatched (`q != p`): `mu = q - p` in logits, which is nonzero. Its size is
  first order in the mismatch.

**Decomposition.** Write `Q_t(s_t, y_t) = E_q[A | s_t, y_t]` and
`V_t(s_t) = E_q[A | s_t]`. The score is a function of `(s_t, y_t)` only, so

    E_q[g] = sum_t E_{s_t~q} [ E_{y_t~q}[ Q_t * grad log p(y_t) ] ]
           = sum_t E_{s_t} [ Cov_{y_t~q}(Q_t, grad log p(y_t)) ]   (signal S)
           + sum_t E_{s_t} [ V_t(s_t) * mu(s_t) ]                  (drift D)

* On-policy: `mu = 0`, so `D = 0` and `E[g]` is the policy gradient.
* Mismatched: `D = -sum_t E_{s_t}[ V_t(s_t) grad KL(q_t || p_t) ]`.
  This is the gradient of `sum_t E[ -V_t KL(q_t||p_t) ]`:
  - at states with `V_t > 0`, `p` is pulled toward `q`. This is bounded:
    `KL >= 0`, and the pull stops at `p = q`;
  - at states with `V_t < 0`, the loss maximizes `KL(q_t || p_t)`, or
    equivalently the cross-entropy `H(q_t, p_t) = -sum_v q_v log p_v`, which is
    unbounded. Tokens the sampler likes more than the trainer
    (`q_v > p_v`) get pushed down (`Delta z = |V| (p - q)`). This repeats
    for as long as `q != p`.

  `D` does not depend on the reward signal. It acts like an extra objective
  that the mismatch adds.

**Why it compounds.** The sampler is `q_theta = f(theta + delta)` (noise,
quantization, staleness): a fixed transformation of the current trainer.
After every update the sampler is refreshed, the offset `q - p` comes back
with roughly the same direction, and `D` pushes the same way again. This is a
systematic bias accumulated over every step, not zero-mean noise. Variance
would average out over steps; this does not.

**Why baselines do not help.** Group-centering (or any per-prompt baseline `b`)
changes `A` by a constant per sequence. That changes `V_t` by `-b` and leaves
`D` nonzero. Centering removes `D` only if `V_t(s_t) = 0` at every
state, which a sequence-level baseline cannot do. With `plus_minus` rewards
`V_t < 0` almost everywhere at the start, which gives the strongest drift
(this is the dev setting). With group-centering, `D` is smaller but nonzero:
failing prefixes have `V_t < 0`.

**Why IS weights and clipping only slow it.** With a detached token weight `w`,
the conditional mean of the weighted score is

    mu^w = E_{v~q}[w_v grad log p_v] = sum_v (q_v w_v - p_v) grad log p_v ,

which vanishes for every `theta` only if `q_v w_v = p_v` for all `v`, i.e.
`w = p/q` with no truncation. Any clamp, drop, sign rule or PPO band changes
`w` away from `p/q` at the tokens where the mismatch is largest. Those tokens
then keep a residual `mu^w != 0`, so `D` is smaller but still there and still
compounds. Clipping more tightly moves `w` further from `p/q` on more tokens, so
the residual drift can get bigger: "stronger clipping can make it worse".
Unclipped token IS does make `mu^w = 0`, but its variance grows with
`chi^2(p||q)` (`E_q[w^2] = sum p^2/q`), which is unbounded for heavy-tailed
mismatch. Sequence-level weights only rescale `A`, so `D` becomes
`E[w_seq V_t mu_t]`, still nonzero. If `w_seq` depends on `p(y_t)/q(y_t)`, the
weight is also correlated with the token, which adds a selection bias.

## 3. The fix: center the score under the sampler

`mu_t` is a deterministic function of the state, so it can be subtracted
from the score without changing the signal:

    g_c = sum_t A [ w_t grad log p(y_t|s_t) - mu^w_t ],
    mu^w_t = sum_v q(v|s_t) w_v grad log p(v|s_t).

**Proposition.** For any `q`, any `w` and any advantage (with or without a
baseline):

    E_q[g_c] = sum_t E_{s_t}[ Cov_{y_t~q}(Q_t, w grad log p(y_t)) ]   and the drift is 0.

*Proof.* `E_{y_t~q}[w_t grad log p(y_t) - mu^w_t | s_t] = mu^w_t - mu^w_t = 0`.
So the `V_t` part of `Q_t` contributes nothing, and only `Q_t - V_t` multiplies
the centered score. With `w = 1`: in logits,
`E_{y~q}[(Q(y) - V)(e_y - q)] = q ⊙ (Q - V_q)` with `V_q = sum_v q_v Q_v`.
This is the exact softmax policy gradient of `E_{v~q}[Q_v]`, written in terms
of the sampler's logits and applied to the trainer's logits. ∎

Properties:

1. **Same as REINFORCE on-policy.** With `q = p` and `w = 1`,
   `mu_t = sum_v p_v grad log p_v = 0`, and the gradient is unchanged. The loss
   value changes by `A * H(p)`, but that term has zero gradient.
2. **No drift.** A token whose advantage does not depend on which token was
   sampled (e.g. the constant `-1` of a failed rollout in `plus_minus`, or
   `V_t` in general) now produces an expected update of exactly zero. The
   fixed points are those of the reward signal, not of `KL(q||p)`. So nothing
   systematic is fed back through the sampler refresh, and there is nothing
   to compound.
3. **No variance blow-up.** Unlike IS, no ratio is introduced. With `w = 1`,
   the per-token logit update is `A (e_y - q)`, with `||e_y - q|| <= sqrt(2)`.
   It is a control variate, not a reweighting. Its effect grows with
   the mismatch and vanishes on-policy.
4. **Loss form.** `grad sum_v sg(q_v w_v) log p_v = mu^w`, so the loss is
   `-A [w_t log p(y_t) + H_{qw}(p)]`, where `H_{qw}(p) = -sum_v sg(q_v w_v) log p_v` is a
   (detached-weight) cross-entropy. The term exactly cancels the implicit
   `+-A * H(q,p)` objective that the mismatch adds.
   `log_mass(log p, log q, c)` in `losses.py` already gives `log(q w)`, so
   any `rl.is.*` correction is centered consistently. For naive IS, `qw = p`
   and the centering is zero, as it should be.

## 4. Estimating mu_t from what the sampler records

`mu_t` needs `q(.|s_t)` over the vocabulary. `losses.py` uses whatever is in
the batch:

* **Full sampler logprobs** (`sampling_vocab_logprobs`): exact,
  `m_t = sum_v sg(q_v w_v) log p_v`.
* **Top-k head `H`** (`sampler_topk_ids/logprobs`, enabled by
  `sampler.vocab_logprobs=k`): exact on `H`. The tail part
  `sum_{v not in H} q_v w_v grad log p_v` is estimated without bias by the
  sampled token itself, `1[y_t not in H] w_t grad log p(y_t)`, since
  `y_t ~ q`. So
  `m_t = sum_{v in H} sg(q_v w_v) log p_v + 1[y_t not in H] w_t log p(y_t)`,
  and `E_q[score - grad m_t] = 0` **exactly for any k**. A sampled tail token
  then contributes `-A sum_{v in H} q_v w_v grad log p_v`: it moves mass between
  the head and the tail, but rare tail samples never sharpen or flatten the
  tail's internal shape. Many of those tail samples are artifacts of the
  sampler (noise or quantization), so this is the safe behavior.
* **Sampled token only** (default batch, no extra sampler output): only
  `q(y_t)` is known. The loss uses the plug-in sampler `qhat`, equal to `q` at
  `y_t` and proportional to `p` elsewhere (`qhat_v = s p_v`,
  `s = (1 - q_y)/(1 - p_y)`). Since `sum_{v != y} p_v grad log p_v = -p_y grad log p_y`,
  this gives the per-token score `((1-q_y)/(1-p_y)) grad log p(y)`, i.e.
  `e_y - qhat` in logits. It is exact for a two-token vocabulary and removes
  the part of the drift visible at the sampled token. It cannot remove the
  drift spread over unobserved tokens: no estimator that only uses
  `q(y_t)` can be zero-mean without the `p/q` ratio, and with the ratio it
  becomes high-variance IS. So this is a fallback. The configuration below
  enables the top-k head, which makes the correction exact at negligible
  cost (`[B, T, k]`).

Implementation notes: all weights are detached (`sg`). Padded or masked
positions are removed with `where`, not multiplication, so garbage head
entries at unused positions cannot produce NaNs. Everything is computed
inside the existing rematerialized vocabulary chunks; the top-k part is a
`[B, chunk, k]` gather through `take_vocab`, so vocabulary sharding is
handled. The signature and the default `correction=None` are unchanged, and
centering is always on. A correction, if one is given, is centered with
`q w` as in Section 3.

## 5. Evidence (dev/mini.py, CPU, plus_minus, noise 0.3, 600 steps, mean reward of last 10 steps)

| setting | REINFORCE (old) | naive token IS | centered, token only | centered, top-8 | centered, top-64 (config) | centered, full |
|---|---|---|---|---|---|---|
| default vocab | 0.57 | 0.66 | 0.65 | 0.86 | 0.86 | 0.86 |
| +200 tail tokens | 0.00 (collapsed) | 0.15 (collapsed) | 0.12 (collapsed) | 0.82 | 0.84 (seed 2: 0.75) | 0.83 |
| +200 tail, noise 0.5 | 0.00 (collapsed) | - | - | 0.49 (peak 0.58) | - | - |

A numerical check of the expected logit update `sum_y q_y grad_z[score - m]`
gives exactly `q - p` for the old loss, and 0 for the full and top-k
estimators (also with a clamped TIS correction). Adding TIS or GSPO on top
of the centering did not help. GSPO's tight sequence band adds the token-correlated selection bias
described in Section 2, so the recommended setting is no IS correction.

## Configuration

sampler.vocab_logprobs=64

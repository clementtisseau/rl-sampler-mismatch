# Centering the score under the sampler: a fix for trainer/sampler mismatch

## Summary

REINFORCE drops the baseline term because the score has zero mean,
`E_{y~p}[grad log p(y)] = 0`. That only holds when the token was sampled
from `p`, the trainer. Here the tokens come from the sampler `q != p`, and
`E_{y~q}[grad log p(y)] = -grad KL(q || p) != 0`. Each token's update
therefore picks up a term `V(s) * (q - p)`. Here `V(s)` is the expected
advantage at the prefix. The term does not depend on which token was good,
so it is a deterministic drift that pulls the trainer toward the sampler
when `V > 0` and pushes it away when `V < 0`. The sampler is rebuilt from
the trainer with the same distortion at every step, so the drift points the
same way each step and builds up.

**Fix (in `losses.py`, on by default).** Center each sampled token's score
under the distribution the token was actually drawn from. For token `y` at
prefix `s`, the logit-space gradient changes from `A w(y) (e_y - p)` to

    A w(y) (e_y - mu),     mu = normalize(q * w(p/q))      (mu = q without correction)

In loss form, for each token:

    - A * w(y) * [ log p(y) - sum_v sg(mu_v) log p_v ]

`mu` comes from the sampler's top-k logprobs, which the batch already
carries as `sampler_topk_ids` and `sampler_topk_logprobs` (top-20, as
inference engines return them). The unobserved tail is modelled below. With
this change the expected update has no reward-independent component, so it
vanishes when the reward carries no information. This holds for any
mismatch, any reward mode and any `rl.is.*` correction.

## 1. Setup and notation

Look at one completion token at prefix `s`. Write `z` for the trainer's
logits, `p = softmax(z)`, and `q` for the sampler's next-token distribution
(noisy, int8/fp8, quantized KV or stale weights). The token `y ~ q`. `A` is
the sequence advantage and `Q(s, y) = E[A | s, y]`. The current loss sums
`-A w(y) log p(y)` over tokens, where `w` is the detached IS weight
(`w = 1` without a correction). Because

    d log p(y) / dz = e_y - p,

the ascent direction for this token, in logit space, is

    g = A w(y) (e_y - p).

The parameter gradient is `J^T g` with a Jacobian `J` that does not depend
on `y`. Everything below is linear in `g`, so it carries over to parameters.

## 2. What is wrong with the current loss

**On-policy target.** If `y ~ p`, then

    E_p[g | s] = sum_y p_y Q(s,y) (e_y - p) = p ⊙ (Q - V_p),    V_p = sum_y p_y Q(s,y).

This is the usual advantage form. Adding any baseline `b(s)` to `A` changes
nothing, because `sum_y p_y (e_y - p) = 0`.

**What the loss actually computes (`w = 1`, `y ~ q`).**

    E_q[g | s] = sum_y q_y Q(s,y) (e_y - p)
               = q ⊙ Q - p V_q                         (V_q = sum_y q_y Q(s,y))
               = q ⊙ (Q - V_q)  +  V_q (q - p).
                 \___________/     \________/
                  policy gradient   drift

The second term is the problem. The per-token score has nonzero mean under
the sampling distribution:

    E_{y~q}[ d log p(y)/dz ] = q - p = d/dz sum_v q_v log p_v = -d/dz KL(q || p).

So the drift equals `V_q(s) * grad(-KL(q || p))`. At every prefix, the loss
adds a hidden term `V_q(s) * KL(q || p)` that does not depend on which token
was good:

* When `V_q(s) > 0` (prefixes likely to succeed), the trainer is distilled
  toward the sampler.
* When `V_q(s) < 0` (prefixes likely to fail), the trainer is pushed away
  from it.

Equivalently, the baseline is no longer free. Adding a constant `b` to all
advantages adds `b (q - p)` to the expected update.

**Why group centering does not remove it.** Centering makes the advantages
sum to zero over a group of whole sequences. It does not make
`V_q(s) = E[A | s]` zero at the prefixes inside a completion. Once a
completion is on a good path, its prefixes have `V > 0`; once it is on a bad
path, `V < 0`. The loss also sums the score over tokens, so the drift is
weighted by sequence length. Longer (often failing) completions contribute
more `V < 0` tokens. With `plus_minus` rewards and a solved task (`A = +1`
everywhere), the whole update is drift: `q ⊙ 0 + 1 * (q - p)`, i.e. pure
distillation into the sampler. The optimum of the RL objective is not a
stationary point of the update.

**Why it compounds instead of averaging out.** The sampler is a fixed
transformation of the trainer, `q = T(p)`: the same weight-noise vector,
the same quantizer, or the same lag. The drift direction is
`±(T(p) - p)`. That is not zero-mean noise; it points the same way at every
step. A step toward `T(p)` makes the new trainer `p' ≈ T(p)`, and the next
sampler is `T(T(p))`. After `t` steps the parameters have moved about
`t * eta * |V| * |T(p) - p|` in a consistent direction, linearly in `t`
rather than as `sqrt(t)`. For weight noise and quantization, `T` puts extra
mass on the low-probability tail: by Jensen, `E[exp(z_v + eps_v)] > exp(z_v)`,
so `q_v / p_v > 1` on average for tail tokens. The `V > 0` drift therefore
keeps moving tail mass into `p`. Entropy rises, the sampler emits even more
tail tokens, and the reward collapses. On `V < 0` prefixes the opposite
drift drives the logits of tokens the trainer considers unlikely
(`p(y) << q(y)`) toward `-inf`, with an `O(|A|)` gradient that never
saturates. That damages the shared parameters.

**Why IS and clipping only slow it down.** Let `m_y = q_y w(p_y / q_y)`
(this is `exp(log_mass(...))` in `losses.py`), `M = sum_y m_y`, and
`mu = m / M`. Then

    E_q[g | s] = sum_y m_y Q (e_y - p) = m ⊙ (Q - V_m) + M V_m (mu - p).

* Exact token IS (`w = p/q`) gives `m = p`, so `mu = p` and the drift is
  zero (apart from the prefix distribution, Section 5). But it has high
  variance.
* Truncating, clamping or dropping the weight makes `mu` a mixture of `p`
  and `q`. The drift becomes `M V_m (mu - p)`: smaller, but the same in
  kind.
* The tighter the band, the more tokens sit at the bound, the closer `mu`
  gets to `q`, and the closer `M` gets to 1. This is why stronger
  token-level clipping can make things worse.
* Sequence-level corrections rescale `A` by one scalar per sequence, which
  leaves the per-token drift `V (q - p)` unchanged.

## 3. The modification

Replace the trainer's centering `p` in the score with the distribution that
generated the token, as reweighted by the correction:

    g_new = A w(y) (e_y - mu),     mu = (q ⊙ w) / sum_v q_v w_v.

**Unbiased with respect to any baseline.** For any prefix-dependent `b(s)`,

    E_q[ b w(y) (e_y - mu) ] = b (m - M mu) = 0.

**Expected update.**

    E_q[g_new | s] = m ⊙ (Q - V_m).

There is no `(q - p)` term left. The update is driven only by differences
in advantage between tokens. If all continuations of `s` are equally good,
the expected update is exactly zero. In particular, it is zero when the
task is solved or the group has no signal. So the RL fixed points are fixed
points of training again, and nothing systematic accumulates.

**Still an ascent direction.** In the plain case `mu = q`, the update
`q ⊙ (Q - V_q)` is the exact policy gradient that the sampler's own logits
would receive. The trainer moves its logits the way the sampler's logits
should move. Its inner product with the true gradient `p ⊙ (Q - V_p)` is

    sum_y q_y p_y (Q_y - V_q)(Q_y - V_p),

which equals `sum p_y^2 (Q_y - V_p)^2 >= 0` at `q = p` and stays positive
for nearby `q`. Its error relative to the on-policy gradient is
`O(||q - p|| * |Q - V|)`. That error is gated by the reward signal, so it
cannot act when the signal is gone, unlike the `V (q - p)` drift.

**Consistency.** The fix collapses to known estimators in the limits:

* `q = p` gives plain REINFORCE.
* Exact token IS gives `mu = p`, which is also plain token IS.
* For any other correction (TIS, MIS, PPO-type bands, `sign=neg`/`pos`),
  the score is re-centered under that correction's effective sampling
  measure `m = q w`, which removes the drift that clipping reintroduces.

## 4. Implementation (`losses.py`)

**Loss form.** The gradient of `sum_v sg(mu_v) log p_v` with respect to `z`
is `mu - p`, because `sum_v mu_v = 1`. So the per-token loss

    -A w(y) [ log p(y) - sum_v sg(mu_v) log p_v ]

has gradient `-A w(y) (e_y - mu)` in logit space.

**Computing `mu` from the top-k.** The batch has the sampler's top-k
next-token logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`, aligned
with `sampling_token_logprobs`), not the full vocabulary. Write `K` for the
top-k set, `Q_K = sum_K q_v`, `P_K = sum_K p_v`, `Q_tail = 1 - Q_K`, and
`P_tail = 1 - P_K`.

* On `K`, `mu` is exact: `m_v = q_v w(p_v / q_v)`.
* The tail is modelled as holding the sampler's remaining mass, spread like
  the trainer's: `q_v = (Q_tail / P_tail) p_v` for `v` not in `K`. This gets
  the total tail mass right, which is where noise and quantization move
  probability, and leaves only the shape inside the tail as an assumption.
* Under this model the ratio `p/q = P_tail / Q_tail` is constant on the
  tail, so its weighted mass is `m_tail = Q_tail w(P_tail / Q_tail)`. The
  code uses the same `log_mass` function for it, so every `rl.is.*`
  correction is handled.
* Normalizing gives `mu_K = m_K / M`, `mu_tail = m_tail / M`, and
  `mu_v = c p_v` on the tail with `c = mu_tail / P_tail`.

**No vocabulary scatter needed.** Since `sum_v sg(p_v) grad log p_v = 0`,
the tail's contribution can be moved onto the top-k ids:

    grad sum_v mu_v log p_v = grad sum_{v in K} sg(mu_v - c p_v) log p_v.

The code checks this term by term against `mu - p`. Only a gather of
`log p` at the `k` sampler ids is needed, done with `take_vocab`, so it
works with a TP-sharded vocabulary and in the existing rematerialized
chunks.

**Safeguards.**

* `P_tail` is floored at `1e-6` (float32 precision of `1 - P_K`).
* `-inf` sampler logprobs get mass 0.
* Masked or stale positions are zeroed.
* If the batch has no top-k keys, the loss falls back to `mu = p` (plain
  REINFORCE).

**Other behavior.** Sequence-level corrections still rescale `A` per
sequence, and their token-level `mu` is `q` (`w = 1` per token). The
signature, metrics and default arguments are unchanged. The re-centering is
on whenever `loss_fn_rl` is called, including with `correction=None`.

## 5. What it does not fix

Prefixes are still visited with the sampler's distribution. Correcting that
needs sequence-level ratios `prod_t p/q`, whose variance grows with length.
This changes how much weight each prefix gets. It does not add a direction
that is independent of the reward, so it does not compound the way the
drift does. Unclipped token-level IS may be added on top (`rl.is.level=token`),
but it is not required. The default (`correction=None`, `mu = q`) needs no
configuration.

## Sanity check (dev/mini.py, CPU; not required)

Mean training reward over the last 10 steps, original loss -> this loss (600 steps):

| setting | original | this loss |
| --- | --- | --- |
| noise 0.3 (default), seed 0 | 0.58 | 0.87 |
| noise 0.3, seed 1 | 0.61 | 0.82 |
| noise 0.5 | 0.00 (collapsed) | 0.57 |
| noise 0.3, 200 extra tail tokens (vocab > top-20) | 0.00 (collapsed) | 0.83 |
| noise 0.5, 200 extra tail tokens | 0.00 (collapsed) | 0.41 |
| noise 0.5 + TIS (clamp at 2) | 0.00 (collapsed) | 0.39 (eval 0.79) |
| no noise | 0.93 | 0.92 (unchanged) |

## Configuration

No configuration changes are required. The modification is active with
the default `correction=None`.

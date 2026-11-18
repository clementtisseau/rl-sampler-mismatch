# Removing the sampler-mismatch drift from REINFORCE

## Summary

The REINFORCE loss `-A log p(y|h)` is only centred when the tokens are drawn
from `p`. When they are drawn from a different sampler `q`, its expected update
at every prefix picks up a term `E_q[A|h] * sum_v q(v|h) grad log p(v|h)`. This
term does not depend on which token was rewarded. It is a deterministic drift
that pulls `p` toward `q` wherever the expected advantage is positive and pushes
it away wherever it is negative. Because the sampler is rebuilt from the trainer
at every step (`q = p + mismatch`), the drift never settles and its effect
accumulates.

The fix is to subtract the sampler's expected score as a per-token baseline:

    loss = - sum_tokens A * ( w_y log p(y|h)  -  sum_v sg(q(v|h) w_v) log p(v|h) ) / n_tokens

Here `sg` is stop-gradient and `w` is the IS weight of the active correction
(`w = 1` with `correction=None`). In logit space the per-token gradient becomes
`A (e_y - q)` in place of `A (e_y - p)`. The expected update at every prefix is
then exactly the covariance between the advantage and the score, which is the
on-policy form. Whatever the sampler is, the update is zero in expectation
unless the reward actually depends on the token. On-policy (`q = p`, no
correction) the extra term has gradient `sum_v p_v grad log p_v = 0`, so the
loss reduces to plain REINFORCE.

## 1. What is wrong with the current loss

Notation: `h` is a prefix, `y ~ q(.|h)` is the next token, `A` is the advantage
of the whole completion (it depends on `y` and on the continuation), and
`s_v = grad_theta log p(v|h)`. Write `z` for the logits at `h`. Then
`d log p(v|h)/dz = e_v - p`, where `p` is the trainer's next-token vector.

The current loss (no correction) updates along `A s_y`. Conditionally on `h`:

    E_q[A s_y | h] = sum_v q(v) E[A | h, v] s_v
                   = E_q[A|h] * sbar(h)  +  Cov_q(A, s_y | h),
    sbar(h) = sum_v q(v|h) s_v .

- **On-policy**, `sbar = sum_v p_v grad log p_v = grad sum_v p_v = grad 1 = 0`.
  Only the covariance remains. It is non-zero only if the advantage depends on
  the token chosen at `h`, so it is the reward signal.
- **Under mismatch**, `sbar != 0`. In logit space, `sbar(h) = sum_v q_v (e_v - p) = q - p`.
  The update at `h` therefore contains

      drift(h) = E_q[A|h] * (q(.|h) - p(.|h))     (in logit space)

  This is a step of size `E_q[A|h]` along the direction that increases
  `sum_v q_v log p_v`, i.e. that decreases `KL(q || p)`. When `E_q[A|h] > 0`,
  `p` is pulled toward `q`. When `E_q[A|h] < 0`, it is pushed away from `q`.

Why the drift is a bias and not noise:

1. **It does not average out over the batch.** `E_q[A|h]` is not zero at most
   prefixes, even with group-centred rewards. Centring gives
   `E[A | prompt] ≈ 0` only at the root. At a deeper prefix, `E[A|h]` is the
   value of `h` minus the group mean. That is positive on prefixes that are
   on track and negative on prefixes that already contain a mistake. With
   `plus_minus` rewards, `E[A|h]` is not zero even at the root. The mismatch
   `q - p` comes from fixed weight noise, quantisation or staleness, so it is
   a fixed function of `h` and is correlated across steps.
2. **It compounds.** After the update, the sampler is re-derived from the new
   trainer: `q' = p' + mismatch`. Pulling `p` toward `q` (on good prefixes)
   does not close the gap, because the sampler moves along with `p` and the
   gap persists. Pushing `p` away from `q` on bad prefixes widens the gap, and
   `KL(q||p)` grows step after step. A larger gap makes the drift larger
   (`||q - p||` grows), which is a positive feedback loop. It is driven by a
   term with no reward content: the update moves `p` along `±(q - p)` at every
   step.
3. **Pushing away from `q` is the destructive half.** On a prefix with
   `E[A|h] < 0`, the update raises the logits of tokens where `p > q` and
   lowers those where `q > p`. Nothing bounds this. It sharpens or flattens
   `p` in directions chosen by the noise pattern of the sampler. Probability
   mass ends up on tokens that the reward never favoured (for an LLM, the long
   tail of the vocabulary), entropy and `kl_samp_train` grow, and the reward
   eventually collapses.

### Why IS weights and clipping do not remove it

With a per-token weight `w_y = w(p_y/q_y)`, the expected update is

    E_q[A w_y s_y | h] = E_q[A|h] * sum_v q_v w_v s_v + Cov_q(A, w_y s_y | h).

For exact token IS (`w = p/q`), `sum_v q_v (p_v/q_v) s_v = sum_v p_v s_v = 0`,
so the drift is gone in expectation. But:

- the variance is unbounded as `p/q -> inf`, and sequence-level weights
  (products over the sequence) are worse;
- every practical method bounds `w` (TIS, MIS, PPO, DAPO, DPPO ...). Then
  `sum_v q_v w_v s_v = sum_v m_v (e_v - p)` with `m_v = q_v w_v`, which is not
  zero unless `m = p`. The drift is still there, now pointing toward the
  clipped measure `m/|m|` instead of `q`. A tighter band leaves `m` closer to
  `q` (or drops more mass) on the tokens that are most mismatched, so the
  residual `m - p` is larger. That is why stronger token-level clipping can
  make collapse worse: clipping cuts exactly the part of the correction that
  cancelled the drift. Dropping tokens (`outside=drop`) also leaves
  `sum_v m_v < 1`. The remainder behaves like a shrink of `p` along `-p`,
  which is a further non-zero bias.

So IS fixes the bias only in its unbounded, high-variance form, and every
stabilised version of it brings part of the bias back.

## 2. The modification

Subtract the expected score under the sampler, with the same IS weighting, at
every token:

    g = A * ( w_y s_y  -  b(h) ),        b(h) = sum_v q(v|h) w_v s_v = E_{y'~q}[w_{y'} s_{y'} | h].

`b(h)` depends on `h` but not on `y`, and it is detached through `sg(q w)`. The
surrogate whose gradient is `g` is

    A * ( w_y log p(y|h) - sum_v sg(q_v w_v) log p(v|h) ).

**Expected update.** Since `b(h)` does not depend on `y`:

    E_q[g | h] = E_q[A w_y s_y | h] - E_q[A|h] b(h)
               = sum_v q_v E[A|h,v] w_v s_v - E_q[A|h] sum_v q_v w_v s_v
               = sum_v q_v (E[A|h,v] - E_q[A|h]) w_v s_v
               = Cov_q(A, w_y s_y | h).

This is exactly the on-policy structure. The term proportional to `E_q[A|h]`
is gone identically, for any `q`, any clipping band and any sign rule. If the
advantage does not depend on the choice at `h`, the expected update at `h` is
zero, so the mismatch has nothing to amplify. The covariance term is the usual
signal. Its magnitude is unchanged, and the score direction is the trainer's
own `s_v`.

In logit space with `correction=None` the per-token gradient is

    current:   A (e_y - p)     -> E_q = E_q[A|h](q - p) + Cov_q(A, e_y)
    modified:  A (e_y - q)     -> E_q = Cov_q(A, e_y)

So the change replaces the trainer's softmax in the REINFORCE gradient by the
sampler's distribution, the distribution the token was actually drawn from.
`e_y - q` has mean zero under `y ~ q` and `e_y - p` does not.

**Properties.**

- **On-policy it is plain REINFORCE.** With `q = p`, `w = 1`, the extra term
  has gradient `sum_v p_v grad log p_v = 0`. The loss value shifts by a
  constant (the entropy times `A`), but the gradient is identical.
- **No new variance source.** `b(h)` is a control variate:
  `E_q[w_y s_y - b(h) | h] = 0`. In logit space, `c = q` minimises
  `E_{y~q} |e_y - c|^2` over constant vectors `c`, with value `1 - |q|^2`.
  So the token factor `e_y - q` is the best-centred version of `e_y`, while
  `e_y - p` carries the extra `|q - p|^2`. Nothing is divided by `q`, so unlike
  the IS fix, no weight can explode.
- **It composes with any `rl.is.*` correction.** The baseline uses the mass
  `m_v = q_v w_v` from the same `log_mass` that defines the correction, so the
  drift the correction leaves behind (Section 1) is cancelled too. With
  sequence-level weights the sequence weight multiplies `A`, and the per-token
  part is still centred.
- **What remains.** The prefixes are still visited under `q` and continued
  under `q`, so the signal is the covariance under the sampler's state
  distribution. That reweights the per-prefix improvement directions but does
  not add a direction without reward content. The mismatch now only affects
  how much each true improvement direction is weighted, and cannot feed back
  on itself.

## 3. Implementation (`losses.py`)

`sampler_mass` computes the detached mass `m_v = q_v w_v` and the surrogate
`sum_v m_v log p_v`. `loss_fn_rl` uses `w_y log p_y - sum_v m_v log p_v` as the
per-token score. The modification is always on: no flag, and it works with
`correction=None`. `sampler_keys` now forwards the sampler's distribution to
the loss.

What `q` is depends on what the sampler recorded:

- **`sampler.vocab_logprobs=full`** (`sampling_vocab_logprobs [B,T,V]`): `q`
  is exact, so the drift is removed exactly. A unit test that enumerates `y`
  at one prefix with constant `A` gives expected-gradient norm ~1e-8, against
  2e-1 for the current loss.
- **`sampler.vocab_logprobs=k`** (top-k head): `q` is known exactly on
  `K = head ∪ {sampled y}`. The leftover mass `q(T) = 1 - q(K)` is spread over
  the tail in proportion to `p`: `q_hat(v) = q(T) p(v)/p(T)` for `v ∉ K`.
  Then `q_hat - p` agrees with `q - p` on `K`, and on the tail it is only off
  in shape. The residual drift is bounded by
  `|E[A|h]| * TV_tail <= |E[A|h]| * q(T)`, and for an LLM with `k` in the
  tens, `q(T)` is tiny at almost every position. On `T` the ratio `q_hat/p` is
  constant, so the IS band is evaluated once on `(p(T), q(T))`. The tail term
  is computed without materialising `q_hat`:
  `sg(m(T)/p(T)) * (sum_V sg(p) log p - sum_K sg(p) log p)`. `p(T)` is floored
  at 1e-4 against float32 cancellation, which can only under-correct.
- **Nothing recorded** (default sampler): `K = {y}` alone, from
  `sampling_token_logprobs`. This still corrects the sampled-token component
  of the drift (the largest single one, since `y` is drawn where `q` is
  large), but it is only partial. Use the configuration below.

Recommended: `sampler.vocab_logprobs=64`. With `k = 64`, the sampler's head covers essentially all of `q`'s mass at
nearly every position, and memory stays at `[B, T, 64]`. `full` is exact and
also supported, but it costs `[B, T, V]` float32, about 150k floats per token
for Qwen3.

Memory: everything is computed inside the existing rematerialised chunks. The
full-vocabulary case adds one `[B, chunk, V]` product. The top-k case adds a
`[B, chunk, k+1]` gather.

## Sanity checks (CPU `dev/mini.py`, noise 0.3, 600 steps, seed 0)

| setting | loss | train reward, last 10 | eval reward, end |
| --- | --- | --- | --- |
| 12 tokens | original | 0.58 | 0.66 |
| 12 tokens | new, nothing recorded (K = {y}) | 0.64 | 0.76 |
| 12 tokens | new, `vocab_logprobs=3` | 0.87 | 0.85 |
| 12 tokens | new, `vocab_logprobs=full` | 0.85 | 0.83 |
| +200 tail tokens | original | 0.00 (collapsed by step 25) | 0.00 |
| +200 tail tokens | new, `vocab_logprobs=8` | 0.83 | 0.79 |

The collapse with tail tokens shows the mechanism of Section 1: the drift
pushes mass onto tokens the reward never selected.

## Configuration

sampler.vocab_logprobs=64

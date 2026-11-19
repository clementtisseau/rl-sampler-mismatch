# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## TL;DR

The REINFORCE loss `-A * log p(y_t)` relies on one identity: the score
function has zero mean under the distribution the tokens come from,
`E_{y~p}[grad log p(y)] = 0`. That identity is why baselines, reward offsets
and tokens that do not matter for the reward cost nothing on-policy. When the
tokens come from a sampler `q != p`, the identity fails:
`E_{y~q}[grad log p(y)] = -grad KL(q || p) != 0`. Every token then gets an
extra update `E[A | prefix] * (q - p)` (in logit space). This update does not
depend on whether the token affects the reward, it has a fixed sign and
direction rather than averaging out, and it is applied again at every step.
That is the drift that makes training collapse.

The fix is to center the score under the **sampler's** next-token
distribution instead of the trainer's:

    loss_t = -A * ( log p(y_t) - sum_v sg(q_t(v)) * log p_t(v) )

Its logit gradient is `A * (e_{y_t} - q_t)`, which has zero mean under `q`
for any `q`. The drift term is removed exactly and the reward signal is kept
exactly. On-policy (`q = p`) the extra term has zero gradient
(`sum_v p grad log p = grad sum_v p = 0`), so the method reduces exactly to
standard REINFORCE. `q_t` is rebuilt from the sampler's top-k logprobs, which
the sampler already returns (`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20).
The leftover mass is spread over the rest of the vocabulary in proportion to
`p`. The change is always on, needs no flag, works with `correction=None`,
and also composes with any `rl.is.*` correction (see section 5).

## 1. Setup and notation

A rollout is `y_1..y_L` with prefixes `s_t = (prompt, y_<t)`. The trainer's
next-token distribution is `p_t = softmax(z_t)` and the sampler's is `q_t`.
`A` is the sequence advantage, a function of the whole rollout. The current
loss is

    L = -(1/N) sum_i A_i sum_t log p(y_{i,t} | s_{i,t})

The gradient of `log p(y|s)` with respect to the logits `z` at that position
is `e_y - p` (`e_y` is the one-hot vector of `y`). One SGD step therefore
moves the logits at `(i, t)` by `eta/N * A_i (e_{y_t} - p_t)`. The parameter
update is the sum of these vectors pulled back through `d z / d theta`, so it
is enough to study the expected logit update at each prefix.

## 2. Expected update: on-policy vs. mismatched

Fix a prefix `s_t` reached by the sampler. The token `y_t ~ q_t` is drawn, and
the continuation is drawn from `q`. Let `Q(s_t, y) = E_q[A | s_t, y_t = y]`
and `V(s_t) = E_q[A | s_t] = sum_y q_t(y) Q(s_t, y)`. Then

    E[ A (e_{y_t} - p_t) | s_t ]
      = sum_y q_t(y) Q(s_t,y) (e_y - p_t)
      = sum_y q_t(y) (Q(s_t,y) - V(s_t)) e_y      +   V(s_t) (q_t - p_t)
        \______________ signal _______________/      \______ drift ______/

using `sum_y q_t(y)(Q - V) = 0` for the first term.

* **Signal.** `Cov_q(A, e_{y_t} | s_t)`. It raises the logits of tokens whose
  continuation does better than average and lowers the others. It is zero
  when the token does not affect the reward.
* **Drift.** `V(s_t)(q_t - p_t)`. On-policy, `q = p` and it is exactly zero,
  which is why REINFORCE needs no correction there. In parameter space it is
  `V(s_t) * E_q[grad log p] = -V(s_t) * grad_theta KL(q_t || p_theta)`.

Properties of the drift:

1. **It does not depend on the reward signal.** It acts on every token,
   including the large majority of tokens whose choice does not affect `A`.
   Its size depends only on `V(s_t)` and on the mismatch `q_t - p_t`.
2. **It is systematic, not noise.** `q_t - p_t` is a deterministic function of
   the weights: fixed noise `delta`, a fixed quantizer, or a fixed lag. Its
   sign is set by `V(s_t)`. So it does not average out across samples or
   steps.
3. **Group-centering does not remove it.** Group-centering makes `E[A] = 0`
   over a group. It does not make `E[A | s_t] = 0` for each prefix. Prefixes
   that already contain a mistake have `V(s_t) < 0` and prefixes on a good path
   have `V(s_t) > 0`. With `plus_minus` / `zero_minus` rewards `V` is mostly
   negative everywhere. Uneven sequence lengths (`sum_t` with a global token
   normalizer) add a length-weighted component as well.
4. **It is asymmetric and unbounded.** When `V > 0`, `p` moves toward `q`,
   which is distillation into the sampler's errors: harmful but bounded,
   since `KL >= 0`. When `V < 0`, the update is gradient ascent on
   `KL(q_t || p)`, which is unbounded. `p` pushes down the tokens the sampler
   over-samples and sends mass to tokens the sampler under-samples, such as
   rare or garbage tokens in the tail of a large vocabulary. Once the policy
   gets worse, `V` becomes more negative, so the push gets stronger.
5. **It compounds.** The sampler is re-derived from the trainer at every step,
   `q = f(theta)`. The drift therefore moves `theta` by
   `-eta V grad KL(f(theta) || p_theta)` at every step, in a direction that
   the reward does not oppose. Nothing pulls the policy back except the
   signal term, which is first order in the advantage spread, while the drift
   is first order in `V`. The bias accumulates over training. This matches
   the observation that the mismatch acts as a compounding systematic bias.

## 3. Why IS weights and clipping only slow it down

A token-level weight `w(r)` with `r = p/q` changes the expected update to

    sum_y q_t(y) w(y) Q(y) (e_y - p_t)  =  signal_w  +  V_w(s_t) (q^w_t - p_t * E_q[w])

where `q^w = q * w`. The drift vanishes only if `q * w` is proportional to
`p`, which means `w` must equal the exact ratio `p/q` up to a constant per
prefix:

* **Naive IS** (`w = r`) is unbiased, but its variance is
  `E_q[r^2] = 1 + chi^2(p || q)` per token, and it multiplies along the
  sequence. Long completions and the many small-`q` tokens of a large
  vocabulary make it unusable.
* **Truncated / clamped / dropped weights** (TIS, MIS, PPO, DAPO, DPPO) leave
  the drift `V (q * w - p * E_q[w])` wherever the band is active. Tighter
  clipping makes `w` closer to a constant, and with a constant `w` the drift
  is exactly the uncorrected `V (q - p)`. This is why stronger token-level
  clipping can make training worse.
* **Sequence-level weights** (GSPO, TOPR, seq-level IS) rescale `A` for each
  sequence but leave every token's score centered at `p`. The drift is
  rescaled, not removed.

All of these try to fix the *distribution of samples*. The real defect is in
the *score*: it is centered at the wrong distribution.

## 4. The fix: center the score under q

Replace the per-token score by

    phi_t = log p(y_t | s_t) - sum_v sg(q_t(v)) log p_t(v)

Its logit gradient is

    grad_z phi_t = (e_{y_t} - p_t) - (q_t - p_t * sum_v q_t(v)) = e_{y_t} - q_t

The expected update becomes

    E[ A (e_{y_t} - q_t) | s_t ] = sum_y q_t(y) (Q(s_t,y) - V(s_t)) e_y = Cov_q(A, e_{y_t} | s_t)

* **The drift is gone exactly, at every prefix, for any `q`.** The signal term
  is unchanged.
* **Shift invariance is restored.** Replacing `A` by `A + c` (any baseline or
  reward mode, even one that depends on the prefix) changes the expected
  update by `c * E_q[e_y - q] = 0`. Baselines are free again, as on-policy.
* **No reward information means no update.** If the token does not affect the
  reward, `Q(s_t, .)` is constant and the expected update is `0`. Mismatch
  alone can no longer move the policy, so it can no longer compound. The
  sampler only changes how strongly each token is weighted (`q` instead of
  `p`) and which prefixes and continuations are evaluated. Those biases are
  multiplied by the advantage spread, vanish where there is no signal, and
  vanish as `q -> p`. For `q` close to `p` the update is
  `sum_y p (Q - V) e_y + O(|q - p| * |Q - V|)`, an ascent direction for the
  true objective.
* **It is exact on-policy.** With `q = p`,
  `grad sum_v sg(p(v)) log p(v) = sum_v grad p(v) = 0` holds sample by sample,
  not only in expectation. The modified loss has the same gradient as the
  original.
* **It does not raise variance.** `e_y - q` and `e_y - p` have the same scale
  and no ratios appear. In parameter space,
  `grad phi_t = grad log p(y_t) + grad KL(q_t || p)`: the subtracted term is
  the sampler-mean of the score, `E_q[grad log p] = -grad KL(q_t || p)`.

Equivalently, the extra term is a per-token baseline in score space:
`A * E_{q_t}[grad log p]`. It is zero for an on-policy sampler and exactly
cancels the off-policy bias otherwise.

### Reconstructing q_t

Only the sampler's top-k logprobs (`H_t`, k = 20) and the sampled token's
logprob are observed. The reconstruction is

    q~_t(v) = q_t(v)                         for v in H_t, plus the sampled token
    q~_t(v) = p_t(v) * q_rest / p_rest        otherwise

with `q_rest = 1 - sum_{H} q` and `p_rest = 1 - sum_{H} p`, both detached.
`q~` sums to 1, so the logit gradient is exactly `e_y - q~`. The leftover
drift is `V (q - q~)`. It lives only in the tail and only measures how far the
tail of `q` is from proportional to `p`, so it is at most `2 V q_rest` and
usually much smaller. The tail sum is computed as
`sum_all sg(p) log p - sum_H sg(p) log p` from the full-vocabulary
log-softmax that the loss already computes in each chunk. If `p_rest < 1e-6`
the tail is ignored. If a batch has no top-k logprobs, the head is only the
sampled token. This fallback is still exact on-policy and removes part of the
drift.

## 5. Composition with IS corrections

With a token-level weight `w(y)` (any `rl.is.*` band, clamp or drop, any
sign rule), the correct centering distribution is `c_t(v) = q_t(v) w(v)`:

    phi_t = w(y_t) log p(y_t) - sum_v sg(c_t(v)) log p_t(v)
    grad_z phi_t = w(y_t)(e_{y_t} - p_t) - (c_t - p_t sum_v c_t(v))
    E_q[grad_z phi_t] = c_t - p_t E_q[w] - c_t + p_t E_q[w] = 0

`losses.py` computes `c_t` on the head with the same `log_mass` function as
the token weight, so it uses the same band, clamp/drop and sign rules. On the
tail it uses `q~ * w(1/s)`. Sequence-level weights still multiply `A`.

## 6. Implementation (`losses.py`)

* `sampler_centering(...)` returns `m_t = sum_v sg(c_t(v)) log p_t(v)` for each
  position of a chunk. It uses `sampler_topk_ids` / `sampler_topk_logprobs`
  when present and the sampled token otherwise.
* In `loss_fn_rl` the per-token term is now `w * logp - center` instead of
  `w * logp`. Masking, normalization, sequence-level corrections, metrics and
  the signature are unchanged. The change is always active, including with
  `correction=None`.

Checks:

* With `q = p` the expected gradient under `q` is 0, to float precision.
* With `q != p` and full-vocabulary top-k the expected gradient is 0.
* With a perturbed `q` and top-5 of 50 tokens, the max expected gradient
  entry for a constant advantage drops from 0.86 to 0.008.

On the CPU dev task (noise 0.3, `plus_minus` rewards, 600 steps):

| Setting | Original loss | Centered loss |
| --- | --- | --- |
| Visible setting | last-10 reward 0.58, eval drifting down from 0.72 to 0.66 | 0.85, eval rising to 0.83 |
| `MINI_EXTRA_TOKENS=40` | collapses to 0 | 0.84 |
| `MINI_EXTRA_TOKENS=200` | collapses to 0 | 0.83 |
| Noise 0.5, `MINI_EXTRA_TOKENS=40` | collapses to 0 within 25 steps | stable, eval about 0.65 |

No configuration change is needed. The defaults (`rl.is.level=null`) are
intended.

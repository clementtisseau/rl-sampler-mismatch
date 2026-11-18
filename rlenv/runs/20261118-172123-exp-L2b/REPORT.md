# Sampler-centered REINFORCE: removing the mismatch drift

## 1. Setup

Fix a prompt and a prefix `x` inside a completion. The trainer has logits `z` at that position and
next-token distribution `p = softmax(z)`. The sampler (noisy, quantized or stale) has next-token
distribution `q`. The rollout draws the next token `y ~ q(.|x)` and continues from there, also with `q`.
The completion gets a sequence-level advantage `A`. At this position, define the conditional
advantage of each candidate token,

    a(y) = E[A | x, y]          (the expectation is over the rest of the rollout and the group)

and its means under the two distributions, `abar_q = sum_y q(y) a(y)` and `abar_p = sum_y p(y) a(y)`.
`e_y` is the one-hot vector of `y`.

The current loss is `-(1/N) sum_i A_i sum_t w_t log p(y_t)` with `w = 1` when no correction is used.
At one position, its gradient with respect to the logits is the score

    grad_z log p(y) = e_y - p .

## 2. Expected update at one position

**On-policy (`q = p`).**

    E_{y~p}[a(y)(e_y - p)] = sum_y p(y) (a(y) - abar_p) e_y = grad_z E_{y~p}[a(y)],

because `E_p[e_y - p] = p - p = 0`. The update is the exact gradient of the expected advantage at that
prefix. A component of the advantage that does not depend on `y` (a baseline, the value of the prefix,
or the `+1/-1` offset of `plus_minus` rewards) contributes nothing in expectation. This baseline
invariance is the reason REINFORCE works.

**Mismatched (`q != p`).** The score no longer has zero mean, `E_{y~q}[e_y - p] = q - p`, and

    E_{y~q}[a(y)(e_y - p)] = sum_y q(y)(a(y) - abar_q) e_y   +   abar_q (q - p)
                             \________ signal ________/       \___ drift ___/

Since `grad_z KL(q || p) = p - q`, the second term equals `-abar_q * grad_z KL(q||p)`. This drift has
four properties:

* **It is a bias, not noise.** It does not depend on which token was sampled. It depends only on the
  value of the prefix `abar_q` and on the mismatch `q - p`. Averaging over more samples does not
  remove it.
* **It is present at every position whose prefix value is not zero.** Group centering makes
  `abar = 0` only at the first completion token, where all members of a group share the prompt.
  Deeper in the completion, prefixes differ and `abar_q(x) != 0`. Failing branches have
  `abar_q < 0`, and on them the loss does gradient **ascent** on `KL(q||p)`: it pushes `p` away from
  the sampler, toward the tokens that `q` under-weights (often junk tail tokens). Successful branches
  have `abar_q > 0`, and on them the loss distills `p` toward the corrupted `q`. The signal term is
  proportional to the spread of `a` across next tokens, which is near zero at most positions, because
  most tokens do not decide the outcome. The drift is proportional to `|abar_q|`, which is O(1) at
  every position. Summed over a sequence, the drift can therefore dominate the signal.
* **It never stops.** At an optimum where `a` is constant over the support, the signal is zero but
  the drift `abar_q (q - p)` is not. With `plus_minus` rewards, once everything is solved, every step
  distills `p` into its own corrupted copy.
* **It compounds.** The sampler is rebuilt from the trainer at every step, `q = f(p)`, and `f` is
  systematic (a fixed weight perturbation, deterministic rounding, a lag). The drift therefore pushes
  in the same direction at every step. With a fixed weight perturbation `delta` and `q - p ~ J delta`,
  the drift becomes a parameter step `Delta theta ~ eta * abar * J^T J delta = eta * abar * F delta`,
  which moves along the same Fisher-projected `+-delta` at every step. With relative noise
  (`delta ∝ theta`) the perturbation grows with the weights, and the motion feeds back into itself.

This matches what is already known: the mismatch acts as a systematic bias in the objective that
compounds over training.

## 3. Why IS weights and clipping only slow it down

With a detached token weight `w(y) = w(p(y)/q(y))`, write `m = q*w`, `Z = sum m` and `c_w = m/Z`:

    E_{y~q}[w a (e_y - p)] = Z * [ sum_y c_w(y)(a(y) - abar_{c_w}) e_y  +  abar_{c_w} (c_w - p) ] .

The drift vanishes only if `c_w = p`, that is, if `w = p/q` exactly (untruncated token IS). Any
truncation, clamp, drop band or TV band makes `c_w` a mixture of `q` and `p` on the clipped tokens.
The drift then becomes `abar (c_w - p)`: smaller, but still not zero. That is why these corrections
slow the degradation down without removing it. A tighter token band leaves more tokens at `w = const`.
That pulls `c_w` back toward `q`, which is why stronger token-level clipping can make things worse.
PPO-style sign-dependent bands apply a different `c_w` to `abar > 0` and `abar < 0` branches, so the
drift on the clipped side is unchanged. Sequence-level weights only rescale `A`, which changes
`abar` but leaves the factor `(q - p)` untouched. Exact IS removes the drift, but its weights `p/q`
are heavy-tailed. They are hard to use in practice, and once they are clipped (even at e^±20) the bias
returns.

## 4. The fix: center the score on the sampling distribution

The drift exists because the score is centered at the wrong distribution: `e_y - p` is centered at
`p`, but `y` is drawn from `q`. The fix uses the score centered at the distribution that actually
generated the token:

    s(y) = e_y - c,      c = q            (no correction)
                         c = q w / sum(q w) (with a token-level correction w)

`s(y)` is the logit gradient of `log p(y) - sum_v c_v log p_v`, with `c` detached, because
`grad_z sum_v c_v log p_v = c - p` when `sum c = 1`. The loss becomes

    L = -(1/N) sum_i A_i sum_t w_t [ log p(y_t | x_t) - sum_v sg(c_v(x_t)) log p(v | x_t) ] .

**Expected update.**

    E_{y~q}[w a (e_y - c)] = sum_y q w a e_y - (sum_y q w a) c = Z sum_y c(y)(a(y) - abar_c) e_y .

* **No drift for any `q`.** `E_{y~q}[w(e_y - c)] = Z(c - c) = 0`, so a constant (`y`-independent)
  advantage produces exactly zero expected update, as in the on-policy case. Baseline invariance is
  restored. The prefix value `abar` no longer matters, and neither does the reward offset.
* **It reduces exactly to REINFORCE on-policy.** If `q = p` and `w = 1`, then `c = p` and the extra
  term has gradient `c - p = 0`. The loss and its gradients are those of the original loss.
* **The update is still an improvement direction.** Without a correction, the update is
  `Delta z = eta sum_y q(y)(a(y) - abar_q) e_y`, which is the exact on-policy gradient of
  `E_{softmax(z')}[a]` taken at `z' = log q` and applied to the trainer's logits. Its inner product
  with the trainer's own gradient `grad_z E_p[a] = p * (a - abar_p)` is
  `eta sum_v p_v q_v (a_v - abar_p)(a_v - abar_q)`. This equals `eta sum p^2 (a - abar)^2 >= 0` at
  `q = p` and stays positive for `q` near `p`. The mismatch now only perturbs the direction of the
  useful signal, a second-order and self-limiting effect. It no longer adds a first-order term that
  is aligned across steps.
* **There is a stationary point at the optimum.** If `a` is constant on the support, the update is
  zero, so training no longer drifts once it has converged.
* **It composes with the IS grid.** With any `rl.is.*` token correction, `c` is the normalized
  `q * w(p/q)`, computed with the same `log_mass` function as the per-token weight. The correction
  then only decides how the per-token signal is weighted, and no correction can reintroduce the drift.
  Naive IS gives `c = p`, which is the original loss. Sequence-level corrections rescale `A` and keep
  `c = q`.

**What the sampler must record.** An estimator that sees only `q(y)` for the sampled token cannot
compute `c = q w / Z` unless `q w` is proportional to something known. The only weight for which
that holds is `w = p/q`, which gives `c = p`. So with only the sampled logprob, the unique drift-free
choice is untruncated token IS. The loss falls back to it automatically when nothing else is
recorded (no correction and no sampler distribution in the batch). That fallback has the variance
problems described in section 3. The framework can record more: `sampler.vocab_logprobs=k` returns
the exact top-k head of `q` at every position (`sampler_topk_ids`, `sampler_topk_logprobs`). The
cost is `B x T x k`, which is ~k/V of the full distribution, and no extra forward pass is needed.

**Top-k estimate of `c`.** Let `K` be the head plus the sampled token, whose `q(y)` is always known.
Then:

* `c` is exact on `K`;
* the leftover mass `Q = 1 - sum_K q` is spread like `p` on the tail:
  `qhat_v = p_v * Q / (1 - sum_K p)` for `v not in K`;
* the tail's contribution `sum_{v not in K} qhat_v log p_v` is computed in closed form from
  `sum_v p_v log p_v` (no scatter into the vocabulary), so it also works with a TP-sharded vocabulary;
* the full distribution (`vocab_logprobs=full`) is used exactly when it is present.

The remaining drift is `abar (q - qhat)`. It is supported on the tail and bounded by
`|abar| * 2 * Q`, compared with `|abar| * ||q - p||_1` before. It is zero when `q` is proportional to
`p` on the tail. The loss logs `sampler_tail_mass` (`Q`) so that this can be monitored.

## 5. Implementation (`losses.py`)

* `loss_fn_rl` keeps its signature. With default arguments, each token's score is
  `log p(y) - center`, where `center = sum_v c_v log p_v` is computed by `_sampler_center` inside the
  rematerialized vocabulary chunks.
* `c = q` with no correction, or `c = normalized q*w` with a token correction. `q` comes from
  `sampling_vocab_logprobs` (exact), or from the top-k head with the tail estimate.
* If the batch carries only `sampling_token_logprobs` and no correction is given, the loss uses
  untruncated token IS (`c = p`), the only drift-free option in that case.
* All masses are detached, and the masked and invalid tail cases are guarded against NaN. A unit test
  confirmed that the logit gradient equals `-A (e_y - q)/N` to float precision when `q` is fully
  known, and equals the IS gradient `-A r (e_y - p)/N` under `rl.is.level=token`.

## 6. Sanity check on the CPU proxy (`dev/mini.py`, noise 0.3, 600 steps)

Training reward is the sampler's reward, averaged over the last 50 steps:

| setting | original loss | centered, `vocab_logprobs=k` |
| --- | --- | --- |
| default vocab, seed 0 | 0.58 (stalls) | 0.85 (k=4) |
| +20 tail tokens, seed 0 | 0.00 (collapses by step 50) | 0.82 (k=4), 0.85 (full) |
| +20 tail tokens, seed 1 | 0.00 | 0.81 (k=4) |
| +20 tail tokens, seed 2 | 0.00 | 0.86 (k=128) |
| +200 tail tokens, seed 0 | 0.00 | 0.80 (k=8) |
| +20 tail tokens, TIS / naive IS (original) | 0.55 / 0.57 (sampler degrades) | |
| +20 tail tokens, only `q(y)` (IS fallback) | | 0.31 (late decay) |

`./run_dev.sh sampler.vocab_logprobs=64` (the same as 128 here, because the vocabulary has 12 tokens) gives peak 0.95, a mean of 0.86 over the last 10 steps, and
no collapse.

## Configuration
sampler.vocab_logprobs=128

# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The REINFORCE loss `-A log p(y)` depends on the score function having zero
mean under the sampling distribution. When the sampler `q` is not the trainer
`p`, the score's mean at every position is `E_{y~q}[grad log p(y)] = -grad KL(q||p) != 0`.
Any part of the advantage that is constant given the prefix (the baseline,
the group mean, the value of the prefix) then stops cancelling. It turns into
a deterministic update that pulls `p` toward the corrupted sampler on
positive-advantage prefixes and pushes `p` away from it on negative-advantage
prefixes. That is the systematic, compounding bias.

The fix subtracts the score's exact conditional mean under `q` at every token:

    loss = -(1/N) sum_{i,t} A_i [ w(y_t) log p(y_t|s_t) - sum_v sg(q(v|s_t) w(v)) log p(v|s_t) ]

(`w = 1` without a correction, `sg` = stop-gradient). The update then has
zero mean for every prefix and every advantage offset, so the drift term is
gone. When `q = p` the gradient of the subtracted term is exactly zero, so the
loss is unchanged on-policy.

## 1. Setup at one token position

Fix a prefix `s` (prompt plus the tokens generated so far). Let `z` be the
trainer's logits, `p = softmax(z)`, and let the next token be drawn as `y ~ q(.|s)`.
Every token of rollout `i` carries the sequence advantage `A_i` (group-centered reward).
Define

- `Q(s,y) = E[A | s, y]`, the expected advantage after choosing `y` (the continuation is still sampled from `q`);
- `Abar(s) = E_{y~q}[Q(s,y)] = E[A | s]`, the expected advantage of the prefix.

Per token, the current loss is `l = -A log p(y)`. Its gradient is `-A g(y)`, with
`g(y) = grad_theta log p(y|s)`. With respect to the logits it is
`grad_z log p(y) = e_y - p`.

The update is `u = A g(y)`. Conditionally on `s`, with `y ~ q`:

    E_q[u | s] = sum_y q(y) Q(s,y) g(y)
               = sum_y q(y) (Q(s,y) - Abar(s)) g(y)   +   Abar(s) * sum_y q(y) g(y).      (1)
                 \______ signal term ________/          \______ offset term ______/

### On-policy (q = p)

    sum_y p(y) g(y) = sum_y grad p(y) = grad 1 = 0.

The offset term is identically zero, whatever `Abar(s)` is. This is why REINFORCE
tolerates any baseline: a constant, a group mean, or the fact that a single
sequence advantage is reused at every token. What remains is
`sum_y p(y)(Q - V)(s,y) g(y)`, the policy gradient. In logits it is `p ⊙ (Q - Abar)`.

### Mismatched (q != p)

    sum_y q(y) g(y) = sum_y q(y) grad log p(y) = -grad_theta KL(q || p)   (q held fixed)
    in logits:  sum_y q(y)(e_y - p) = q - p  != 0.

So the expected update contains the extra term

    b(s) = Abar(s) * (q - p)   (logits),   i.e.   -Abar(s) * grad KL(q(.|s) || p(.|s)).        (2)

Properties of this term:

1. **It is a bias, not noise.** It is deterministic given `(s, q, p)`. More
   samples do not average it away. Its size is `|Abar(s)| * ||q - p||`, which
   does not depend on how informative the reward is.
2. **Its direction comes from the sampler, not from the task.** On prefixes
   that tend to succeed (`Abar > 0`), `p` is distilled toward `q`: the trainer
   learns the sampler's corruption. On prefixes that tend to fail (`Abar < 0`),
   the update is gradient *ascent* on `KL(q||p)`, which is unbounded. It pushes
   down the tokens `q` over-weights and pushes up the tokens `q` under-weights.
3. **Group centering does not remove it.** Centering makes `E[A | prompt] ≈ 0`
   only at the first completion token. At later positions `Abar(s_t) = E[A | s_t]`
   is nonzero precisely when the prefix is informative. A rollout has many
   tokens, and each carries the full `A_i`, so the bias is summed over the
   whole sequence.
4. **It compounds.** The sampler is rebuilt from the trainer each step:
   `q = C(p)` for a fixed corruption `C` (weight noise, int8/fp8, KV
   quantization, staleness). With the `Abar > 0` part, `p <- p + eta*Abar*(C(p) - p)`
   has no fixed point when `C` is not the identity. The trainer absorbs `C`, the
   next sampler applies `C` again on top, and the error accumulates step after
   step. The `Abar < 0` part increases `KL(q||p)`, and that raises `||q - p||` and
   with it the size of (2): a positive feedback loop. Together these explain the
   slow drift that ends in collapse.

### Why IS weights and clipping only slow it down

With a detached weight `w(y) = w(p(y)/q(y))`, the offset term becomes
`Abar * sum_y m_y g(y)` with `m_y = q(y) w(y)`. In logits this is `Abar * (m - |m| p)`.
It vanishes only if `m ∝ p`, which means the untruncated ratio `w = p/q`
(naive token IS, which has high variance). Any band, clamp or drop replaces
`m` by something like `min(p, c q)` or `p * 1[in band]`. Then `m - |m| p != 0`,
and the leaked bias keeps the form in (2) with a smaller effective
`q - p`. A tighter token band moves `m` further from `p`, so the leaked term
can grow ("stronger clipping can make it worse"). A sequence-level weight
multiplies `A_i` by a scalar and leaves `sum_y q(y) g(y) != 0` at every
position, so the offset term remains.

## 2. The fix: center the score under the sampler

Replace the per-token score `w(y) g(y)` by

    ghat(y) = w(y) g(y) - mu(s),     mu(s) = E_{y'~q}[w(y') g(y')] = sum_v q(v) w(v) g(v).

Since `mu(s)` is the exact conditional mean, `E_{y~q}[ghat(y) | s] = 0` for every
prefix and for any `q`, `w`. Therefore, for any `c(s)` that does not depend on `y`:

    E_q[A ghat | s] = sum_y q(y) Q(s,y) ghat(y) = sum_y q(y) (Q(s,y) - c(s)) ghat(y).            (3)

Baseline invariance is restored under `q`, and the offset term (2) is
identically zero. For `w = 1`:

    E_q[A ghat | s] = sum_y q(y) (Q(s,y) - Abar(s)) g(y)      (logits: q ⊙ (Q - Abar)).

This is the covariance under `q` between a token's advantage and its score.
It raises tokens that are better than the prefix average and lowers tokens
that are worse. It has no component along `q - p`, so `p` no longer moves
toward or away from `C(p)` independently of the reward. The fixed-point-free
drift and the KL feedback loop are both removed. The remaining off-policy
error is the reweighting `q ⊙ (.)` in place of `p ⊙ (.)` of the
*zero-mean* signal `Q - Abar`. It is `O(||q - p||)` times the advantage
*spread*, not the advantage offset, and its sign follows the sampled
outcomes rather than a fixed sampler direction, so it does not accumulate the
same way.

**Loss.** Because `mu` uses detached weights, a surrogate with gradient `ghat` is

    l = -A [ w(y) log p(y) - sum_v sg(q_v w_v) log p_v ]
      = -A [ w(y) log p(y) + H_w(q, p) ],

which adds a per-token cross-entropy to the sampler, weighted by `-A`. In
logits, for `w = 1`, the gradient is `-A (e_y - q)` instead of `-A (e_y - p)`.
The normalizer that REINFORCE subtracts uses the sampler's distribution, which
is the one the token was actually drawn from.

**No change on-policy.** If `q = p`, `sum_v p_v grad log p_v = grad sum_v p_v = 0`
exactly. The gradient equals plain REINFORCE (and plain IS-REINFORCE for any `w`).

**With an IS correction.** The same centering uses `m_v = q_v w(p_v/q_v)`,
computed with the existing `log_mass`. Then `E_q[ghat] = 0` holds for every
band, clamp, drop or sign rule, so all the grid's corrections lose their
leaked bias as well. The correction is optional. The default (`correction=None`)
already removes the bias.

## 3. Implementation (`losses.py`)

`loss_fn_rl` keeps its signature. For each completion token,
`score = w * log p(y) - base`, where `base` comes from `sampler_baseline`
and is the surrogate `sum_v sg(m_v) log p_v`. The term is always on.

- **Full vocabulary** (`sampling_vocab_logprobs`): exact `m = q w`.
- **Top-k head** (`sampler_topk_ids/logprobs`, from `sampler.vocab_logprobs=k`):
  `q` is known on the head and at the sampled token `y`, which is added to
  the known set if it lies outside the head. The leftover mass
  `Q_tail = 1 - Q_known` is spread over the unrecorded tokens in proportion
  to `p`: `qhat_v = Q_tail p_v / P_tail`. The full-vocab tail sum is not needed,
  because `sum_v p_v grad log p_v = 0` gives
  `sum_{tail} p_v grad log p_v = -sum_{known} p_v grad log p_v`. So only `k+1`
  gathered log-probs per position are used (TP-safe through `take_vocab`). The
  residual bias is `Abar * sum_{tail}(q_v - qhat_v) g(v)`. It is bounded by the
  tail mass `Q_tail` (reported as metric `q_tail`) and has no component along
  the head of `q - p`. With an IS correction, the tail is treated as one block
  with ratio `P_tail/Q_tail`. Safeguards: `P_tail >= 1e-6`, tail coefficient `<= 100`.
- **Sampled token only** (no head recorded): the same formula with known set
  `{y}`. It reduces to the per-token weight `(1 - q(y))/(1 - p(y))`, which removes
  the bias only on high-probability tokens. This is why the configuration
  below records a top-k head.

I checked numerically that the gradient with respect to the logits equals
`-A (e_y - qhat)/N` in every mode, and equals naive IS when `w = p/q`.
On the CPU mini setup (noise 0.3), plain PG plateaued at an eval reward of
about 0.70. With 100 extra tail tokens it collapsed to 0. The centered loss with
`vocab_logprobs=4/8` reached about 0.83 in both cases (two seeds), and did not collapse.

## Configuration
sampler.vocab_logprobs=64

# Sampler-centered REINFORCE: removing the mismatch drift from the RL loss

## Summary

The REINFORCE loss `-A * log p(y)` relies on the fact that the score
`grad log p(y)` has zero mean when `y ~ p`. That fact is the reason a baseline,
group centering, or the many tokens that do not affect the reward cost nothing
in expectation. When the tokens come from a different sampler `q`, the score's
mean is `E_q[grad log p] = -grad_p KL(q || p)`, which is not zero. Every token
then adds `A * (-grad KL(q||p))` to the update. This is a value-weighted KL
gradient: it pulls `p` toward `q` on prefixes with positive value and pushes `p`
away from `q` on prefixes with negative value. It is a bias, not noise. Its
direction comes from the perturbation, so it points the same way at every step,
and it grows the mismatch that produces it.

The new loss subtracts that mean inside the loss. It uses
`grad log p(y) - E_{v~q}[grad log p(v)]` as the score, so the score has zero
mean under `q` as it does on-policy. In logit space this score is exactly
`e_y - q`, the score of the sampler itself. The expected update becomes the
on-policy policy gradient of the policy that generated the data. There are no
importance weights, so there is no variance blow-up and no clipping bias. The
sampler's distribution is needed only on its top-k ids. The rest of the
vocabulary goes into one tail bucket, and the identity stays exact on that
coarse-grained vocabulary.

## 1. Setup and notation

- `h` is a prefix (prompt plus generated tokens), `z = z_theta(h)` are the
  trainer's logits and `p = softmax(z)`. The sampler's next-token distribution
  is `q = q(.|h)`, so `y ~ q`, and the prefix distribution `d_q` is also
  produced by `q`.
- `A` is the sequence advantage (for example group-centered `r_i - mean_j r_j`
  or `+-1`). Define `Q(h, y) = E[A | h, y]` and `V(h) = E_{y~q}[Q(h, y)]`.
- For a softmax, `grad_z log p(y) = e_y - p` and
  `grad_z KL(q || softmax(z)) = p - q`.

The current loss is `L = -(1/N) sum_i A_i sum_t log p(y_it | h_it)`. Its
logit-space gradient at each token is `g = A (e_y - p)`, up to sign and the
1/N factor.

## 2. Expected update of the current loss

Fix a prefix `h` and take the expectation over `y ~ q` and over the future:

    E[g | h] = sum_y q_y Q(h,y) (e_y - p)
             = sum_y q_y (Q(h,y) - V(h)) e_y  +  V(h) (q - p)
             = q ⊙ (Q - V)                   -  V(h) * grad_z KL(q || p).        (1)

The total expected update is `E_{h ~ d_q}` of (1), backpropagated through
`dz/dtheta`.

**On-policy (`q = p`).** The second term is zero. What remains,
`p ⊙ (Q - V)`, is the exact policy gradient of `J(p)` in logit space. Adding
any constant to `A` changes nothing (baseline invariance). A token whose
choice does not affect the reward (`Q(h,.) = V(h)`) contributes zero in
expectation.

**Mismatched (`q != p`).** The extra term `-V(h) grad_z KL(q_h || p_h)` is
present at every generated token:

1. It is a bias. It does not shrink with batch size, and no IS-free
   variance-reduction trick removes it.
2. Its sign follows the value of the prefix. Where `V(h) > 0` it is gradient
   descent on `KL(q||p)`. Where `V(h) < 0` it is gradient ascent on
   `KL(q||p)`, which drives `p` away from the sampler.
3. Group centering does not cancel it. `sum_i A_i = 0` in each group only
   makes `V = 0` at the shared prompt, which is the first token. After that,
   `V(h)` is the conditional value of the partial answer (for group
   centering, about `(1-1/G)(V_r(h) - J(x))`). It is negative on every prefix
   that has already gone wrong. Under `plus_minus` it is negative on every
   prefix whenever the success rate is below 1/2.
4. It dominates the useful signal. The first term in (1) is non-zero only on
   tokens that change the expected reward, which is a small fraction of a
   long completion. The second term has size `|V| * ||q - p||` on every
   token, so for most tokens the expected update is pure drift.
5. It compounds. The sampler is rebuilt from the trainer at every step with
   the same kind of perturbation (a fixed noise draw, the same quantizer, the
   same staleness), so `q - p` points in a consistent direction and the drift
   adds up roughly linearly in the number of steps, not like `sqrt(steps)` as
   zero-mean noise would. On the negative-value prefixes it raises
   `KL(q||p)`, which enlarges `q - p` and therefore the drift itself. That
   positive feedback is the collapse.

## 3. Why IS corrections and clipping only slow it down

With the token-level weight `w = p_y / q_y`:

    E_q[w A (e_y - p) | h] = sum_y p_y Q (e_y - p) = p ⊙ (Q - V_p),

so the drift term cancels. The price is variance:
`E_q[w^2] = sum_y p_y^2 / q_y = 1 + chi^2(p || q)`. This is heavy-tailed for
tokens that `q` rarely samples but `p` likes. Single tokens with huge `w`
produce gradient spikes (in a dev run, one tail token's ratio caused a 100x
grad-norm jump and the run collapsed in the next step). Every way of taming `w` brings the drift back.
A clamp `min(w, c)` gives

    E_q[min(w,c) (e_y - p)] = sum_y min(p_y, c q_y)(e_y - p)
                            = - sum_y (p_y - c q_y)_+ (e_y - p)  != 0,

and a drop band `[a, b]` gives `-sum_{y outside} p_y (e_y - p)`. Each is again
`V(h)` times a fixed vector set by the mismatch. A tighter band moves more mass
outside, so the residual drift is larger. This is why stronger token clipping
can make things worse. Sign-dependent rules (PPO, DAPO, TOPR) and
ratio-dependent filters at sequence level (GSPO) apply different
distortions to positive and negative advantages, or select samples by their
ratio. The selected samples are then no longer distributed as `q`, and the
`V(h)`-weighted term reappears.

## 4. The fix: center the score under the sampler

Subtract from each token's score its mean under the distribution that produced
the token:

    s(y) = grad log p(y) - E_{v~q}[grad log p(v)].

As a loss, with stop-gradients `sg`:

    L = -(1/N) sum_i A_i sum_t [ log p(y_t) - sum_v sg(q_v) log p(v) ].         (2)

The gradient of the second term is `sum_v q_v grad log p(v)`. Equivalently, it
can be written with `sg(q_v - p_v)`, since `sum_v p_v grad log p(v) = 0`.

**Logit space.** `grad_z s = (e_y - p) - (q - p) = e_y - q`. This is the score
of the sampler itself, `grad_{z_q} log q(y)`. The expected update is therefore

    E[A (e_y - q) | h] = q ⊙ (Q - V_q),                                         (3)

which is (1) with the drift term removed. Properties:

- **Baseline invariance is back.** `E_q[e_y - q] = 0`, so a constant in `A`,
  any non-centered reward mode, or any token that does not matter contributes
  nothing in expectation, for every `q`. With no reward signal
  (`Q = V`), the expected update is exactly zero. The old loss would still
  move by `V (q - p)` per token.
- **It is the on-policy gradient of the policy that is actually sampled.**
  The sampler's logits are `z_q = z_theta + epsilon`, where `epsilon` is the
  weight-noise, quantization, KV-cache or staleness error. The true gradient of
  the sampler's reward is `grad_theta J(q) = E_{d_q}[(dz_q/dtheta)^T A (e_y - q)]`.
  Loss (2) gives `E_{d_q}[(dz_theta/dtheta)^T A (e_y - q)]`, which has the
  same sampling distribution (`d_q`), the same score, and no importance
  weights. Both prefixes and continuations are on-policy for `q`, so the
  state-distribution mismatch that token-level IS ignores does not arise. The
  only remaining difference is the Jacobian (`dz_theta/dtheta` instead of
  `dz_q/dtheta`). That is a preconditioning error on the true signal, and it
  vanishes when the signal vanishes. It is not a term that pushes `p` around
  when `A` carries no information. Because `q` is rebuilt from `p` every step,
  improving `q` through `p`'s logits is exactly what training needs.
- **Variance.** There are no ratios, and `||e_y - q|| <= sqrt(2)`, the same
  bound as on-policy REINFORCE. Nothing needs clipping, so no clipping bias is
  added.
- **No feedback loop.** There is no `KL(q||p)` term with a `V`-dependent sign,
  so the trainer is never pushed away from its sampler and the mismatch stays
  at the size set by the perturbation.

## 5. Implementation with partial knowledge of q

Equation (2) needs `q` on the whole vocabulary. Storing `[B, T, V]` for a 150k
vocabulary is too expensive, but the sampler can record its exact top-k
(`sampler.vocab_logprobs=k`, keys `sampler_topk_ids` / `sampler_topk_logprobs`).
At each position, let `K` be the sampler's top-k ids. `K` depends only on the
prefix, not on `y`. Let `T` be the remaining ids. Coarse-grain the vocabulary
into `K ∪ {T}`:

    p~_v = p_v,  q~_v = q_v        (v in K)
    p~_T = P_T = 1 - sum_K p_v,    q~_T = Q_T = 1 - sum_K q_v
    y~ = y if y in K, else T

On this categorical, both distributions are known exactly, and (2) becomes

    L = -(1/N) sum_i A_i sum_t [ log p~(y~_t) - sum_{u in K∪{T}} sg(q~_u) log p~_u ].   (4)

**Exactly drift-free.** `E_{y~q}[grad log p~(y~)] = sum_u q~_u grad log p~_u`,
so the expected gradient for a constant `A` is exactly 0 for any `q` and any
`k`.

**What it computes.** With `grad_z log P_T = p_T/P_T - p`, the logit-space
score is `e_y - q^` for `y in K` and `p_T/P_T - q^` for `y in T`. Here
`q^ = (q on K) + Q_T * p(.|T)` is the sampler's head with its tail shaped like
the trainer's tail. The expected update is `q^ ⊙ (Q^ - V_q)`, where `Q^` equals
`Q` on `K` and the `q`-average of `Q` over `T` on the tail. This is the exact
sampler policy gradient on the coarse-grained vocabulary. What is lost is the
redistribution of mass inside the tail, which is mostly mismatch noise anyway.
It still uses no importance weights. With `K = V` (`sampler.vocab_logprobs=full`,
key `sampling_vocab_logprobs`) the code uses (2) directly.

**Numerics.** `P_T` is computed as `1 - sum_K p_v` in float32. If it falls
below `1e-6` (cancellation; this happens when `k >= V`) the bucket is dropped,
which is the full-vocabulary case.

**Fallback (no vocabulary information).** If only `q(y_t)` is available, the
only score that has zero mean under `q` is the token ratio, so the loss uses
`sg(p/q) log p(y)`. This is naive token IS: drift-free in expectation but
heavy-tailed, as in section 3. It is the default when the batch has no
sampler logprobs. The configuration below turns on the top-k version, which
is the intended operating mode.

**Corrections.** `loss_fn_rl` keeps its signature. With `correction=None`, the
centered loss (4), (2) or the fallback is chosen automatically from the batch
keys. A sequence-level `ISCorrection` still multiplies the advantages. A
token-level one only changes the fallback's ratio. Neither is needed, and
ratio-dependent ones bring part of the drift back (section 3).

**Check.** On a toy categorical, the `q`-expectation of the loss gradient with
constant `A`, computed by enumerating every token, is zero to float precision
(about 3e-8) in all three modes (full, top-k, none). With token-dependent `A`,
the full mode reproduces `q ⊙ (A - E_q A)` exactly.

## 6. Dev runs (CPU mini setup, relative weight noise 0.3, plus_minus rewards, 600 steps)

| setting | original loss | centered, top-k (k=8/32/64) | centered, full | fallback (token IS) |
| --- | --- | --- | --- | --- |
| vocab 12 | plateau, last 0.58 | last 0.83-0.85, peak 0.95 | last 0.86 | last 0.64 |
| vocab 212 (200 extra tokens) | collapses to 0 at step about 25 | last 0.73-0.84, no collapse (3 seeds) | last 0.84 | survives, with dips |
| vocab 62, noise 0.5 | collapses to 0 immediately | last 0.53, still rising | – | collapses |

## Configuration

sampler.vocab_logprobs=64

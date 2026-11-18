# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

With a mismatched sampler, REINFORCE's score function is no longer mean zero
under the distribution that produced the samples. Without that property the
advantage's offset (its baseline) stops cancelling and becomes a **drift**
`(V - b)(q - p)` in logit space. The drift is not the gradient of any reward.
The mismatch fixes its direction and the prefix's advantage fixes its sign, and
because the sampler is refreshed from the trainer at every step it adds up over
training. IS weights and clipping can remove it only with an unclipped
ratio `p/q`, which has unbounded variance. Clipping, dropping or
sequence-level weights all leave part of it in place.

The fix is to center each token's score by its expectation under the sampler:

```
loss = -(1/N) sum_i A_i sum_t w_it [ log p(y_t | s_t) - sum_v pi_t(v) log p(v | s_t) ],
pi_t  = q(. | s_t) (no correction; in general pi_t ∝ q w(p/q), detached)
```

The logit gradient changes from `A (e_y - p)` to `A (e_y - q)`. This is exactly
mean zero under `q` for any advantage offset. It is identical to REINFORCE
per sample when `q = p`, and in expectation it equals the exact policy gradient
of the sampler's own return, so it is an ascent direction with no drift term.
It needs `q` over the vocabulary. The sampler already records the top-k head
(`sampler.vocab_logprobs=k`), and the remaining vocabulary goes into one
lumped bucket.

## 1. Setup

Notation: a state `s` (prompt plus prefix) and trainer logits `z(s)`, with
`p = softmax(z)`. The sampler's distribution is `q(.|s)`, from noisy, quantized
or stale weights. Rollouts are drawn from `q`, the sequence advantage is `A` (a
reward minus a baseline: `2r-1` in `plus_minus`, `r - mean_group(r)` in
`group_centered`), and the current loss is

```
L = -(1/N) sum_i A_i sum_t w_it log p(y_it | s_it)        (w = 1 without correction)
grad_z log p(y|s) = e_y - p(.|s)                         (softmax score)
```

For a token at state `s` reached under `q`, define the sampler action value
and value

```
Q(s,y) = E_q[A | s, y],      V(s) = sum_y q(y|s) Q(s,y)
```

Both continuations run under `q`, which is how the data is generated. A
constant offset of the advantage (the reward baseline `b`, or the
`-1` of `plus_minus`) moves `Q` and `V` by the same amount.

## 2. What goes wrong: the score is not mean zero under q

REINFORCE relies on the identity `sum_y p(y) grad log p(y) = 0`. Any constant
added to the advantage then vanishes in expectation, and that is why baselines
are "free". When the samples come from `q`, the identity that matters is
`sum_y q(y) grad log p(y)`, and it is not zero:

```
E_{y~q}[e_y - p] = q - p .
```

At a state `s`, the expected logit update of the current loss (`w = 1`) is therefore

```
E_q[A (e_y - p)] = sum_y q(y) Q(s,y) (e_y - p)
                 = q ⊙ Q - V p
                 = q ⊙ (Q - V)   +   V (q - p).                      (1)
                   └ useful ┘        └ drift ┘
```

* The first term is the exact policy-gradient direction of the sampler's own
  objective (Sec. 4). Its inner product with `Q` is `Var_q(Q) >= 0`.
* The second term, `V(s) (q - p)`, does not depend on which tokens are better.
  It is present even when every token at `s` is equally good (`Q` constant),
  which is the case for most of the vocabulary. Its direction is the mismatch
  `q - p`, and its sign is the advantage of the prefix:
  - On prefixes with `V < 0` (failing prefixes, most prefixes early in
    training under `plus_minus`, and failing branches under group centering),
    `z += eta |V| (p - q)`. The update pushes `p` **away** from the sampler:
    it raises tokens the sampler under-samples (`p > q`) and lowers tokens it
    over-samples, without regard to reward. Concretely, pushing down a sampled
    token moves its mass onto other tokens in proportion to `p`. On-policy that
    mass returns because those tokens are sampled in proportion to `p` and
    pushed down in turn. Off-policy, tokens with `p > q` are under-sampled, so
    they gain more mass than they lose, and the effect is largest in the
    low-probability tail.
  - On prefixes with `V > 0`, the update pulls `p` toward `q`, that is, toward
    the trainer weights plus the perturbation.
* **It compounds.** The sampler is a fixed function of the trainer (`theta +
  Delta`, quantize(`theta`), `theta_{t-k}`), so `q - p ≈ J_z(theta) * Delta` is
  a smooth, persistent vector field, not fresh zero-mean noise. Each step moves
  `z` along it. The refreshed sampler follows `p` and keeps the same offset, so
  `q - p` does not shrink and the displacement grows roughly linearly in the
  number of steps. In practice this shows up as probability mass leaking onto
  junk tokens, the trainer/sampler KL growing, and then the reward collapsing.
  The term is first order in the mismatch and is multiplied by `|V|`, which is
  large exactly when training is going badly, so degradation feeds itself.

For whole sequences, summing (1) over tokens gives
`E_q[A sum_t grad log p(y_t|s_t)] = sum_t E_{s_t~q}[q⊙(Q-V) + V(q-p)]`. The drift
appears at every position, and the per-sequence `A` multiplies all of them.

## 3. Why IS corrections and clipping only slow it down

Give each token a detached weight `w(r)`, with `r = p(y)/q(y)`. Write
`m(y) = q(y) w(p(y)/q(y))` and `|m| = sum_y m(y)`. Then

```
E_q[w A (e_y - p)] = sum_y m(y) Q(y) (e_y - p)
                   = m ⊙ (Q - V_m) + |m| V_m ( m/|m| - p ),     V_m = sum m Q / |m|.     (2)
```

The drift vanishes for every `V` if and only if `m ∝ p`, which means `w ∝ p/q`
on every token, including tokens that are never sampled.

* Unclipped token IS (`w = p/q`) satisfies the condition, but its variance is
  `sum_y p(y)^2/q(y)`, which is unbounded on tail tokens where `q` is small.
  It also does not correct the prefix distribution.
* TIS clamps at `C`: `m(y) = min(p(y), C q(y))`, so `m` loses mass exactly on
  the tokens with `p > C q`, the ones the sampler under-samples. Those are the
  tokens the drift raises. On a failing prefix the update on such a token is
  `+eta |V| (|m| p(y) - C q(y)) > 0`. A smaller `C` puts more tokens in this
  regime, so **stronger token-level clipping can increase the drift**. MIS
  and PPO-style dropping (`w = 0`) are worse, because dropped tokens are never
  pushed down at all. Sign-dependent variants (PPO, TOPR) leave different
  biases for `A > 0` and `A < 0`, and these do not cancel either.
* Sequence-level weights (GSPO, TOPR) multiply all the tokens of a sequence by
  one scalar. This cannot make the per-token measure `m(.|s)` proportional to
  `p(.|s)`, so the term `V(q-p)` survives at every state.

So corrections reduce the drift's magnitude, by shrinking the step on very
mismatched tokens, but they do not change its direction. Training still
degrades, only more slowly. This matches what was already known about these
corrections.

## 4. The fix: center the score under the sampling distribution

Replace the score of each token by its deviation from its own expectation
under the measure that generated it:

```
s_t = log p(y_t|s_t) - sum_v pi_t(v) log p(v|s_t),      pi_t = sg( m_t / |m_t| ),  m_t(v) = q(v|s_t) w(p(v)/q(v))
loss = -(1/N) sum_i A_i sum_t w_it s_it
```

Without a correction, `w = 1` and `pi = q`. The term `sum_v pi(v) log p(v)` is
the negative cross-entropy of the sampler against the trainer. Because
`sum_v pi(v) = 1`, the log-normalizer of `p` cancels:
`s_t = z_y - sum_v pi(v) z_v`. The logit gradient is

```
grad_z s_t = (e_y - p) - (pi - p) = e_y - pi .                              (3)
```

**Property 1 (no drift for any baseline).**
`E_q[w (e_y - pi)] = m - |m| pi = 0`. A constant added to `A`, which includes
the reward baseline, the `-1` of `plus_minus`, and the prefix value `V` in (1),
has zero expected effect again. The drift term of (1)/(2) is removed exactly,
not only reduced. This holds per state, at every position, whatever form the
mismatch takes.

**Property 2 (identical on-policy).** If `q = p` (and `w = 1`),
`sum_v p(v) grad log p(v) = 0` holds identically, not only in expectation, so
the per-sample gradient equals REINFORCE's. I checked this numerically: the
maximum difference was 1e-8. The change only matters off-policy.

**Property 3 (it is the policy gradient of the sampler's own objective).**
With `w = 1`, the expected update at `s` is

```
E_q[A (e_y - q)] = q ⊙ (Q - V),
```

which is exactly `grad_{z_q} J(q)` for the sampler's logits `z_q`, because
`grad_{z_q} log q(y) = e_y - q`. Over whole sequences,
`grad J(q) = E_q[A sum_t (e_{y_t} - q_t)]`. The trajectory distribution,
the continuation values `Q` and the score all refer to the same policy `q`, so
no prefix (sequence-level) correction is needed. The update direction `g`
satisfies `<g, Q> = Var_q(Q) >= 0`: it never decreases the sampler's return to
first order. Its fixed points (`g = 0` on the support of `q`) are those of the
true objective, namely `q` concentrated on maximal-`Q` actions.

The parameters receive `J_p(theta)^T g`, where `J_p = dz_p/dtheta`. The exact
update for `q` would use `J_q`. The mismatch therefore enters only as a
Jacobian distortion of a vector `g` that is itself driven by `Q - V`.
It is multiplicative: it scales and rotates a useful direction and vanishes
where `g` vanishes. In particular, it vanishes for reward-irrelevant tokens and
at convergence. The original loss instead adds the term `V (q - p)`, which is
present even when `g = 0`. For mismatches that act on the logits, such as
quantized activations or a KV cache that perturbs `z` by a slowly varying
offset `delta(theta)`, `J_q ≈ J_p` and the update is exactly the sampler's
policy gradient.

**Property 4 (no ratios, so no variance blow-up).** No importance ratio is
used. The extra term is a deterministic expectation over the vocabulary, a
Rao-Blackwellized control variate: it is a function of `s` only, so it
reduces variance rather than adding to it. For tokens the sampler under-samples,
the step stays bounded: the push-up of a rare token `v` is `A q(v)`, not
`A p(v)`.

**Generalization to corrections.** With a correction `w`, using
`pi ∝ q w(p/q)` (computed with the existing `log_mass`) makes (2) drift-free
for any `w`. For `w = p/q`, `pi = p` and the extra term is identically zero:
plain IS is recovered, as it should be. Every point of the `rl.is.*` grid
therefore stays usable and gains the same guarantee. The recommended
configuration is still no correction, because clipping adds nothing once the
drift is gone and only discards signal.

## 5. Implementation (`losses.py`)

`loss_fn_rl(hidden, head, batch, correction=None, chunk=128)` keeps its
signature and chunked, rematerialized structure. Each chunk adds
`-A w sum_v pi_v log p_v` to the token's score. Where `q` comes from depends on
what the batch contains:

1. `sampling_vocab_logprobs` [B,T,V] (`sampler.vocab_logprobs=full`): `pi` is
   exact over the whole vocabulary.
2. `sampler_topk_ids` / `sampler_topk_logprobs` [B,T,k]
   (`sampler.vocab_logprobs=k`, **recommended**): the categories are the k
   head tokens `H` plus one rest bucket `R`, with
   `q(R) = 1 - sum_H q` and `p(R) = 1 - sum_H p` (floored at 1e-6). The
   gradient becomes `e_y - [q_H, q(R) p_R/p(R)]`. It is exact on the head and
   mass-correct on the tail, so the only remaining drift lies inside the
   tail:

   ```
   V (q_R - q(R) p_R / p(R)),     with L1 norm <= 2 |V| q(R),
   ```

   which is small when the k head tokens cover most of the sampler's mass. For
   Countdown with Qwen3, k = 64 covers essentially all of it. This is also the
   exact gradient of the objective coarsened to the categories `H ∪ {R}`. Top-k
   ids are resharded to the token layout before the vocabulary gather, so the
   loss also runs with a TP-sharded vocabulary.
3. Only `sampling_token_logprobs` (fallback): with `q(y)` alone, (2) shows that
   the only weight that makes the score mean zero under `q` is `w = p/q`.
   With no correction, the loss falls back to unclipped token-level IS, which
   is `pi = p` in the general formula above. This is drift-free but has
   higher variance, so the head should be recorded.

The loss also reports a `center_prob` metric: the mean `pi(y_t)`, the
centering mass on the sampled token.

## 6. Sanity checks (CPU `dev/mini.py`)

These ran with `plus_minus` rewards, 600 steps, and the results are the eval
reward at the end:

| setting | original loss | TIS (high=2, clamp) | this loss, top-k (k=4..16) or full |
| --- | --- | --- | --- |
| noise 0.3, seeds 0/1 | 0.67 / 0.68 (plateau) | – | 0.82–0.86 |
| noise 0.5 | collapses to 0.00 | – | 0.66–0.71, stable |
| noise 0.3, +100 tail tokens, seeds 0-2 | collapses to 0.00 (seeds 0, 2) | 0.77 (seed 0) | 0.75–0.83, stable |
| noise 0.5, +100 tail tokens | – | collapses to 0.00 | 0.74, stable |

With group-centered rewards (noise 0.5, +100 tail tokens), the loss reached
0.71 / 0.57 against 0.55 / 0.57 for the original. The effect is smaller there,
because `V - b` averages to zero at the root. The token-only fallback (IS) was
stable on one seed and collapsed late on another, which is why I recommend
recording the top-k head.

## Configuration

sampler.vocab_logprobs=64
rl.is.level=null

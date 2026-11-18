# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The REINFORCE loss `-A · log p(y)` is only free of a hidden term when the
tokens are sampled from `p`. If they come from a different sampler `q`, the
score `∇ log p(y)` no longer has zero mean, and its mean is
`E_{y~q}[∇ log p(y)] = -∇ KL(q ‖ p)`. Multiplied by the advantage, this adds a
term to the objective that does not depend on the reward:
`-Σ_s Ā(s) KL(q_s ‖ p_s)`. Wherever the conditional mean advantage is negative
it pushes the trainer **away** from the sampler, and it pushes the same way at
every step. The fix is to center the score under the sampler:

```
loss = -(1/N) Σ_tokens A · [ log p(y_t) - Σ_v sg(q(v | s_t)) log p(v | s_t) ]
```

The gradient of the bracket with respect to the logits is `e_y - q` instead of
`e_y - p`, and it has zero mean under `y ~ q` in every state. Only the part of
the advantage that varies with the action inside a state still moves the
policy. `losses.py` applies this in `loss_fn_rl` with its default arguments.
It reads `q` from the sampler's top-k logprobs (see Configuration), plus the
sampled token.

## 1. What is wrong with the current loss

Notation: `s` is a decoding state (a prefix), `z = z(s)` are the trainer logits,
`p = softmax(z)`, and `q = q(·|s)` is the sampler's next-token distribution.
The sampler has noisy, quantized or stale weights, so `q ≠ p`. A token
`y ~ q` gets advantage `A`, and the loss is `-Σ_t A log p(y_t | s_t)`. The
gradient with respect to the logits at one token is

    ∇_z log p(y) = e_y - p.

**Expected update.** Write `A(s, y) = E[A | s, y]` and
`Ā(s) = E_{y~q}[A(s, y)]`. The expected ascent direction, over states `s`
visited by the sampler (`d_q`), is

    g = E_{s~d_q} Σ_y q(y|s) A(s,y) (e_y - p)
      = E_{s~d_q} Σ_y q(y|s) (A(s,y) - Ā(s)) (e_y - q)      ... (i)  reward signal
      + E_{s~d_q} Ā(s) (q - p)                              ... (ii) drift

To get this, use `e_y - p = (e_y - q) + (q - p)` and
`Σ_y q(y) (A(s,y) - Ā(s)) = 0`. In parameter space, term (ii) is

    Ā(s) Σ_y q(y) ∇_θ log p_θ(y|s) = -Ā(s) ∇_θ KL(q(·|s) ‖ p_θ(·|s))   (q held fixed).

So the current loss secretly also optimizes

    J_hidden(θ) = - E_{s~d_q} Ā(s) · KL(q_s ‖ p_θ,s).

Properties of this term:

* **It does not depend on the reward.** It is zero only if `q = p`. It does not
  vanish in a state where every action is equally good (`A(s,y) ≡ Ā(s)`). In
  such a state the on-policy update would be exactly zero, but this one is
  `Ā(s)(q - p)`. So with a mismatch, a reward-optimal policy is never a fixed
  point.
* **Its sign follows the advantage, and it pushes the trainer away from the
  sampler.** Where `Ā(s) < 0` it *maximizes* KL(q ‖ p). Such states are prefixes
  that tend to fail, most states early in training under `plus_minus` or
  `zero_minus` rewards, and especially states reached after a token that the
  sampler likes and the trainer does not. The cheapest way to increase
  KL(q ‖ p) is to drive `p(v) → 0` on tokens where `q(v) > 0`, which is the
  low-probability tail. That is why mismatched tail tokens are where collapse
  starts. Where `Ā(s) > 0` it pulls `p` towards the *perturbed* model, which
  means fitting the noise.
* **Baselines are no longer free.** A constant baseline `b` adds
  `-b E_q[∇ log p] = b ∇KL(q‖p) ≠ 0`. Group centering makes `Σ A = 0` at the
  root state of a group, which kills (ii) there. Deeper states have
  `Ā(s) ≠ 0`, because success and failure separate along the trajectory. The
  term also adds up over every completion token, so long failing completions
  contribute the most.
* **It compounds.** The sampler is the trainer plus a fixed perturbation (or a
  fixed quantization map), and it is refreshed from the trainer every step. In
  logits, `z_q = z_p + Δ(s)`, where `Δ` is set by the perturbation and not by
  the update. Term (ii) is `Ā(s)(softmax(z_p + Δ) - softmax(z_p))`. This
  direction is set by `Δ` and it comes back with the same sign at every step,
  so it adds up linearly over training rather than averaging out like zero-mean
  noise. As `p` sharpens away from `q`, `|q - p|` and the mismatch KL grow
  (`kl_samp_train` rises), so the term gets larger as it acts. That is the
  feedback loop behind collapse.

**Why IS and clipping only slow it down.** With an exact token ratio
`w = p/q`, `E_q[w ∇ log p] = Σ_y p(y) ∇ log p(y) = 0`, so the drift cancels in
expectation, but only through a large-variance weight. Every practical
correction bounds or drops `w` (TIS, MIS, PPO, DPPO). With a modified weight
`w̄`:

    E_q[w̄ (e_y - p)] = Σ_y q w̄ e_y - (Σ_y q w̄) p = Z (π - p),   π = q w̄ / Z,  Z = Σ q w̄,

so the drift comes back as `-Ā Z ∇KL(π ‖ p)`, with `π ≠ p` whenever any
weight is clipped or dropped. Stronger clipping moves `π` further from `p`
(toward `q`, or toward `q` with holes cut out), which makes the residual drift
*larger*. Sequence-level weights fix the state distribution, but only through
weights whose variance grows exponentially with length, and once these are
clipped they leave the same residual. A token-level correction also never
corrects `d_q` itself. In every case the problem is that the score being
weighted is centered at `p`, and the data come from `q`.

## 2. The fix: center the score under the sampler

Replace the score by its sampler-centered version:

    ŝ(y | s) = log p(y|s) - Σ_v sg(q(v|s)) log p(v|s)
             = log p(y|s) + CE(sg q, p)

    ∇_z ŝ = (e_y - p) - (q - p) = e_y - q,      E_{y~q}[e_y - q] = 0.

**Derivation of the expected update.** Repeat the decomposition above with
`e_y - q`:

    g_new = E_{s~d_q} Σ_y q(y|s) A(s,y) (e_y - q)
          = E_{s~d_q} Σ_y q(y|s) (A(s,y) - Ā(s)) (e_y - q)      = term (i) only.

Term (ii) is gone exactly, in every state, for any `Ā(s)`, whatever its sign
or size. Consequences:

1. **Correct fixed points.** If all actions in a state have the same advantage,
   the update there is zero. Constant baselines, uncentered rewards
   (`plus_minus`, `zero_minus`) and length effects cannot move the policy.
   Baselines are free again: `E_q[b · (e_y - q)] = 0`.
2. **It is on-policy REINFORCE for the sampler, applied in logit space.** With
   `z_q = z_p + Δ(s)`, we have `∂z_q / ∂z_p = I`. On-policy REINFORCE on the
   *sampler's own* rollouts would move its logits by `A (e_y - q)`, and this is
   exactly what the new loss applies to the trainer's logits. The sampler is
   rebuilt from the trainer, so at the next step it moves by the same amount up
   to second order (`δz_q = δz_p + O(‖δθ‖ · ‖∂Δ/∂θ‖)`). The trainer/sampler
   pair therefore follows ordinary on-policy policy-gradient dynamics. Those
   dynamics are stable and improve exactly the policy that generates the data.
   The offset `Δ` is carried along and is not amplified. The only remaining
   error comes from the parameter Jacobian (`J_p` in place of `J_q`). It is
   proportional to the size of the perturbation times the size of the update,
   and its sign is not fixed by `Ā`.
3. **No variance from ratios.** The centering term is a deterministic expectation
   over the vocabulary, with no `p/q` weights. It is never clipped, so it adds
   no clipping bias, and it cannot blow up on tail tokens where `p ≪ q`.

Contrast with IS: IS changes *which distribution the gradient estimates*
(from `q` back to `p`). It pays for that in variance, and as soon as it is
clipped it reintroduces bias. Centering keeps the estimator on `q`, the
distribution that actually produced the data, and only removes the part of the
score that has nonzero mean under `q`.

## 3. Combining with an IS correction (optional)

If a token-level correction `w(·)` is given, centering at `q` is no longer
enough: `E_q[w (e_y - q)] = Σ q w e_y - Z q ≠ 0`. The general condition is that
the weighted score has zero mean under `q`:

    E_{y~q}[ w(y) (e_y - π) ] = 0   ⟺   π(v) = q(v) w(v) / Σ_u q(u) w(u).

`loss_fn_rl` therefore centers with `π ∝ q·w`. It evaluates the same
`ISCorrection` band and sign rules on every known token (`w = 1` without a
correction, which gives `π = q`). For an exact unclipped ratio `π = p`, which
recovers IS. Sequence-level weights are constant within a sequence, so they
keep `π = q`.

## 4. What is known about q, and how the loss uses it

The centering needs `q(·|s)` over the vocabulary. The framework can record the
sampler's exact top-k head (`sampler.vocab_logprobs=k`, giving
`sampler_topk_ids` and `sampler_topk_logprobs`) or the full distribution
(`=full`). It always records `q(y)` for the sampled token. `loss_fn_rl` uses
whatever the batch has:

* `sampling_vocab_logprobs` present: `π = q` (times `w`) exactly.
* otherwise `q` is known exactly on `S = top-k ∪ {y}`. The rest of the sampler
  mass `1 - q(S)` is spread over the tail in proportion to the trainer:
  `q̂(v) = p(v) (1 - q(S)) / (1 - p(S))` for `v ∉ S`. Then
  `Σ_v q̂(v) log p(v) = Σ_{v∈S} q(v) log p(v) + c · (Σ_v p log p - Σ_{v∈S} p log p)`
  with `c = (1 - q(S)) / (1 - p(S))` detached. This only needs gathers at the k
  ids and the full-vocabulary `Σ p log p` that the chunked log-softmax already
  computes, so memory stays at `[B, chunk, V]`. The total tail mass is exact,
  so the remaining drift is `Ā · (q_tail - q̂_tail)`. That only measures how
  the *shape* of `q` and `p` differ beyond the top-k, and it is negligible
  for k of a few dozen.
* with no sampler vocabulary at all (default `sampler.vocab_logprobs=null`) the
  same formula with `S = {y}` is used. The gradient is exact at the sampled
  coordinate (`1 - q(y)`), and the rest of the vocabulary is assumed to have
  `p`'s shape. This removes only part of the drift. Use the configuration below.

`losses.py` checks this numerically. With full `q`, or with top-k at
`k = |V|`, the logit gradient of the loss equals
`-A (e_y - q) · mask / N` to float precision.

## 5. Evidence (dev/mini.py, CPU)

All runs use `plus_minus` rewards and a 600-step run. The table gives eval
reward at the start, the peak, and the end.

| setting | original loss | sampler-centered |
| --- | --- | --- |
| noise 0.3, 12-token vocab, seed 0 | 0.56 → 0.72 → 0.66 | 0.56 → 0.85 → 0.84 (full q) |
| noise 0.3, +100 tail tokens, seed 0 | 0.39 → **0.00** (collapse by step 50) | 0.39 → 0.84 → 0.80 (top-4) |
| noise 0.5, +100 tail tokens, seed 1 | 0.42 → **0.00** (collapse) | 0.40 → 0.60 → 0.58 (top-4) |
| noise 0.3, +100 tail tokens, seed 2 | (TIS: 0.36 → 0.78) | top-1: 0.37 → 0.78; top-4 + TIS: 0.36 → 0.80 |

With only the sampled token known (no top-k), the drift is reduced but not
removed: 0.39 → 0.10 instead of 0.00. This is why the configuration below
records a top-k head. With `group_centered` rewards and noise 0.5, the
original loss did not collapse in this short run, and both losses reach
about 0.57. Centering costs nothing there and protects against the
non-root-state drift that builds up over longer runs.

## Configuration

sampler.vocab_logprobs=32

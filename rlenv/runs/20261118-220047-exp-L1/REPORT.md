# Stable RL under a sampler/trainer mismatch

## What is wrong with the current loss

Let `q` be the sampler's distribution over completions `y` and `p` the trainer's. `losses.py` computes, with `correction=None`,

    loss = -(1/N) sum_i A_i sum_t log p(y_it | y_i<t),   y_i ~ q,

so its expected gradient is `g0 = E_q[A ∇log p]`. The quantity we want is the policy gradient of the trainer, `∇J = E_p[A ∇log p] = E_q[W A ∇log p]`, where `W(y) = p(y)/q(y) = exp(ℓ(y))` and `ℓ(y) = sum_t (log p_t - log q_t)` is the sequence log-ratio. The bias of the current loss is therefore

    b0 = E_q[(1 - W) A ∇log p].

This is not zero-mean noise. Two facts make it systematic:

1. With `A` constant, `E_q[∇log p] = -∇_p KL(q||p)` (q is fixed), so the unweighted push is an imitation of the sampler: positive advantages pull `p` toward `q`, negative advantages push `p` away from `q`'s typical completions.
2. The sampler is re-synced to the trainer every step (`staleness=0`), so `q` is always the trainer plus the same fixed perturbation. The bias does not average out across steps, and as `KL(q||p)` grows the bias grows with it. That is the compounding feedback loop.

The mismatch also acts on the negative side in a specific way. The negative-advantage term `E_q[A^- ∇log p]` pushes mass off the sampled completions. When `W < 1`, which is the typical case since `E_q[log W] = -KL(q||p) ≤ 0`, the unweighted term over-pushes exactly the completions the sampler over-produces. The mass removed from them flows into the trainer's own tails, which is the leakage behind the reward collapse.

## Why the existing corrections are not enough

- **Per-token ratios `r_t = p(y_t|y<t)/q(y_t|y<t)`** (naive IS, TIS, MIS, PPO, DAPO). The exact weight for a sequence-level advantage is the product over all tokens, `W = prod_t r_t`, including future factors that `A` depends on. A per-token weight `E_q[r_t A ∇log p_t]` corrects only the current factor and is biased. Its variance is also unbounded: `log W` is a sum over up to 512 tokens. Unclipped, it blew up in 3 of 4 runs in our stress tests. Clamping it adds a one-sided bias that grows with the clamp strength, which matches the observation that stronger token-level clipping makes things worse.
- **Exact sequence weight `W`** is unbiased, but `E_q[W^2] = 1 + chi^2(p||q)` grows exponentially with the length `T` under any per-token mismatch. The estimator is dominated by a few sequences.

## The modification

The default is now a sequence-level, sign-aware truncated weight:

    w(y) = min(1, W(y))   if A(y) < 0
    w(y) = 1              if A(y) >= 0
    loss = -(1/N) sum_i w_i A_i sum_t log p(y_it | y_i<t),   w_i stop-gradient.

This is `ISCorrection(level="seq_sum", high=1.0, outside="clamp", sign="neg")`, the TOPR point of the grid, and it is now `DEFAULT_CORRECTION` in `losses.py`. It is active when `correction=None`. An explicit `rl.is.*` correction still overrides it.

Why it works:

1. **It fixes the over-push where it matters.** On the negative side, `min(1, W) = W` whenever `W < 1`. That is the typical regime, and there the weight is exact, which removes the unweighted over-push on the sampler's favourite completions. The only remaining bias on negatives is the cap in the set `{A<0, W>1}`: the cap removes the explosive weight in that set and makes the update conservative there.
2. **It is bounded, so the variance is bounded.** `|w| <= 1`, so the per-sequence gradient never exceeds the on-policy magnitude, whatever `T` is. The step size is not amplified by the mismatch.
3. **It is sequence-level.** The weight uses the full `ℓ(y)`, so it accounts for the future factors that `A` depends on. No per-token ratio is involved.
4. **The residual bias is one-sided and conservative.** For positive advantages we keep `w=1`. The bias `E_q[(1-W)A^+ ∇log p]` remains, but it does not amplify and it does not reduce the learning signal from successes. Capping positives too (`min(1,W)` on both signs) was slightly worse in our runs.

Formally, the only deviation of the default estimator from the target is `E_q[(W - w) A ∇log p]`, which is non-zero only on `{A>0}` (where `w=1`) and on `{A<0, W>1}` (where the cap applies). Both are one-sided, bounded, and do not feed the entropy/tail leakage of the negative side.

## Evidence (dev/mini.py, CPU, 400 steps, 2 to 3 seeds per setting)

Peak and last-eval reward, group-centered, plus-minus rewards:

| setting | none (old default) | per-token IS (no clip) | TIS (high=2 clamp) | seq-level both signs, min(1,W) | **default (TOPR-style)** |
|---|---|---|---|---|---|
| noise 0.6, seed 0 | collapsed (0.00) | collapsed (0.00) | collapsed (0.00) | 0.60 | **0.62** |
| noise 0.6, seed 1 | collapsed (0.00) | collapsed (0.00) | — | 0.39 | **0.48** |
| noise 0.3 + 200 tail tokens, seed 1 | collapsed (0.10) | 0.66 | — | 0.57 | **0.58** |
| noise 0.3 + 200 tail tokens, seed 2 | collapsed (0.00) | collapsed (0.15) | — | 0.48 | **0.59** |
| visible setting, noise 0.05, seed 0 | 0.79 | — | — | — | **0.78** |

The default collapses in none of the stress settings. Per-token IS is unstable in 3 of 4 settings. The visible mild setting is unchanged within noise. These are CPU toy runs with few seeds, so treat them as supporting evidence, not as proof.

## Caveats

- Tested only on the mini task and only with weight-noise mismatch. Staleness, int8/fp8 and KV quantization were not run.
- With LLM-length completions, `W` is tiny for most negatives under a large per-token mismatch, so the negative signal is heavily attenuated. This is exact (it is the true weight), but the variance-reduction gain may come at a cost in learning speed.
- The old unweighted behaviour is no longer reachable via `rl.is.level=null`, because `None` now selects the default. This is intentional.

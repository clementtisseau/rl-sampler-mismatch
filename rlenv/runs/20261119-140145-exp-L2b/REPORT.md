# Sampler mismatch in REINFORCE: a reward-independent drift toward the sampler

## 1. What is wrong with the current loss

Take one completion token x_t drawn from the sampler q(.|s), given prefix s.
The trainer's next-token distribution is p(.|s) = softmax(z), with logits z.
With the default arguments (no correction) the loss contributes, per token,

    L_t = -A * log p(x_t | s),

where A is the advantage of the sequence (group-centred, so mean zero over a group).

Gradient with respect to the logits, using grad_z log p(x) = e_x - p:

    grad_z L_t = -A * (e_{x_t} - p).

Expected over the token drawn from the sampler, x_t ~ q(.|s), with A held fixed at this position:

    E_q[grad_z L_t] = -A * (q - p).

**On-policy (q = p).** The expectation is -A (p - p) = 0. A constant advantage contributes nothing to
the expected update. This is the baseline-invariance of REINFORCE, and it is why the estimator is unbiased.

**Mismatched (q != p).** The expectation is -A (q - p), which is nonzero. Its effect on the logits is

    Delta z  proportional to  A * (q - p)   (gradient descent on L).

For A > 0 the trainer is pushed toward the sampler's distribution q, and for A < 0 it is pushed away from q,
whatever the reward signal says. This is a drift that does not depend on reward at all. It is first order in
the mismatch (q - p), and it does not cancel across prefixes, because each prefix has its own nonzero A and its own
(q - p). The sampler is refreshed from the trainer at every step, so the drift compounds: the trainer absorbs the
sampler's perturbation (noise, quantization, staleness) and the next sampler starts from a further-shifted point.
This is the systematic, non-zero-mean bias the task describes.

The same drift appears through the sampler's entropy and tail. When the sampler puts more mass on tail tokens than
the trainer does (q > p on the tail), positive-advantage steps push the trainer to put more mass there, and that
mass is then sampled, rewarded less, and pushed away again. This is why the failure looks like a collapse
rather than noise.

**Why importance sampling only slows it down.** Per-token IS, w = p/q with w detached, gives
E_q[w grad log p] = sum_x p(x) grad log p(x) = grad sum_x p(x) = 0, so it removes this bias exactly. The drift
returns when w is clipped or otherwise truncated (TIS, PPO, DAPO, the band of DPPO). Clipping makes the
effective weight w' = min(w, c) with E_q[w' grad log p] != 0, and the residual bias grows with the clipped mass.
That is consistent with stronger clipping making things worse. Uncapped weights have heavy tails for
tokens with q much smaller than p, so they trade the bias for variance.

## 2. The modification: a sampler-centred control variate

The batch already carries the sampler's top-K logprobs per position (K = 20, `sampler_topk_ids`,
`sampler_topk_logprobs`). Let q̂(y|s) be the sampler's top-K distribution renormalised over the K tokens:

    q̂(y|s) = softmax_K(sampler_topk_logprobs)_y,   with q̂ detached.

Define the baseline (a function of the prefix only)

    b(s) = sum_y q̂(y|s) log p(y|s),

and use the per-token loss

    L_t = -A * ( log p(x_t | s) - b(s) ).

Because b(s) does not depend on x_t, this is a control variate for the score term. Its gradient is

    grad_z b = sum_y q̂(y) (e_y - p) = q̂ - p.

**Expectation under the sampler.** With x_t ~ q and A fixed at this position:

    E_q[grad_z L_t] = -A * ( E_q[e_x - p] - (q̂ - p) ) = -A * ( (q - p) - (q̂ - p) ) = -A * (q - q̂).

The trainer-dependent term p cancels exactly. What remains is the difference between the full sampler
distribution and its top-K renormalised version, which is the sampler's tail mass beyond the top K. It is zero
whenever the sampler's support is inside the top K (for K >= V it is exactly zero for every q, p), and it is
bounded by the tail mass of q, which is small for a top-20 cut of a language model distribution.

**On-policy.** With q = p, the baseline has zero mean, so the estimator stays unbiased and simply has lower variance
than the plain score.

So the estimator's expected update no longer contains the term A(q - p) that drove the drift, and the on-policy
behaviour is preserved. The modification needs no ratio, no clipping, and no hyperparameter, and it is active
with the default arguments.

**Scope.** The baseline removes the reward-independent part of the bias, which is the part that does not vanish
when all advantages in a prefix are equal, and which compounds. It does not remove the part that comes from the
advantage depending on the sampled token itself. Writing A(x) for that dependence, the expected update is
Σ_x q(x) A(x) (e_x - q̂) instead of Σ_x p(x) A(x) (e_x - p). The difference is the usual off-policy error in
the reward-driven signal. It is bounded by the advantage times the mismatch, and it has no systematic direction
toward q. The task's hint restricts the analysis to a fixed advantage, so this residual is outside that analysis.

**Correction path.** When a correction is given, the baseline is not added. With w = p/q the score term is
already zero-mean under q, and adding b(s) would subtract an expectation that is not compensated by w. The
clipped corrections are left unchanged.

## 3. Checks

I ran the mini setup in `dev/mini.py`, which has the same loss interface and the same sampler mismatch (weights
perturbed by relative Gaussian noise, refreshed each step). Each run is 400 steps, lr 0.2, plus-minus rewards.

- On-policy (noise 0), seed 0: the original and the modified loss both reach about 0.90 at the end. The
  behaviour matches, as the derivation predicts.
- Noise 0.6, 3 seeds: the original loss collapses to reward 0 in all 3 seeds (peak 0.25 / 0.23 / 0.13,
  final 0.00). The modified loss stays stable (last-20 mean reward 0.40 / 0.35 / 0.29, eval 0.62 / 0.48 / 0.57).
- Default mini setting (noise 0.3, 600 steps): eval reward 0.66 for the original, 0.84 for the modified loss.

## Configuration

No `rl.is.*` changes are needed. The modification is active with `correction=None` by default.

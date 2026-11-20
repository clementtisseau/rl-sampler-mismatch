# Sampler-mismatch bias in REINFORCE, and a sampler-centered score

## 1. What is wrong with the current loss

With `correction=None` the loss is plain REINFORCE on sampler rollouts:

    L = -(1/N) sum_i A_i sum_t log p(x_it | s_it),

where x_it ~ q(. | s_it) is drawn from the sampler (q), while the gradient is taken on the trainer (p).
Write z for the logits at one position, p = softmax(z), and use grad_z log p(x) = e_x - p.

**On-policy (q = p).** For a fixed state and a fixed advantage A, the expected score is zero:

    E_p[grad_z log p(x)] = sum_x p(x)(e_x - p) = p - p = 0.

So a constant advantage (a baseline offset) adds only zero-mean noise. The expected update is the true policy gradient.

**Mismatched (q != p).** The same expectation under the sampler is

    E_q[grad_z log p(x)] = sum_x q(x)(e_x - p) = q - p  != 0.

For a sample with advantage A the loss therefore has expected update

    E_q[A grad log p] = E_p[A grad log p] + sum_x (q(x) - p(x)) A(x) (e_x - p),

and when A does not depend on the sampled token,

    E_q[A grad log p] = A (q - p)     (a systematic drift in logit space).

Three consequences follow.

1. The drift is not zero-mean across steps. The sampler is refreshed from the trainer, and the perturbation (weight noise, quantization, staleness) is a fixed bias direction, so q - p has a consistent sign. The trainer moves by lr * sum_i A_i (q - p) every step, and the next sampler starts from the moved trainer. That is the compounding the task describes.
2. The drift scales with the advantage offset. The loss is a weighted cross-entropy toward q: for A > 0 it does maximum likelihood on the sampler's distribution, and for A < 0 it pushes the trainer away from q. In the Countdown-style setting most unsolved rollouts have A < 0 (for example, plus_minus gives A = -1 to every unsolved rollout). The trainer is then pushed away from whatever the sampler over-produces (noise-inflated tails), and the update shrinks those tails.
3. Group centering does not remove it. Group centering cancels the offset only among rollouts that share a prefix (the prompt, and the first answer token). At later positions the subset of rollouts with a given prefix has nonzero advantage sum, so the offset bias A_bar (q - p) remains.

**Why importance weights and clipping only slow it.** The per-token IS weight w = p/q gives E_q[w grad log p] = sum_x p(x) grad log p(x) = 0 exactly, so naive IS removes the offset bias, at the cost of heavy variance wherever q << p. Clipped weights (TIS, MIS, PPO, DAPO) do not. With a clip c on the upper side,

    E_q[min(p/q, c) grad log p] = sum_x min(p, c q)(e_x - p) = -sum_x (p - c q)_+ (e_x - p),

which is nonzero whenever some token has p > c q. The bias grows as the clip tightens, which explains why stronger token-level clipping can make things worse.

## 2. The fix: subtract the sampler-expected score

Replace the sampled-token score by a centered score. For each position let

    B_t = E_q[grad log p] = sum_x q(x|s_t) grad log p(x|s_t),

and use the per-token score g_t = grad log p(x_t) - B_t. The loss becomes

    L = -(1/N) sum_i A_i sum_t [log p(x_it) - B_t] (no IS weight).

**Unbiasedness.** For a fixed state and advantage that does not depend on the next token,

    E_q[g_t] = E_q[grad log p] - B_t = B_t - B_t = 0 = E_p[grad log p].

The corrected expected update is therefore the on-policy update, for any q. The offset term A_bar (q - p) that caused the drift is removed exactly. Each term is a control variate: it has zero mean under the sampler, so it adds no bias.

**It vanishes on-policy.** B_t = sum_x p(x)(e_x - p) = 0 when q = p, so the estimator reduces to plain REINFORCE with no added variance.

**Residual bias.** If A depends on the next token, the expected update differs from the on-policy one by

    R = sum_x (q(x) - p(x)) (A(x) - A_bar_q) (e_x - p),   A_bar_q = sum_y q(y) A(y).

This is the advantage variation across alternatives times the mismatch. The offset part A_bar_q (q - p) has been removed, and what remains is only the part that token-level IS would also correct.

**Computable from the sampler's top-K.** Inference engines return the top-K logprobs at each position, and `train_rl.py` passes them as `sampler_topk_ids` and `sampler_topk_logprobs` (K = 20). Let Q_K = sum_{x in K} q(x), P_K = sum_{x in K} p(x), and rho = (1 - Q_K)/(1 - P_K). The tail of p satisfies sum_{tail} p grad log p = -sum_{x in K} p grad log p, because the full sum is zero. Approximating the tail of q by rho times the tail of p gives

    B_hat_t = sum_{x in K} q(x) grad log p(x) - rho * sum_{x in K} p(x) grad log p(x).

This is exact when K covers the vocabulary. It satisfies B_hat = 0 when q = p, since then rho = 1. In code, B_hat is the gradient of the surrogate sum_K sg(q) log p - rho * sum_K sg(p) log p, with rho and the probabilities stop-gradiented. The only approximation is the shape of the tail of q, and that enters only through rho.

## 3. Why this matches the observed behaviour

The mismatch enters only through q, so the bias A_bar (q - p) has the same form for every mismatch source: weight noise, int8/fp8 weights, quantized KV cache, and staleness (q = an older p). The correction covers all of them with the same formula, and it does not depend on how the mismatch was produced.

In the CPU proxy (`dev/mini.py`, with 200 extra vocabulary tokens so that tails matter) the uncorrected loss collapsed to zero reward (last-50 mean 0.00 on seeds 2 and 3 at noise 0.3 and 0.6, plus_minus rewards). The corrected loss kept reward at 0.74 and 0.84 at noise 0.3, and 0.30 and 0.40 at noise 0.6. The collapse came with the trainer's tail mass going to zero, which is the drift described in §1. On-policy (noise 0) the corrected and uncorrected runs agree to floating-point precision for the first ten steps and stay close afterwards (last-50 reward 0.917 for both), as the derivation predicts. Under group_centered rewards the uncorrected loss degrades less, and the correction gives a small gain (for example 0.24 to 0.27 and 0.31 to 0.32 at noise 0.6). Token-level IS did worse than both under group centering (about 0.55 at noise 0.3), which fits its variance cost. At noise 0.9 both variants collapse, so this fix is not a cure for arbitrarily large mismatch.

## Configuration

No configuration changes are needed. The correction is active with the trainer defaults (`rl.is.level=null`, which passes `correction=None`) and with `loss_fn_rl`'s default arguments. Without `sampler_topk_*` in the batch, the loss falls back to plain REINFORCE.

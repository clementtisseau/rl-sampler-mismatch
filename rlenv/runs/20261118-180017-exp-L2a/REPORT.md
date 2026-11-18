# Sampler-mismatch-robust REINFORCE: a sampler-centered score

## What is wrong with the current loss

Notation. Trainer p_t(v) = p_θ(v | h_t) is the next-token distribution at position t (h_t the prefix), and q_t(v) is the sampler's distribution at the same prefix. Rollouts are drawn from q. The loss in `losses.py` with `correction=None` has w = 1, so its gradient is

    g = -(1/N) Σ_i A_i Σ_t ∇_θ log p_t(x_it),   x_it ~ q_t(· | h_it),

where A_i is the group-centred advantage of rollout i. Taking the expectation over the sampler gives

    E_q[ ∇ log p_t(x_t) | h_t ] = Σ_v q_t(v) ∇ log p_t(v) = ∇_θ β_t,   β_t(θ) := Σ_v q_t(v) log p_t(v).

**On-policy (q = p), the baseline is free.** Here Σ_v p_t(v) ∇log p_t(v) = Σ_v ∇p_t(v) = ∇ Σ_v p_t(v) = 0 identically. So E_p[(A - b)∇log p] = E_p[A∇log p] for any constant b. The group-centring is then only variance reduction, and the expected update is the true policy gradient ∇E_p[R].

**Mismatched (q ≠ p), the baseline is not free.** The term Σ_i A_i ∇β_t(h_it) survives. Its sign depends on the sign of each advantage:

- A > 0 (solved rollouts): the update performs ∇ β_t, which is gradient ascent on Σ_v q log p, i.e. it pulls p toward q (descent on KL(q‖p) plus a constant).
- A < 0 (unsolved rollouts): the update performs -∇β_t, which pushes p away from q and moves mass toward the tokens that q rarely samples, i.e. the tail.

Group-centring removes Σ_i A_i but not this term, because β_t depends on the prefix h_it, which differs between rollouts. The bias is therefore systematic and has a fixed direction for a given sign of A, not zero-mean noise. Since the sampler is refreshed from p at each step, the drift accumulates over training. This is the observed collapse, with tail mass under q growing and then reward collapsing.

**Why IS and clipping only slow it down.** The unbiased correction for a constant advantage is w = p/q, since E_q[(p/q) ∇log p] = Σ_v p ∇log p = 0. Clipping replaces it by w' = min(p/q, c). Then E_q[w' ∇log p] = Σ_v min(p, c·q) ∇log p ≠ 0 whenever p > c·q, so clipping reintroduces a bias that grows as c decreases. Unclipped p/q has heavy tails in the sampler's low-probability tokens, which is the high-variance regime. In a local check on `dev/mini.py` (weight noise 0.4, extra tail tokens), TIS, TOPR and naive token IS all still degrade (peak reward 0.28 to 0.39, then falling toward 0 to 0.30), while the sampler-centred loss stays near 0.70.

## The modification

Keep the REINFORCE weight w(r) from `rl.is.*`, which is 1 by default. Replace the per-token score log p(x_t) by its sampler-centred version

    s_t = log p_t(x_t) - β_t,      β_t = Σ_v q_t(v) log p_t(v)   (q_t detached).

The loss is -(1/N) Σ_i A_i Σ_t w_it s_it, with the same w, mask and normalisation as before.

**Unbiasedness under q.** Conditionally on the prefix, E_{x~q}[s_t | h_t] = Σ_v q_t(v)(log p_t(v)) - β_t = 0 as a function of the sample, and the gradient is ∇ of the same expression. So for any sample-independent weight, the constant-advantage bias ∇β_t is removed exactly, for any q. Since the prefix-level terms have zero conditional mean, the sum over t of these centred scores has zero mean for any constant advantage.

**Exact on-policy reduction.** If q = p, then β_t = Σ_v p log p and ∇β_t = Σ_v p ∇log p = 0 identically (as a function of θ, not only in expectation). The gradient is therefore exactly that of the original loss on-policy, so nothing is lost when the sampler is exact.

**Top-k sampler (the practical case).** Full [B, T, V] sampler logprobs are too large for a 151k vocabulary. The sampler records the exact top-k ids and logprobs, so q is known on a set K with mass Σ_K q, and the tail mass Q_tail = 1 - Σ_K q is known. Let P_tail = 1 - Σ_K p (trainer, computed from its own logits) and ρ = Q_tail / P_tail, detached. The tail term is extrapolated from the trainer's tail:

    Σ_{v∉K} q(v) ∇log p(v)  ≈  ρ · Σ_{v∉K} p(v) ∇log p(v)  =  -ρ · Σ_{v∈K} p(v)∇log p(v),

since Σ_v p ∇log p = 0. This gives

    β_t ≈ Σ_{v∈K} q(v) log p(v) - ρ Σ_{v∈K} p(v),

whose gradient is Σ_K q∇log p - ρ Σ_K p∇log p. On-policy, ρ = 1 and the top-k gradient is exactly zero for every k, which was checked numerically. Under mismatch the tail reweighting uses the actual q tail mass, so the bias is only approximated on the tail, not dropped.

## Why it works

The loss is a REINFORCE estimator, and its expectation under the sampler is correct on-policy. The defect is a constant-advantage term that is zero on-policy and a sampler-dependent bias off-policy. A baseline is an exact control variate only if its expectation under the sampling distribution is known. The sampler's own distribution q is the only one the sampler records, so β_t, the expectation of the score under q, is the control variate that removes this term. It is computed exactly from the recorded sampler distribution, costs one extra reduction over the vocabulary already in the chunked loss, and is zero on-policy.

Unlike IS, this involves no ratio p/q, so there is no variance blow-up and no clipping bias. The residual error is the reward-correlated part of the mismatch, E_q[A ∇log p(τ)] - E_p[A ∇log p(τ)], which is linear in q - p but has no constant sign and does not accumulate as a drift toward or away from the sampler's tail.

## Configuration

sampler.vocab_logprobs=64

The loss uses whatever sampler distribution the batch carries. With `sampler.vocab_logprobs=null` (default), there is no q to centre by, and the loss reduces to the original. Top-k is used for memory: at the stated sizes a full [B, T, V] array is far too large. `full` gives the exact β (ρ = 0 path) if memory allows.

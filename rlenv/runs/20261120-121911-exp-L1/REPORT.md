# Sampler-mismatch stability: a sampler-expected baseline for the REINFORCE score

## What is wrong with the current loss

`loss_fn_rl` with `correction=None` (w = 1) is the surrogate

  L(θ) = -(1/N) Σ_i A_i Σ_t log π_θ(x_it | s_it),   x_it ~ q,

where π_θ is the trainer, q the sampler, A_i the group-centred advantage and N the number of completion tokens. Its gradient is a Monte-Carlo estimate of

  g = -E_{x~q} [ Σ_t A ∇ log π_θ(x_t | s_t) ].

The on-policy gradient is the same expression with x ~ π_θ. The two agree only if the score has zero mean under the sampling distribution, and it does not. Write the logits z of the trainer, so ∇_z log π(x|s) = e_x − π(·|s). At one position,

  E_{x~q}[ ∇_z log π(x|s) ] = Σ_x q(x|s)(e_x − π(·|s)) = q(·|s) − π(·|s) ≠ 0.

Under π itself this mean is zero. Under a mismatched sampler it is not.

For a fixed advantage A (the part that does not depend on the current token), the expected logit update at that position is therefore A(q − π). With A > 0 it pulls π toward q; with A < 0 it pushes π away from q. Group centring makes the advantages sum to zero over a group, but it does not remove this term at each position, because A is correlated with the states the sampler visits. The resulting drift E[A (q − π)] is systematic, not zero-mean noise.

It compounds because the sampler is rebuilt from the trainer every step. Here q_t = π_{θ_t} + δ with a fixed relative weight perturbation δ (or a fixed quantization or KV error). Each step moves θ toward its own noisy copy, and the next step's error has the same structure, so the errors add up instead of cancelling.

## Why importance sampling and clipping only slow it down

Per-token IS, w = π/q, makes the expected score exact for constant A: E_q[(π/q)(e_x − π)] = Σ_x π(x)(e_x − π) = 0. The price is variance: w is unbounded where q ≪ π, and it is large on the tail tokens that a noisy or quantized sampler over-produces. Token clipping at c, TIS-style, replaces w by min(w, c). Then E_q[min(π/q, c)(e_x − π)] ≠ 0 on every clipped token, so the bias comes back, and it is strongest exactly where the sampler disagrees most. This is why stronger token-level clipping can make things worse.

## The fix: subtract the sampler-expected score

Let q̂(·|s_t) be the sampler's distribution (detached). Define the sampler-expected log-probability

  b_t(θ) = Σ_x q̂(x | s_t) log π_θ(x | s_t),

with q̂ detached, so ∇b_t = Σ_x q̂ ∇log π, the q-mean of the score. The modified loss is

  L_cv(θ) = -(1/N) Σ_i A_i Σ_t w_t [ log π_θ(x_it | s_it) − b_it ],   w_t = 1 when no correction is set.

Gradient per position: ∇log π(x_t) − ∇b_t. In logit form this is e_{x_t} − q̂. Its mean under x_t ~ q̂ is zero:

  E_{x~q̂}[ e_x − q̂ ] = q̂ − q̂ = 0.

So for any advantage that is constant across the sampled token at that position, the drift term A(q̂ − π) is replaced by A(q̂ − q̂) = 0. The baseline removes exactly the q-mean of the score, which is the term that the trainer's on-policy gradient does not contain.

Checks:
- On-policy, q̂ = π: ∇b_t = Σ_x π ∇log π = 0, so L_cv has the same gradient as L. The fix is inactive when there is no mismatch.
- The baseline depends only on the state s_t and the sampler's distribution, not on the sampled token, so it does not add a term that depends on x_t. It is the standard expected-score control variate.
- The sampler already returns its top-K logprobs per position (`sampler_topk_ids`, `sampler_topk_logprobs`), so q̂ is available without new data. We renormalize q̂ over its K entries. The error from dropping the tail is O(tail mass of q̂) times the spread of log π on the tail, which is small for K = 20 on LLM vocabularies. If the top-K entries are absent from the batch, the baseline is switched off and the loss reduces to the original one.

## What remains

The baseline is exact only for advantages that do not depend on the sampled token. The reward still depends on the sampled token and on the future tokens drawn from q, and removing that part needs sequence-level importance weights, which are high-variance. So L_cv removes the systematic drift that makes the current loss collapse, but it is not a complete off-policy correction. In the CPU stand-in (`dev/mini.py`, noise 0.6 relative weight perturbation, with and without 40 extra vocabulary tokens), the current loss fell to eval reward ≈ 0 within 25 steps (2 seeds, with 40 extra tokens; also collapsed without them), while L_cv kept eval reward at about 0.4–0.6 over 600 steps (2 seeds). At noise 1.0 the baseline slows the decline but does not prevent it.

## Configuration

No configuration changes are needed. The modification is active with the default arguments of `loss_fn_rl`.

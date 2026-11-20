# Stable RL under sampler/trainer mismatch

Notation: trainer policy p_theta, sampler policy q (same prompt and prefix, per token). Tokens y_t are sampled from q. A is the advantage.

## What is wrong with the current loss

The default loss is  L = -(1/N) sum_t A * log p(y_t),  with gradient  g = (1/N) sum_t A * grad log p(y_t).
The policy gradient we want is E_{y~p}[A grad log p(y)]. Under sampling from q, per token (conditional on the prefix),

E_q[A grad log p] = E_p[A grad log p] + sum_k (q_k - p_k) A_k grad log p_k.

The second term is a systematic bias. It is not zero-mean noise, and it does not shrink with more samples. Two parts matter.

1. **Baseline term.** For any constant b, E_q[b grad log p] = b * sum_k (q_k - p_k) grad log p_k, which is nonzero (it is b times the gradient of the cross-entropy H(q, p)). On-policy it would be exactly 0 (sum_k p_k grad log p_k = grad sum_k p_k = 0), which is why baselines are free. Under mismatch they are not. With a negative mean advantage (for example plus/minus rewards, or any non-centered batch) the update is -|b| * grad CE(q, p), which pushes p *away* from q. The mismatch grows. Group-centering removes the mean only per group, and only over whole sequences. It does not cancel token by token.
2. **Unbounded negative-sample gradient on tail tokens.** The perturbed sampler puts extra mass q_k >> p_k on tokens the trainer finds unlikely. Those tokens are mostly wrong, so A < 0. The term A*grad log p_k has no saturation as p_k -> 0. The trainer keeps pushing them down, which distorts the shared weights. The sampler is the trainer plus noise that is relative to the weights, so it is refreshed every step. A distortion in the trainer becomes a larger mismatch in the sampler, which then produces more of the same gradient. This is the compounding loop. In my CPU experiments, correcting only positive-advantage samples collapsed to 0 reward, so the negative-advantage tokens are the culprit.

Hard clipping or dropping tokens by ratio (TIS, MIS, PPO-style bands) removes exactly the terms that were cancelling the bias. The cancellation is E_q[r grad log p] = sum_k grad p_k = 0 with r = p/q, and clipping breaks it. This is why stronger token clipping can make things worse.

## The modification (active by default)

1. **Token-level importance weight with exact cancellation.** w_t = min(r_t, 10), r_t = p(y_t)/q(y_t), detached. For r < 10, E_q[w grad log p] = grad sum_k p_k = 0 for a constant advantage, so the baseline term vanishes. The weight also vanishes smoothly as p_k -> 0, so tail tokens cannot be pushed down without bound (w * grad log p = grad p / q is bounded). The cap at 10 is only a safeguard against rare enormous ratios. It is a clamp, not a drop, and it is far above any in-band ratio, so little bias remains.
2. **Self-normalization.** The loss is  L = -(sum_t w_t A grad-score) / (sum_t w_t)  instead of dividing by the token count N. Plain IS is unbiased but its variance is heavy-tailed. Dividing by sum w (which has expectation N) is the self-normalized IS estimator. It has a small O(1/N) bias, and a few outlier ratios cannot dominate the step. Empirically this was the largest single stabilizer at high noise.
3. **Batch-centered advantages.** A <- A - mean(A) over the batch. Since a constant b still contributes b * sum_k (q_k - p_k) grad log p_k to the bias, and IS only cancels it up to the cap and normalization error, removing the global mean kills the leftover baseline term. It is a valid baseline for REINFORCE and a no-op for group-centered rewards.

If a `correction` is passed, it is used as given (the advantage centering stays on, and the self-normalization applies to token-level corrections). With `correction=None`, the default above is used.

## Configuration

No configuration changes are needed.

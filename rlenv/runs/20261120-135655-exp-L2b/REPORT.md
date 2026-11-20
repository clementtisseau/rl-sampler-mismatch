# Sampler-mismatch-robust REINFORCE: a sampler-centred score

## Problem with the current loss

For a completion token x_t sampled at position t with prefix h, the loss in `losses.py` (no correction) is

    L = -(1/N) sum_t A_t * log p_theta(x_t | h_t),       x_t ~ q(. | h_t),

where p_theta is the trainer, q the sampler, A_t the (group-centred) advantage of the sequence. Fix a prefix h and write
s(x) = grad_theta log p_theta(x | h), Q(x) = E[A | h, x] (the advantage the token x earns under the sampler), and
V_q = E_q[A | h] = sum_x q(x) Q(x). The expected gradient of the loss at this position, under the sampler, is

    E_q[grad L_t] = -sum_x q(x) Q(x) s(x) = -Cov_q(Q, s) - V_q * c_q,   where c_q = sum_x q(x) s(x).

**On-policy (q = p).** Two facts hold. (i) The score has zero mean, c_p = sum_x p(x) grad log p(x) = grad sum_x p(x) = 0. (ii) So the expected update is -Cov_p(Q, s), the policy gradient with baseline V_p, and V_p drops out. The update is unbiased for the reward objective.

**Mismatched (q != p).** The score no longer has zero mean under the sampler: c_q = sum_x q(x) grad log p(x) = grad(-H(q, p)) is the gradient of the cross-entropy from q to p. This is first order in the mismatch q - p. The loss therefore contributes the term -V_q c_q, a systematic drift of p towards (V_q > 0) or away from (V_q < 0) the sampler's distribution at every position, weighted by the mean advantage of the prefix. It is not zero-mean noise. Because the sampler is refreshed from the trainer at each step, p moves towards q, the next q is a fresh perturbation, and the drift accumulates over training. This is the compounding degradation. Importance sampling with w = p/q removes the term, since E_q[(p/q) Q s] = sum_x p(x) Q(x) s(x) is the on-policy quantity, but clipping (TIS, PPO-style bands) reintroduces bias, and unclipped weights have heavy tails when q is far below p on some tokens. The two failure modes are consistent with the known observations.

## Modification: subtract the sampler-expected score

Replace the sampled score s(x_t) by its centred version s(x_t) - c_q, where c_q is computed from the sampler's distribution at the same position:

    L = -(1/N) sum_t A_t * [ log p_theta(x_t | h_t) - sum_y q(y | h_t) log p_theta(y | h_t) ].

The bracket is the sampled log-likelihood minus the sampler-expected log-likelihood at that position. The baseline is a function of the sampler distribution only, so it is a valid control variate.

**Expected update.** Under x_t ~ q, with the baseline term's gradient c_q (q fixed, p at the current theta):

    E_q[grad L_t] = -sum_x q(x) Q(x) (s(x) - c_q) = -sum_x q(x) Q(x) s(x) + V_q c_q = -Cov_q(Q, s).

So the expected update is exactly the covariance between the advantage and the score under the sampler. The bias term -V_q c_q is removed exactly, whatever the mismatch and whatever the advantage level.

**On-policy check.** When q = p, the baseline is sum_y p(y) log p_theta(y), whose gradient at theta = theta_sampler is c_p = 0. The loss gradient is then exactly the usual -A s(x), so the modification does not change the on-policy estimator at all, and its expectation is -Cov_p(Q, s).

**Residual.** The remaining error is Cov_q(Q, s) - Cov_p(Q, s): a mismatch in how the advantage covaries with the score. It is first order in q - p times the variation of Q across tokens, not times the mean advantage. Removing it exactly would need importance weights p/q, which bring back the variance problem. The control variate therefore removes the part of the bias that compounds (the mean-advantage drift towards q) and leaves only the covariance distortion, which is bounded by the mismatch and does not accumulate as a drift.

**Variance.** No importance ratio appears, so the gradient cannot blow up when q is small on a sampled token. The extra term is a sum over the sampler's top-k distribution, which is already returned by the sampler (`logprobs_topk=20` in `train_rl.py`, `sampler_topk_ids` and `sampler_topk_logprobs` in the batch). The top-k truncation leaves a tail mass eps = 1 - sum_top-k q, whose contribution to the baseline is at most eps times the tail score; for an LLM vocabulary eps is small.

**Corrections.** If an `ISCorrection` is given, the same centring is applied to the weighted score, w(x) log p(x) - sum_y q(y) w(y) log p(y), with w evaluated on the top-k entries; the default (`correction=None`) is the pure form above. No `rl.is.*` configuration is needed: the modification is active with default arguments, and the sampler top-k fields are already present in the batch. If they are absent, the loss falls back to the plain sampled score.

## Configuration

No configuration changes are needed. The modification is active by default and uses the sampler top-k logprobs that `train_rl.py` already provides.

## Empirical sanity check (not required)

On `dev/mini.py` with 4 seeds, 300 steps, sampler weight noise 0.3, the plain loss reaches an eval reward of about 0.70 and the centred loss about 0.78. With weight noise 1.0 the plain loss collapses to 0 within about 25 steps, while the centred loss keeps a reduced reward (0.1 to 0.4). With an extra 30-token tail added to the vocabulary, the plain loss collapses in 2 of 4 seeds and the centred loss is stable in all 4 (eval about 0.71 to 0.81).

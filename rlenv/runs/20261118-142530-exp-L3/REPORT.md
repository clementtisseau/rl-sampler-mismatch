# Sampler mismatch in REINFORCE: the drift term

## Problem
Fix a prefix h. The token y is drawn from q(.|h); p is the trained model; s_v = grad_theta log p(v|h). The REINFORCE update A*s_y has expectation

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy, s_bar = sum_v p(v) grad log p(v) = grad sum_v p(v) = grad 1 = 0, so only the covariance (the reward signal) remains. When q != p, s_bar(h) = sum_v (q(v)-p(v)) s_v is not zero. The update then contains a drift E_q[A|h] * s_bar(h). It does not come from the reward signal. It pushes p toward q's mode structure at prefixes with positive expected advantage, and away from it where the advantage is negative. The sampler is refreshed from the trainer every step, so the bias is systematic and compounds, instead of averaging out like noise. Group-centered advantages have zero mean per prompt, but not per prefix, so the drift does not cancel.

Token-level importance weights w(p/q) rescale each term but do not remove the drift. Clipping or dropping tokens makes the weighted score no longer sum to zero over v, so it can add drift. This is why stronger clipping can hurt.

## Fix: centered score
Replace the score of the sampled token by s_y - s_bar(h). The loss uses the per-token surrogate

A * w * ( log p(y|h) - sum_v sg(q(v|h)) log p(v|h) ).

The gradient is A w (s_y - s_bar). For w = 1 and q fixed,
E_q[A (s_y - s_bar)|h] = E_q[A s_y|h] - E_q[A|h] s_bar = Cov_q(A, s_y|h).
The drift is removed exactly at every prefix, whatever q is, and the covariance (the signal) is left unchanged. Nothing about E[A|h] has to be estimated. s_bar is a control variate that depends only on h, and it has the same form that vanishes on-policy. So the loss reduces to ordinary REINFORCE when q = p, and the correction is automatically zero when there is no mismatch.

## Implementation
`losses.py` is active by default and works with correction=None. It uses the sampler's next-token distribution:
- sampling_vocab_logprobs (full): the sum over the whole vocabulary is exact.
- sampler_topk_ids / sampler_topk_logprobs (top-k): sum_{v in topk} q_v log p_v plus a tail term (1 - sum_topk q) * log(1 - sum_topk p). The tail term is there because the p-weighted mean score over the tail set is grad log P_p(tail). This is exact if q and p agree on the tail's internal shape.
- If neither is recorded the centering is skipped (it falls back to plain REINFORCE), so the config below is needed.
If an IS correction is also given, its weight w multiplies the centered score.

## Configuration
sampler.vocab_logprobs=full

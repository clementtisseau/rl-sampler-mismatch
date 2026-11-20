# Sampler-centred REINFORCE score under sampler mismatch

## 1. What is wrong with the current loss

With no correction, `loss_fn_rl` minimises

    L = -(1/N) sum_{s,t} A_s * log p_theta(x_st | x_s,<t) * mask,     w = 1,

where x ~ q is drawn from the sampler (q = p_theta with noise, quantization, KV error or staleness) and p = p_theta is the trainer. The gradient it follows is the sampled estimator

    g_q = E_{x~q}[ A * grad log p ],

whereas the policy gradient is g_p = E_{x~p}[ A * grad log p ]. Write both at one context in logit space z, where grad_z log p(x) = e_x - p, and let A_x be the expected advantage of token x in that context:

    g_q = q o A - (sum_x q_x A_x) p
    g_p = p o A - (sum_x p_x A_x) p

(o is elementwise product). The bias is

    g_q - g_p = (q - p) o A - Delta * p,      Delta = sum_x (q_x - p_x) A_x.

The second term is the problem. The score has non-zero mean under the sampler:

    E_q[ grad_z log p(x) ] = sum_x q_x (e_x - p) = q - p  != 0  whenever q != p.

So any advantage component that is constant over the actions at a context, b, adds a bias b (q - p). Under plus_minus advantages most rollouts fail, so b < 0 and the update is dz = eta * b (q - p) = eta |b| (p - q): the trainer is pushed away from its own sampler's distribution. This is a systematic direction, set by the mismatch and not by sampling noise, so it does not average out across steps. Because the sampler is refreshed from the trainer every step, the displacement accumulates. This is the compounding degradation.

Zero-mean noise would not give this. The bias is a deterministic function of q - p, and the variance of the score does not cancel it.

## 2. Why IS and clipping only slow it down

With the exact weight w = p/q, the estimator is unbiased, since E_q[(p/q) A (e_x - p)] = p o A - (sum p A) p = g_p. Clipped or truncated weights (TIS, MIS, PPO bands) are not p/q wherever they bind. Where p > c q the weight is c, so the bias is (p/q - c) q times the score. Stronger clipping makes w deviate from p/q over more tokens and so adds bias, which is consistent with stronger token-level clipping making things worse. Clipping does not remove the bias term at all; it only reshapes it.

## 3. The modification: sampler-centred score

Replace the score of the sampled token by its centred version

    s(x) = log p(x) - E_{v~q}[ log p(v) ],

where q is the sampler's distribution, estimated from the top-k sampler logprobs that the rollout already returns (k = 20 in `train_rl.py`; q~ is q renormalised over those k ids and detached). The loss uses s(x) in place of log p(x).

Since grad_z sum_v q_v log p_v = sum_v q_v (e_v - p) = q - p (sum_v q_v = 1),

    grad_z s(x) = (e_x - p) - (q - p) = e_x - q.

Three properties follow.

(a) Identity at no mismatch. If q = p, the gradient is e_x - p, the usual score. Numerically the loss gradient and reward curve equal those of the plain loss when the mismatch is zero (checked on the CPU model).

(b) Zero mean under the sampler. E_q[e_x - q] = q - q = 0. So for any constant b, E_q[b (e_x - q)] = 0: the bias b (q - p) of Section 1 is removed exactly, whatever the value of b.

(c) Residual. The expected gradient with the centred score is

    g_c = E_q[A (e_x - q)] = q o A - (sum_x q_x A_x) q.

    g_c - g_p = (q - p) o (A - abar_p) - Delta * q,      abar_p = sum_x p_x A_x.

This vanishes when A is constant, and it is first order in q - p. The baseline-driven term that compounded (Delta * p) is replaced by terms that do not contain a constant advantage. Group-centred advantages keep abar_p and Delta small, so the remaining error is the (q - p) o A part, which only an IS weight can remove.

## 4. Why centring is not combined with an IS weight

With an IS weight the plain score is already unbiased, so centring must be off. With w = p/q,

    E_q[(p/q) A (e_x - q)] = p o A - (sum_x p_x A_x) q,

which differs from g_p by (sum_x p_x A_x)(p - q). Centring then adds a bias. This was confirmed empirically: with a TIS band (level=token, high=2, clamp) the centred variant collapsed on 3 of 3 mismatch seeds, while plain TIS survived on the same 3 seeds. The code therefore applies the centred score only when `correction is None`. Under an IS correction it keeps the plain score, so behaviour with a correction is unchanged.

## 5. Practical notes

- The sampler's top-k distribution omits tail mass beyond the top k. This adds an error of order sum_{v not in top-k} q_v (e_v - q), which is small for LLM vocabularies at k = 20.
- If the batch lacks `sampler_topk_ids`, the code falls back to the plain score.
- No configuration changes are needed. The modification is active by default, with `correction=None`.

## 6. Evidence (CPU model `dev/mini.py`, not a substitute for the real trainer)

Plus_minus advantages, lr 0.2, 400 steps, 40 extra vocabulary tokens that are never correct. Plain PG collapsed (reward to 0, absorbing) on 7 of 8 runs at noise 0.3 and 8 of 8 at noise 0.5. The centred score did not collapse on any of the 8 runs (last-10 reward 0.48 to 0.82). The default `dev/mini.py` setting was stable with the centred score (no collapse, final reward about 0.84).

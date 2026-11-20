# Sampler mismatch in REINFORCE: the loss is biased toward the sampler

## 1. What the current loss does

With no correction, `loss_fn_rl` returns

    L(p) = -(1/N) sum_i A_i sum_t log p(y_it | y_i<t),

where y ~ q is the sampler, p is the trainer, A_i is the group-centered advantage, and N is the number of completion tokens. Its gradient is -(1/N) sum_i A_i sum_t grad log p(y_it), so the update is an estimate of

    g(q) = E_q[ A(y) grad log p(y) ].

The on-policy objective is g(p) = E_p[ A(y) grad log p(y) ] = grad E_p[R].

Take one position with logits z, p = softmax(z), and a sampled token y. Then grad_z log p_y = e_y - p. Write A_v for the advantage the continuation starting with token v would receive (so A_y is the one we observe). Under q:

    E_q[ A (e_y - p) ] = q ⊙ A - (q · A) p,

and on-policy:

    E_p[ A (e_y - p) ] = p ⊙ A - (p · A) p = grad_z E_p[A].

The difference, with delta = q - p, is

    bias = delta ⊙ A - (delta · A) p.

Three facts follow.

1. **On-policy (q = p) the estimator is exact.** delta = 0, so bias = 0. The score has zero mean under p, so the group baseline is free.
2. **Under mismatch the score has nonzero mean.** In logit space E_q[grad_z log p_y] = q - p, which is nonzero, so a baseline or constant offset leaks through. The bias is first order in delta and it is a reweighting of each token's advantage by q_v rather than p_v. Tokens that the sampler over-produces relative to the trainer (q_v > p_v) get too much update. Tokens the trainer likes but the sampler rarely produces get too little.
3. **The bias is systematic, not zero-mean noise.** Its expectation is nonzero for every batch with delta ≠ 0, so averaging over steps does not remove it. The sampler is then refreshed from the trainer every step, so each biased update moves the trainer in the direction that makes q and p diverge further. This is the compounding the task describes. In the dev runs the sampler-to-trainer KL `kl_samp_train` grows monotonically (about 1.4 to 3.4 nats/token at noise 0.6 over 600 steps) while the sampler's reward stays low.

## 2. Why per-token importance weights are the right fix, and where they stop

The unbiased correction is an importance weight. Sampling from q and weighting each token by w_v = p_v / q_v gives

    E_q[ (p_y / q_y) A (e_y - p) ] = sum_y p_y A_y (e_y - p) = p ⊙ A - (p · A) p = E_p[ A (e_y - p) ].

This removes the bias exactly at each position, for any advantage that depends only on the token being scored. The weight is applied with stop-gradient, so the gradient is w · grad log p, the usual form of the IS-corrected score. In code, `is_weight` computes w = exp(clip(log p - log q, ±20)), detached.

Per-token weights are not enough when the reward depends on future tokens. In the true policy gradient, the score of token t is multiplied by the reward of the whole continuation, and that continuation is drawn from q, not p. The correct weight for token t is therefore the product of ratios over the whole future,

    prod_{s >= t} p_s / q_s,

and the per-token weight p_t / q_t omits the future factors. The residual is

    E_q[ (prod_{s} w_s - w_t) grad log p_t A ],

which is nonzero and first order in delta. Including the full product is unbiased, but its variance grows with sequence length, and at these mismatch levels it is far too large to use. Clipping bounds the variance but adds bias in the clipped region; the more aggressively one clips, the more the bias moves away from the on-policy target. This explains why the known corrections (naive IS, TIS, MIS, DAPO and the rest) slow the degradation but do not remove it.

## 3. The modification

When `correction is None` (the default), `loss_fn_rl` now applies the token-level importance weight w = p/q, detached, to each token's score term. With a correction, behaviour is unchanged: a token-level correction is used as given, and sequence-level corrections keep unit token weights.

Justification, from section 2: this removes the first-order per-token bias, which is the part that does not average out, at the cost of variance. Clipping is not applied, so the bias from clipping is absent.

Empirical status (CPU `dev/mini.py`, 600 steps, group-centered rewards, not a GPU result):

- At the visible setting (noise 0.3) the baseline PG run does not collapse, and the default is indistinguishable from it on the reward and eval curves.
- At noise 0.6 the sampler-to-trainer KL at the end of training is lower with the weight (about 1.8 to 2.0 nats/token) than with plain PG (2.5 to 3.3). The trainer's clean eval reward is slightly higher. The sampler-side reward is lower (about 0.15 to 0.18 vs 0.25 at the end of training).
- With many extra vocabulary tokens (tail-heavy sampler), the weight reduces the tail's push-down in the sampler's distribution. This is the cost of the correction: it makes sampler-side learning slower.

I did not reproduce a collapse for plain PG in the CPU setup, so the stability benefit is supported by the KL drift argument above, not by a collapse-to-stable comparison.

No configuration changes are needed: the modification is active by default.

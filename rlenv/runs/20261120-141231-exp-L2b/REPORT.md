# Sampler mismatch in REINFORCE: a sampler-expected score baseline

## 1. What is wrong with the current loss

Fix a position with prefix h (all earlier tokens). Let p = p_theta(. | h) be the
trainer's next-token distribution, q = the sampler's distribution at the same
position, and let A(x) = E[A_seq | h, x_t = x] be the expected group-centered
advantage of a rollout whose next token is x. The loss in `losses.py` uses one
sampled token per position with weight 1 (correction=None):

    L = - sum_t A_seq * log p(x_t | h_t),     x_t ~ q.

Gradient with respect to the logits z at this position (softmax parameterization,
grad_z log p(x) = e_x - p):

    E_{x~q}[ A(x) (e_x - p) ] = sum_x q(x) A(x) (e_x - p).

On-policy (q = p) the target is the policy gradient of E_p[A], which is
sum_x p(x) A(x) (e_x - p). The bias appears when q != p. Write
Abar = sum_x q(x) A(x) for the q-mean advantage at this prefix. Then

    G_current = sum_x q(x) A(x)(e_x - p) = sum_x (q - p)(x) A(x)(e_x - p) + Abar (q - p),

where the identity sum_x (q-p)(x)(e_x - p) = q - p was used. The second term,
Abar (q - p), is a first-order bias. It does not vanish when the advantage is
constant over the prefix. Two facts make it systematic:

- Abar(h) is generally nonzero. Group centering makes advantages sum to zero
  over a group of whole sequences, not conditionally on each prefix. A prefix
  that already leads to failure has Abar(h) < 0, and a prefix on a good path has
  Abar(h) > 0. So the bias is present at most positions.
- For constant A, the expected update is A (q - p). Positive advantages move p
  toward q, and negative advantages move p away from q. Wherever the sampler
  over-samples a token (for example noise-induced tail mass in the sampler's
  distribution), a positive-advantage rollout pushes that token up in the
  trainer. Because the sampler is refreshed from the trainer at every step
  (and the mismatch is a fixed perturbation of the same weights), the bias is
  re-injected each step and accumulates into a drift. This matches the
  observation that the error compounds.

Clipping and token-level IS (TIS, MIS, PPO) reduce the weight of out-of-band
tokens but leave the mean term Abar (q - p) in place for in-band tokens, so the
bias is only shrunk. The exact weight p/q removes the bias in expectation, but
clipping it makes the estimator biased again (section 3), which is consistent
with tighter clipping making collapse worse.

## 2. The modification

Subtract from each token's score its expectation under the sampler at the same
prefix. Define the sampler-expected log-likelihood

    b(h) = sum_y q(y | h) log p_theta(y | h),        with q treated as a constant (stop-gradient),

and use the per-token loss

    L = - sum_t A_seq * [ log p_theta(x_t | h_t) - b(h_t) ],     x_t ~ q.

`b(h)` depends only on the prefix, not on the sampled token x_t. It is a control
variate for the score. In `losses.py` the sum over y runs over the sampler's
recorded top-20 distribution (`sampler_topk_ids`, `sampler_topk_logprobs`,
which `train_rl.py` already passes to the loss), and the baseline is not
stopped: its gradient is part of the correction.

## 3. Why it works: derivation

Gradient of b with respect to the logits, with q fixed:

    grad_z b(h) = sum_y q(y) grad_z log p(y) = sum_y q(y)(e_y - p) = q - p.

Expected update of the modified loss at this prefix, with x ~ q:

    E_q[ A(x)(e_x - p) ] - Abar * (q - p)
      = sum_x q(x) A(x)(e_x - p) - Abar (q - p)
      = sum_x q(x) (A(x) - Abar)(e_x - p).

So the mean-advantage term, which is the bias in the current loss, is removed
exactly. The surviving update depends on the advantage only through its
deviations A(x) - Abar within the prefix.

Checks:

1. On-policy, q = p. Then grad_z b = sum_y p(y)(e_y - p) = 0 at every prefix
   (sum_y grad p(y) = grad 1 = 0). The modified loss has the same gradient
   as the current loss, pointwise in the batch, not only in expectation. Nothing
   is changed when the mismatch is zero.
2. Mismatched, q != p. The modified update is
   G_new = sum_x q(x)(A(x) - Abar)(e_x - p). The target is
   T = sum_x p(x) A(x)(e_x - p) = sum_x p(x)(A(x) - Abar_p)(e_x - p). The error is
     G_new - T = sum_x (q - p)(x) (A(x) - Abar)(e_x - p),
   which is first order in |q - p| and multiplied by the within-prefix advantage
   variation, not by its mean. For the current loss the error includes the extra
   term Abar (q - p), which is first order in |q-p| times the prefix mean
   advantage.
3. Variance. grad_z b = q - p is small when the mismatch is small, and it is
   exactly zero on-policy, so the baseline adds no variance in the regime that
   matters. The baseline uses only the sampler's recorded distribution, so it
   covers every mismatch source the sampler records (weight noise, int8/fp8
   weights and activations, quantized KV cache, stale weights) without modeling
   any of them.

Why not plain importance sampling. The exact per-token IS weight w = p/q gives
E_q[w (e_x - p)] = sum_x p(x)(e_x - p) = 0, so it is also unbiased for constant
advantages. Its variance is large when p/q is large on sampled tokens, and
clipping that variance trades it for the bias described in section 1. The
baseline above removes the same first-order term with no weight and no clip.

## 4. Limits

- The sum over y uses the top-20 sampler distribution. This is exact when the
  vocabulary has at most 20 tokens (the mini setup with no extra tokens). For a
  real LLM the omitted tail adds a small error, proportional to the sampler's
  tail mass times the trainer's log-probabilities of those tokens.
- If the batch has no sampler top-k entries, the loss falls back to the
  original estimator. `train_rl.py` always passes them (logprobs_topk=20).
- When a token-level IS correction is given, the baseline is not applied: its
  combination with a per-token weight w is not zero-mean unless w is computed
  over the whole vocabulary. The default (correction=None) path is the one this
  report describes.

## 5. Empirical check (mini setup, not part of the deliverable)

Evaluation reward at step 599, two seeds per setting, original vs modified loss
(`dev/mini.py`, plus_minus reward):

| relative weight noise | lr | extra tokens | original | modified |
| --- | --- | --- | --- | --- |
| 0.3 | 0.2 | 0 | 0.66 | 0.83 |
| 0.6 | 0.2 | 0 | 0.00 (collapsed) | 0.53 to 0.60 |
| 0.3 | 0.2 | 30 | 0.25 to 0.70 | 0.80 to 0.85 |
| 0.6 | 0.4 | 30 | 0.00 (collapsed) | 0.44 (one seed) |

Configuration: none. The change is in `losses.py` only, active under the
default arguments of `loss_fn_rl`.

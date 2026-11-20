# Sampler-centred score for REINFORCE under sampler mismatch

## Setting

At a completion position, with prefix h, the trainer defines p(.|h) = softmax(z) over the
vocabulary, and the sampler draws the next token from q(.|h). Per token the loss uses the
score of the sampled token x ~ q, with advantage A (per sequence, constant over its tokens):

    L_t = -A * w * log p(x | h),          w = 1 without a correction.

The ascent direction is A * w * grad_z log p(x|h), with grad_z log p(x|h) = e_x - p.

## 1. What is wrong with the current loss

Take the expectation over the next token drawn from the sampler, with w = 1:

    E_{x~q} [ grad_z log p(x|h) ] = sum_x q(x) (e_x - p) = q - p.

So the expected ascent direction of one position is

    E_q[ A * grad_z log p ] = A * (q - p).                                  (1)

- On-policy (q = p), this is 0 for any constant advantage. A token-level term with no
  reward dependence contributes nothing in expectation; it only adds variance.
- Under mismatch (q != p), it is nonzero for every advantage, including A = 0 sequences that
  have no reward signal at all. For a token y with q(y) > p(y), the logit of y is pushed up
  by A (q(y) - p(y)), whatever the reward. For A > 0 the loss pulls p toward q; for A < 0 it
  pushes p away from q. Summed over the batch, this is equivalent to maximizing
  sum_x q(x) log p(x) weighted by A, i.e. -A * CE(q, p): the trainer is being fitted to the
  sampler's distribution rather than to the reward.

This is a deterministic drift, not zero-mean noise: its expectation (1) does not vanish over a
batch when E[A | h] != 0 at a prefix. Because the sampler is refreshed from the trainer at each
step, the mismatch is re-injected every step, and the drift accumulates. Heavy tails are the
worst case: sampler noise, int8/fp8 rounding and quantized KV make q put more mass than p on
unlikely tokens, and (1) then raises exactly those tokens.

Why the usual corrections only slow it down. Write r = p/q and let c be a clip level.

- Unclipped token IS, w = r, is unbiased at a single position:
  E_q[(p/q) (e_x - p)] = sum_x p(x)(e_x - p) = 0. Its variance grows with E_q[(p/q)^2] = 1 + chi^2(p||q),
  which is large in the tails.
- Clipping (TIS, clamp, drop, PPO bands) makes the weight q * min(r, c) = min(p, c q) in the
  expectation. The expected update becomes
      E_q[ min(r, c) (e_x - p) ] = sum_x min(p(x), c q(x)) (e_x - p),
  which is zero only when p <= c q everywhere. Otherwise it is a nonzero bias, and it grows as
  c shrinks. This is why stronger clipping can make things worse. Clipping trades variance
  for bias but does not remove the bias term in (1).

## 2. The modification

Subtract, from each token's score, its expectation under the sampler:

    score_t = grad_z [ log p(x|h) - b(h) ],     b(h) = sum_y q(y|h) log p(y|h),   q detached.

Its gradient is grad_z b = sum_y q(y)(e_y - p) = q - p, which is exactly the drift in (1).
So the expected update becomes

    E_q[ A (grad log p(x|h) - (q - p)) ] = A (q - p) - A (q - p) = 0      (constant A).     (2)

The drift vanishes for every sampler q, and no weight p/q is needed, so there is no 1/q variance
and no clipping.

General advantages. Let A_x be the advantage of choosing x at h, and Abar_q = sum_x q(x) A_x.
Then the expected update of the modified loss is

    U_q = sum_x q(x) A_x (e_x - p) - Abar_q (q - p) = sum_x q(x) (A_x - Abar_q) e_x.        (3)

The on-policy softmax policy gradient for this position is

    U_p = sum_x p(x) (A_x - Abar_p) e_x.

So, relative to the on-policy gradient, the modified loss replaces the weighting p by q and
centres the advantage under q. The unmodified loss gives

    U_orig = sum_x q(x) A_x e_x - Abar_q p,

so the difference is exactly the term Abar_q (q - p), which is the drift.

Interpretation of the residual. The residual error of the modified loss is

    U_q - U_p = sum_x (q - p)(x) (A_x - Abar_q) e_x - (Abar_q - Abar_p) p.

The first term weights the mismatch q - p by the centred advantage A_x - Abar_q, which has zero
mean under q. Tokens that are neutral for reward (A_x = Abar_q) get no update, regardless
of q. So the mismatch only changes how strongly each token's advantage is weighted. It does not
push the trainer toward the sampler's distribution. The second term is a shift of the
baseline. Both are of order |q - p| |A|, but neither is a systematic pull toward q.

Sanity check. At q = p, b(h) = -H(p|h) is a constant in x, and its gradient (with q detached)
is sum_y p(y)(e_y - p) = 0 in expectation. The loss then reduces, in expectation, to the usual
REINFORCE gradient, and the baseline only changes variance.

Estimating b. The sampler returns its top-K log-probs per position (K = 20 in train_rl.py), so
b is computed over that support: b_K(h) = sum_{y in K} q(y|h) log p(y|h). The exact drift
(q - p) is replaced by q_K - (1 - eps) p, where eps = 1 - sum_{y in K} q(y) is the top-K tail
mass. The error between the two gradients is q_tail - eps p, with L1 norm at most 2 eps. So the
bias left by the top-K truncation is bounded by twice the tail mass of the sampler (times |A|),
which is small when the sampler is close to the trainer.

## 3. Implementation

In `losses.py` (default, no flag): the per-token score is `w * (logp - baseline)` where
`baseline = sum_{y in top-k} exp(sampler_topk_logprobs) * logp_vocab[y]`, with the sampler
weights stop-gradiented. This applies with and without an `ISCorrection`, and falls back to the
plain score if the batch has no top-k fields. It adds O(K) work per token and no new config.

## Configuration

(no changes required; the modification is on by default)

## Check (dev/mini.py, CPU; not required, one toy setting)

Sampler noise 0.6, 600 steps, reward mean over the last 10 steps (peak in brackets):
plain score 0.00 (0.13-0.25) on 3 seeds; sampler-centred score 0.33-0.39 (0.45-0.53) on 3 seeds;
TIS (token, high=2, clamp) 0.13 (0.26); naive token IS 0.00 (0.27). Same loss, default arguments.

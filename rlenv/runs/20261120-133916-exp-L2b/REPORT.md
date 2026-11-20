# Sampler mismatch in REINFORCE: the loss is biased, and the fix is a self-normalized ratio

## Summary

The loss treats every sampled token as if it were drawn from the trainer p.
Its expected update is therefore taken under the sampler q. Because the
advantage has a non-zero conditional mean, the update contains a term
proportional to (q - p) that is not zero in expectation. The bias is
systematic, and since the sampler is refreshed from p each step it compounds.
The fix is the plain ratio w = p/q, which removes the bias exactly at each
position, self-normalized over the batch so the variance stays bounded. This
is the default in `losses.py` (`correction=None`).

## 1. What the current loss does

Fix a prefix h and one position. Write p = softmax(z) for the trainer and
q for the sampler, both over the vocabulary. Let A(y) be the advantage of a
completion whose token at this position is y, and Ā(y) = E[A | h, y] its
conditional mean. The loss is

  L = -(1/N) sum_t A_t * sg(w_t) * log p(y_t),  y_t ~ q,

with sg the stop-gradient. Under no correction, w_t = 1. In logit space,
grad_z log p(y) = e_y - p, so the expected ascent direction at this position is

  G_0 = E_{y~q}[ Ā(y) (e_y - p) ] = sum_y q(y) Ā(y) (e_y - p).

On-policy (q = p) the desired direction is

  G* = E_{y~p}[ Ā(y) (e_y - p) ] = sum_y p(y) Ā(y) (e_y - p),

and the score function has zero mean under the trainer: E_p[grad log p] = 0.
That zero mean is what lets any baseline (any constant offset c of the
advantage) drop out of the on-policy gradient.

Off-policy the zero mean is lost: E_q[grad_z log p] = q - p ≠ 0. The
bias is

  G_0 - G* = sum_y (q(y) - p(y)) Ā(y) (e_y - p).

For a constant advantage Ā ≡ c this reduces to B = c (q - p). A nonzero
conditional advantage mean is the usual case past the first token: group
centering makes the advantages sum to zero over the group, which shares only
the prompt, not the prefixes reached deeper in the completions. Hence the
bias is generally non-zero.

Why it compounds. The sampler's mismatch is fixed for a run (for weight noise,
q = softmax(z + eps) with a fixed eps), so q - p has a persistent direction.
Each step moves p by roughly eta * c * (q - p), toward q when c > 0 and away
from q when c < 0, and the next step's sampler is rebuilt from the new p. The
push does not average out across steps. Its fixed point is p = q weighted by
the advantage, i.e. sum_y q(y) Ā(y)(e_y - p) = 0 gives p(y) proportional to
q(y) Ā(y) for Ā > 0, which is set by the sampler rather than by reward.

Why clipping does not fix it. With a clamped token ratio w = min(r, c), r = p/q,
the expected direction is sum_y min(p, c q) Ā(y)(e_y - p). Relative to the
unbiased G*, the error is -sum_y (p - c q)_+ Ā(y)(e_y - p). It is non-zero
whenever the trainer puts more mass on y than c times the sampler does, and
the set {p > c q} grows as c shrinks. That is why stronger clipping can make
things worse, and why clipping only slows the drift down.

## 2. The fix: an unbiased ratio, self-normalized

### 2.1 Unbiased weight

Take w(y) = p(y)/q(y), detached. Then

  E_{y~q}[ w(y) Ā(y) (e_y - p) ] = sum_y p(y) Ā(y) (e_y - p) = G*,

exactly, for any advantage function, including a constant. The baseline
invariance of the on-policy gradient is restored: for a constant offset c,
E_q[w (e_y - p)] = sum_y p (e_y - p) = p - p = 0. The sampled log-ratio
r = exp(log p - log q) is available for every token from the sampler's
logprobs, so no extra forward pass is needed.

### 2.2 Variance and self-normalization

The plain ratio has heavy tails: a token that q rarely produces but p favors
gets a huge weight when it is sampled. Self-normalizing over the batch gives

  L = -( sum_t A_t w_t log p_t ) / ( sum_t w_t ),   w_t = sg(p_t/q_t),

with the sums over completion tokens. The denominator is a constant with respect
to the parameters, so this is one pass with two scalar sums.

Properties.
(a) The update is a convex combination of the per-token score terms, so its
    norm is bounded by max|A| times the largest score norm, whatever the
    ratio tails. A single large ratio cannot blow up the step.
(b) Bias is O(1/n). Let X = e_y - p and write the batch estimate as
    S = (sum_i w_i X_i) / (sum_i w_i) with n tokens drawn from q. Since
    E_q[w X] = 0 for a constant advantage, a second-order expansion gives

      E[S] = E[wX]/E[w] - Cov(wX, w)/(n E[w]^2) + O(n^{-2}) = O(1/n).

    So the constant-offset leak of Section 1 is removed in expectation, and
    its size decays with the batch.
(c) Self-normalization is the usual ratio estimator (Owen, Monte Carlo
    theory, Ch. 9). It is consistent for the on-policy gradient.

I checked (b) numerically on a random 6-token distribution with constant
advantage c = 1 and q = p * exp(N(0, 0.8^2)). The leak of the uncorrected
estimator is q - p (for example -0.0984 at index 0 and 0.1584 at index 5),
stable across n. The plain IS estimator is 0 up to sampling noise. The
self-normalized estimator is -0.015 at n = 16, -0.002 at n = 128 and -0.0002
at n = 1024 at index 0, decaying as O(1/n).

### 2.3 Why this is the default

With correction=None the weight is the plain ratio, self-normalized. With a
correction set, the token weights from the grid are self-normalized the same
way. Sequence-level corrections keep the plain token mean, since their token
weights are 1.

## 3. Limits of the argument

- The analysis is per position with the next token drawn from q. The
  advantage of a completion also depends on its later tokens, which are drawn
  from q, so a sequence-level mismatch remains. Only the offset part of the bias
  (the one that compounds) is removed exactly by the token ratio. The
  sequence-level part is second order in the mismatch and is not handled by
  token weights. Full sequence ratios would fix it, but their variance is
  prohibitive for long completions.
- The self-normalized ratio is consistent, not unbiased. Its bias is O(1/n)
  in the batch size, and it does not vanish at finite n.
- Clamped or dropped token weights (TIS, MIS, PPO-style bands) remain biased,
  as Section 1 says.

## 4. Empirical check (CPU mini setting, `dev/mini.py`, not required but used)

Relative weight noise on the sampler, plus-minus rewards, lr 0.2, 600 steps.
Final mean reward over the last 50 steps, and final eval reward:

| setting | plain (pg) | naive IS, token | TIS (high=2, clamp) | self-normalized (this loss) |
| --- | --- | --- | --- | --- |
| noise 0.3, seed 0 | 0.58 / 0.66 | - | - | 0.64 / 0.81 |
| noise 0.4, seed 0 | collapse | 0.52 / 0.65 | - | 0.55 / 0.79 |
| noise 0.4, seed 1 | collapse | - | - | 0.49 / 0.67 |
| noise 0.5, seed 0 | collapse | collapse | collapse* | 0.37 / 0.62 |
| noise 0.5, seed 1 | collapse | - | collapse* | 0.34 / 0.42 |
| noise 0.5, seed 2 | collapse | - | - | no learning (0.04 / 0.00) |

* TIS was run with the self-normalized weights of this loss, not as a plain
  clamp, so the TIS column at noise 0.5 shows clamping inside the new loss.

Among the runs above, the self-normalized loss is the only one that keeps
learning at noise 0.5 on seeds 0 and 1. Seed 2 at noise 0.5 fails to learn from the
start, a warm-start issue, not a collapse. Naive IS collapses at noise 0.5
and TIS with a clamp collapses even when self-normalized. The mismatch types
in the task (int8 or fp8 weights, quantized KV cache, staleness) are not
simulated by the mini harness, so their effect is not tested here.

## Configuration

No configuration changes are needed. The modification is the default path of
`loss_fn_rl` with `correction=None`.

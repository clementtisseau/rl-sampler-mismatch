# Reference: score centering

Ground truth for the judge. Source: Marek & Ryabinin, *Score Centering
Stabilizes Off-policy Reinforcement Learning* (arXiv:2609.20807), and the
reference implementation in `losses.py` (`loss_fn_rl`, `score_center=True`).

## Setting

RL for LLMs samples completions from a sampler q (the inference engine) and
updates a trainer p. Under training-inference mismatch (TIM) q != p: the
sampler runs quantized, perturbed, or stale weights, or simply different
kernels. Training with REINFORCE on these samples becomes unstable and often
collapses. Before this work, the known facts were: the mismatch exists, it
hurts, it acts as a systematic bias rather than as noise (Dong et al., 2026),
and importance-sampling corrections (TIS, MIS/IcePop, PPO-style clipping,
sequence-level ratios) slow the problem down without removing it.

## The contribution

### 1. Decomposition of the expected per-token update

At one position, with prefix h, sampled token y ~ q(.|h), advantage A of the
whole completion, and score s_v = grad_theta log p(v|h), REINFORCE updates
along A * s_y. Its expectation over the sampled token decomposes as

    E_q[A s_y | h] = E_q[A | h] * s_bar(h)  +  Cov_q(A, s_y | h)
                     \_______ drift ______/    \_____ signal ____/

    s_bar(h) = sum_v q(v|h) grad log p(v|h)     (expected score under the sampler)

On-policy (q = p), s_bar = sum_v p_v grad log p_v = grad sum_v p_v = 0, so the
drift vanishes. Under mismatch s_bar != 0: every update contains a term
proportional to the mean advantage at that prefix, pointing along s_bar,
which pulls the trainer toward the sampler (a distillation toward q). The
sampler is refreshed from the trainer, so the error compounds through the
sync loop. This is the mechanism behind instability: a bias, not variance.

Key point: the drift lives at each prefix. Its size is E_q[A | h], the
advantage expected *given that prefix*, which is non-zero at individual
prefixes even when the advantage has zero mean over the batch.

### 2. The correction: center the score, not the reward

Subtract the sampler-expected score from each token's score:

    s~_y = s_y - s_bar(h)

Then E_q[A s~_y | h] = Cov_q(A, s_y | h): the drift term is gone, the signal
is untouched. Properties:
- on-policy the correction is exactly zero (s_bar = 0);
- it is additive and deterministic given the prefix: no importance ratios,
  no clipping, no masking, so no exploding weights;
- it is multiplied by the same advantage A as the token's score;
- what remains approximate: the covariance is measured under q instead of p.

### 3. A tractable loss

s_bar needs the sampler's distribution at every position and a gradient for
every vocabulary entry. Both are avoided with a scalar surrogate whose
gradient is the corrected score, using stop-gradient weights:

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) )

With a large vocabulary the sampler only returns its top-k head H (k = 128 in
the paper; k = 32 is enough in their tests). The tail of q is modeled as
proportional to p with mass ratio rho = q_tail / p_tail. Since
sum_v p_v grad log p_v = 0, the tail term contributes no gradient and only the
head terms remain:

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v )

The sampled token's own logprob is kept even when y_t is outside the head.

### 4. Composition and positioning

- Composes with token-level IS corrections: with weight w(p/q) on the sampled
  token, center under the effective sampling mass q * w instead of q (and
  rho * w(1/rho) for the tail). SC + TIS/MIS performs best under stale
  weights.
- Group centering (GRPO) subtracts a baseline from the *reward*. That cancels
  the drift on average over the batch, not at each prefix, so it shrinks the
  drift without removing it.
- Classical baselines subtract an expected *reward* to reduce variance.
  Score centering subtracts an expected *score* to remove a bias. The paper
  frames it as a bias correction, not a control variate.

## Evidence

Qwen3-0.6B on Countdown under synthetic weight noise, int8 sampler or stale
weights; Qwen3-30B-A3B on math under increasing sampler quantization. Under
the largest noise only SC (alone or with TIS/MIS) trains stably; DPPO and TIS
collapse.

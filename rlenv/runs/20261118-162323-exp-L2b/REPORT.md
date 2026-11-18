# Sampler-centred REINFORCE under sampler/trainer mismatch

## 1. What is wrong with the current loss

Notation. At a completion position with prefix s, the trainer is p(.|s) and
the sampler is q(.|s). Logits are z, with p = softmax(z), so
grad_z log p(x) = e_x - p, where e_x is the one-hot vector of token x.
A is the advantage of the sequence (group-centred). In `losses.py` the
per-token term is `-A * w * log p(x)`, with x ~ q. With `correction=None`,
w = 1.

The expected gradient of this term at a fixed prefix, averaging over the
sampled token x ~ q, is

    E_{x~q}[ A * grad_z log p(x) ] = A * sum_x q(x) (e_x - p) = A * (q - p).

On-policy (q = p) the same expression is A * sum_x p(x)(e_x - p) = 0.
So the sampled-token score has zero mean only when the sampler is exact.
Under mismatch it has mean A(q - p), a systematic push of the logits
toward the sampler's distribution (for A > 0) or away from it (for A < 0).

This is not zero-mean noise. The direction q - p is fixed by the weight
perturbation. The sampler is refreshed from the trainer every step, so the
perturbation persists and the drift accumulates. Each step moves p toward
q, the next sampler is p plus the same kind of perturbation, and so the
error compounds. Group centring does not remove it, because E[A | s] for a
given prefix need not be zero.

## 2. Why importance-sampling corrections only slow it down

The naive per-token weight w = p(x)/q(x) is exact for a constant advantage:

    E_{x~q}[ (p/q)(x) (e_x - p) ] = sum_x p(x)(e_x - p) = 0.

So it removes the mean bias, but with high variance. The weight is
unbounded where q is small relative to p, and the estimator uses a single
sample of the score. Clipped weights (TIS, MIS, PPO, DAPO) replace w by
min(p/q, c) or a band. Let c be the clip level and m(x) = min(p/q, c). Then

    E_{x~q}[ m(x) (e_x - p) ] = sum_x min(p(x), c q(x)) (e_x - p),

which is nonzero whenever min(p, cq) is not proportional to p. The bias is
the mass of p(x) - c q(x) on the clipped tokens. A stronger clip (smaller c)
makes this bias larger, which matches the observation that tighter clipping
can make things worse. Per-token IS also does not correct the future
tokens: the advantage A depends on continuations drawn from q, and only the
current token is reweighted.

## 3. The fix: subtract the sampler's expected score

The bias comes from the term A * E_q[grad log p]. Subtract exactly that
term. Replace the per-token score by a control-variate version that is
centred under q:

    grad_z [ log p(x) - sum_y q(y) log p(y) ] = (e_x - p) - (q - p) = e_x - q.

The baseline sum_y q(y) log p(y) depends on the trainer's parameters, so
its gradient is included; this is the q - p term. The loss term is

    L_t = -A * [ log p(x_t) - sum_y q(y|s_t) log p(y|s_t) ],   x_t ~ q.

**Unbiasedness.** For any q and p, and for constant A at the position,

    E_{x~q}[ e_x - q ] = sum_x q(x)(e_x - q) = q - q = 0.

So the expected update at every position is zero, as it is on-policy. The
mismatch drops out of the expectation entirely, and nothing compounds.

**On-policy consistency.** At q = p the term is e_x - p, the standard
REINFORCE score, and the baseline vanishes in expectation. With advantages
that depend on x, the on-policy expectation is unchanged, since q = p.

**Variance.** No importance weight appears. The factor e_x - q has norm at
most sqrt(2), so the update per token is bounded, unlike p/q.

**Residual error.** For x-dependent advantages Â(s, x), the new estimator has
expectation sum_x q(x) Â(x)(e_x - q), versus the on-policy target
sum_x p(x) Â(x)(e_x - p). The difference is O(|q - p| |Â|), not the constant
drift A(q - p), and it averages out over the group. The future-token
mismatch (Q under q versus Q under p) is not removed by this loss.

**Sampler information.** The loss needs q(.|s) at each position. Two
options are supported, both recorded by the sampler:

- full vocabulary: `sampler.vocab_logprobs=full`, giving `sampling_vocab_logprobs`,
  so the baseline is exact;
- top-k head: `sampler.vocab_logprobs=k`, giving `sampler_topk_ids` and
  `sampler_topk_logprobs`, so the baseline drops the tail mass of q.

If neither is present, the loss falls back to the plain sampled-token score
(the old behaviour), since the sampler's full distribution is then unknown.

## 4. Check

Numerical check with autograd (random z, q over 7 tokens, float64):
the expected gradient under q of the old term equals A(q - p) exactly, and
the expected gradient of the new term is zero to machine precision. On-policy
the old term has zero mean.

In the CPU harness (`dev/mini.py`, noise 0.3, 600 steps) the sampled-token
REINFORCE baseline collapses to near-zero reward on a large vocabulary (2
seeds), while the full-vocabulary version stays near 0.82 (2 seeds). On the
small vocabulary, naive per-token IS and TIS reach about 0.6 reward (3 seeds),
versus about 0.84 for the full-vocabulary version. These runs are a sanity check, not part of the argument.

## Configuration

sampler.vocab_logprobs=full

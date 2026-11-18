# Sampler-centered REINFORCE: a loss that stays stable under sampler/trainer mismatch

## Summary

The REINFORCE loss implicitly relies on an identity that only holds on-policy:
the expected score is zero, `E_{y~p}[grad log p(y)] = 0`. That identity is what
makes the update independent of any part of the advantage that does not depend
on the sampled token. When rollouts come from a sampler q != p, the expected
score is `E_{y~q}[grad log p(y)] = -grad KL(q || p) != 0`. Every advantage then
adds a reward-independent term `A * E_q[grad log p]` to the update. That term
has a fixed sign (it follows the sign of A), it points along the sampler's
error, and it comes back at every step because the sampler is rebuilt from the
trainer each time. Over training these terms add up instead of averaging out.

The fix is to subtract the score's expectation under the sampler at every
token, which centers the score under q:

    loss = -(1/N) sum_i A_i sum_t [ log p(y_t) - sum_v sg(q_t(v)) log p_t(v) ]

In logit space the per-token gradient changes from `A (e_y - p)` to
`A (e_y - q)`. That gradient has exactly zero mean under y ~ q, for any q. When
q = p it equals plain REINFORCE's gradient, so nothing changes on-policy.

## 1. What is wrong with the current loss

Notation: prefix s_t, trainer next-token distribution p = p_theta(.|s_t) with
logits z, sampler distribution q(.|s_t), sampled token y_t ~ q, and an advantage
A that may depend on the whole completion. The current loss with
`correction=None` is

    L = -(1/N) sum_i A_i sum_t log p(y_{i,t} | s_{i,t}),     g = -grad L.

**On-policy (q = p).** For any c that does not depend on y_t given s_t,

    E_{y_t~p}[c * grad log p(y_t)] = c * sum_y p(y) grad log p(y) = c * grad sum_y p(y) = 0.        (1)

So E[g] = grad E_p[R]: a baseline, a constant reward offset, the group mean,
the length of the sequence and the visited states all drop out of the
expectation. Only the part of A that is correlated with the sampled token
drives learning. This is the policy-gradient theorem, and (1) is the property
that makes it work.

**Mismatched (q != p).** The same expectation is

    E_{y~q}[grad log p(y)] = sum_y q(y) grad log p(y) = grad_theta sum_y q(y) log p_theta(y)
                           = -grad_theta KL(q || p_theta)                                (2)

where q is held fixed. In logit space, `grad_z log p(y) = e_y - p`, so (2) is
`q - p`, which is not zero. Decompose the advantage of a token into
`A = b(s_t) + (A - b(s_t))` with b(s_t) = E_q[A | s_t]. The expected update is then

    E_q[g] = E_q[(A - b) grad log p]              (the reward signal)
           + E_q[ b(s_t) * (-grad KL(q||p)(s_t)) ] (the bias)                        (3)

The second term is not part of any objective the trainer is meant to optimize:

* With b > 0 it is a step down KL(q||p), pulling p toward the sampler. With
  b < 0 it is a step up KL(q||p), pushing p away from the sampler. Pushing p
  away from q also raises the trainer's mass on whatever the sampler
  under-weights and lowers it on whatever the sampler over-weights. That is
  why negative samples are the dangerous ones.
* The term has a fixed direction set by the sampler's error, for example
  `delta` for weight noise theta_q = theta + delta, the quantization error of
  int8/fp8 or a quantized KV cache, or theta_{t-k} - theta_t for stale
  weights. It is not zero-mean noise. Since q is rebuilt from the updated
  trainer at every step (q_{t+1} = f(theta_{t+1})), the sampler error reappears
  with the same sign, so the drift does not cancel from one step to the next.
  To first order, theta_T - theta_0 contains `-eta * sum_t b_t * J_t^T J_t * delta`,
  which grows linearly in T. The trainer leaves the region where it was a good
  policy, and the reward collapses.
* Group-centered advantages remove only the common-mode part
  (sum_i A_i = 0 per group). The bias in (3) is weighted by each token. It is
  sum_i A_i sum_t (q - p)(s_{i,t}), which is zero only if the sampler error
  along a trajectory is uncorrelated with that trajectory's advantage.
  Failed completions are usually longer, and they go through exactly the
  states where the sampler errs, for example low-p tokens that q
  over-samples. Both effects make the correlation non-zero, so the bias
  survives centering. With uncentered rewards (`plus_minus`, `zero_minus`)
  the bias is large from the first step.

**Why IS and clipping only slow it down.** Under a per-token weight
w(y) = w(p(y)/q(y)),

    E_{y~q}[w(y) grad log p(y)] = sum_y q(y) w(y) (e_y - p)        (logit space)
                                 = Z (mbar - p),    mbar = q w / Z,  Z = sum_y q w.        (4)

This is zero only when mbar = p, that is when w is exactly the ratio p/q. Any
truncation, clamp, drop or band moves mbar back toward q, and the bias
returns: `Z (mbar - p)` ranges from 0 (pure IS) to `q - p` (no correction).
Tighter token clipping moves mbar further toward q, which is why it can make
things worse. Untruncated IS has unbounded variance, and the ratio is
especially unreliable on exactly the tail tokens where q and p differ most.
Sequence-level weights scale A, but `E_q[grad log p] != 0` still holds
inside each token, so they leave (3) intact.

## 2. The modification

Center the per-token score under the distribution the score is actually
averaged against:

    score_t = log p(y_t) - sum_v sg(m_t(v)) log p_t(v),
    m_t = q_t w(p_t/q_t) / Z_t   (m_t = q_t with no correction)                (5)

    L = -(1/N) sum_i A_i sum_t w(y_{i,t}) score_{i,t}.

**Unbiasedness of the baseline part.** In logit space,
`grad_z score_t = (e_y - p) - sum_v m(v)(e_v - p) = e_y - m`. Hence, for any
c that does not depend on y_t,

    E_{y~q}[ c w(y) (e_y - m) ] = c ( q w - Z m ) = 0.        (6)

This is identity (1), restored under q. Constant offsets, group means, lengths
and visited states again contribute nothing in expectation. The
reward-independent drift term of (3) is gone, whatever the mismatch is.

**Unchanged on-policy.** If q = p (and w = 1), then
`grad sum_v sg(p(v)) log p(v) = sum_v p(v)(e_v - p) = 0`. The gradient is
identical to plain REINFORCE's (checked numerically). The same holds for
untruncated IS, where m = p.

**What remains is a signal, not a drift.** Without a correction,

    E_{y~q}[A (e_y - q)] = sum_y q(y) (Q(y) - Qbar_q) e_y        (logit space)

with Q(y) = E[A | s_t, y] and Qbar_q = E_q[Q]. This is the exact policy
gradient of E_q[Q] with respect to the logits of a softmax policy sitting at
q. In words, the trainer's logits are moved in the direction that would
improve the sampler. That direction vanishes when Q is constant, and it has a
positive inner product with the trainer's own improvement direction
`sum_y p (Q - Qbar_p) e_y` whenever q and p are close. The sampler error now
only scales and slightly rotates the reward signal. It no longer adds a
reward-free, sign-coherent push, so nothing accumulates across refreshes.
Formally, a training run with all rewards equal has zero expected update,
which is the behavior of on-policy REINFORCE, while the current loss drifts by
`A * grad E_q[log p]` per step.

**Interpretation.** The added term `-A sum_v q(v) log p(v)` is `+A` times the
cross-entropy H(q, p). It is a per-token control variate that cancels, in
expectation, the `A * (-grad KL(q||p))` hidden in the sampled score. On
average, a negative-advantage token no longer pushes p away from the sampler,
and a positive one no longer pulls p toward it. Only the deviation of the
sampled token from the sampler's mean moves the logits.

## 3. Estimating m when q is only partly known

The baseline needs `sum_v q(v) log p(v)`, and the sampler records q exactly
only for some tokens. `losses.py` uses whatever the batch contains:

* `sampling_vocab_logprobs` (`sampler.vocab_logprobs=full`): exact q.
* `sampler_topk_ids/logprobs` (`sampler.vocab_logprobs=k`): q is exact on
  K = top-k ∪ {y_t}. The remaining mass 1 - q(K) is spread over the tail in
  proportion to p:
  `qhat(v) = (1 - q(K)) p(v) / p(V \ K)` for v not in K. This is the
  minimum-KL-to-p completion of q's known part. It equals p when q = p, so
  the on-policy equivalence still holds exactly.
* Only `sampling_token_logprobs` (the default batch): K = {y_t}. This always
  works and has no extra cost.

For c = None the remaining bias is `q - qhat`, which is non-zero only in the
tail, where q's shape differs from p's. Its total size is at most
`2 (1 - q(K))`. With a top-k head that holds almost all of the sampler's
mass, it is negligible. With K = {y_t} alone, a numerical check on random
softmax pairs roughly halves `|bias|_1` relative to `q - p` for small
vocabularies but helps little for large ones. That is why the configuration
below records a top-k head.

With a correction, (5) uses `m ∝ qhat * w(p/qhat)`, computed with the
existing `log_mass` (the band, sign and TV-statistic rules are applied per
vocabulary entry). By (4)-(6) this keeps the score centered under any
token-level correction, so TIS, MIS, PPO and DPPO no longer reintroduce the
bias through their truncation. Sequence-level corrections only rescale A, and
(6) holds for any such c.

Implementation notes. Everything except log p is detached. The K-membership
mask is built from one-hot comparisons (no scatter on a possibly TP-sharded
vocabulary axis, resharded the same way as in `models/sampling.py`). All
work happens inside the existing rematerialized chunks, so memory stays at
[B, chunk, V]. The extra batch input is [B, T, k].

## 4. Sanity checks (CPU mini setup, weight noise 0.3)

The deliverable does not require these; they are here only as a sanity check.

* Gradient identity: the per-token logit gradient equals `-(A/N)(e_y - q)`
  with full q. It also matches top-k with k = V, and it equals the original
  gradient exactly when q = p. All IS configurations produce finite
  gradients.
* plus_minus rewards (dev default): final eval reward, original 0.66 (and
  falling) vs. 0.84 with top-4 and 0.85 with the full vocabulary.
* Adding 100 never-correct vocabulary tokens (`MINI_EXTRA_TOKENS=100`):
  the original loss collapses to reward 0 within 100 steps. The centered loss
  with k = 2, 4 or 8 reaches 0.82-0.83, close to the on-policy run (0.90).
  With K = {y} only, it still collapses.
* group_centered with noise 0.6: original 0.40 vs. 0.58 at the end.

## Configuration

sampler.vocab_logprobs=32

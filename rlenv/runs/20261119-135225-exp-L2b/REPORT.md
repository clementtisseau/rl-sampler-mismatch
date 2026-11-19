# REPORT: a score-function control variate for REINFORCE under sampler mismatch

## 1. What is wrong with the current loss

With no correction, `loss_fn_rl` computes, per completion token,

    L = -(1/N) sum_b A_b sum_t log p(y_bt | s_bt),

where y_bt is sampled from the sampler q, not from the trainer p, and A_b is the
(group-centred) advantage of the completion. Each token's contribution is a
REINFORCE score term with weight 1.

Fix a position with prefix s, and write z for the trainer's logits at s, so
p = softmax(z). For the sampled token y, the gradient of -log p(y|s) with respect
to z is e_y - p, where e_y is the one-hot vector of y. Treat the advantage A as
fixed at this position (the hint's single-token analysis). Taking the expectation
over y ~ q(.|s):

    E_{y~q}[ grad_z ( -A log p(y|s) ) ] = -A * sum_y q(y) (e_y - p) = -A (q - p).

So the expected update in logit space is the vector A (q - p).

- On-policy (q = p) the expected update is A * (p - p) = 0 for any constant A.
  Only the reward-dependence of A carries signal. The score function has zero mean
  under p, so the estimator is unbiased.
- Under mismatch (q != p) the expected update is A (q - p), which is nonzero even
  when A is constant. For A > 0 it pushes the trainer toward the sampler's
  distribution q. For A < 0 it pushes the trainer away from q. Mass that the sampler
  over-produces (for example noise-induced tail tokens) is systematically reinforced
  for positive advantages. It is not zero-mean noise: it is a deterministic function
  of (p, q) that does not average out over the batch.
- The bias compounds, because the sampler is refreshed from the trainer at every
  step. Each step moves p toward the q produced by the previous p.

## 2. Why the usual importance-sampling fixes only slow the drift

The IS-weighted term w = p(y)/q(y) gives E_q[w (e_y - p)] = sum_y p(y)(e_y - p) = 0.
So unclipped token-level IS is unbiased per position. Its cost is variance, which
grows as p/q becomes large (and is unbounded in the tail). Clipping or capping w,
as TIS, PPO and DAPO do, reintroduces bias. With m_y = min(p_y, c q_y) the clipped
weight gives

    E_q[min(p/q, c) (e_y - p)] = sum_y m_y (e_y - p) = m - (sum_y m_y) p,

which is nonzero whenever the cap binds. This is why stronger clipping (smaller c)
does not remove the drift, and can make it worse: the bias persists while the
useful signal shrinks.

## 3. The modification: a score-function control variate

The bias is the expectation of the score under q. That expectation is a known
function of the policy and the sampler, so it can be subtracted. Define

    b(s) = E_{v~q(.|s)} [ log p(v|s) ] = sum_v q(v|s) log p(v|s)   (q detached).

Its gradient in logit space is grad_z b = sum_v q(v) (e_v - p) = q - p. Replace the
per-token term log p(y|s) by

    log p(y|s) - b(s).

The loss becomes

    L_cv = -(1/N) sum_b A_b sum_t [ log p(y_bt|s_bt) - sum_v sg(q(v|s_bt)) log p(v|s_bt) ].

Expected logit gradient at a position, with A fixed:

    E_{y~q}[ -A ( grad_z log p(y) - grad_z b ) ]
      = -A [ sum_y q(y)(e_y - p) - (q - p) ]
      = -A [ (q - p) - (q - p) ] = 0.

So the mismatch bias A(q - p) cancels exactly. The on-policy case is unchanged in
expectation: when q = p the baseline term has zero mean, and the per-sample term
is the standard score e_y - p. The variance is comparable to plain REINFORCE, and
no importance weight is involved, so nothing can blow up and nothing is clipped.

Equivalently, the per-sample logit update is A (e_y - q) instead of A (e_y - p).
Its mean under q is zero whatever q is.

### Implementation (losses.py)

- `b(s)` is computed from the sampler's top-k distribution, which the trainer
  already receives (`sampler_topk_ids`, `sampler_topk_logprobs`, with
  `logprobs_topk=20`). The gather `take_vocab(logp_vocab, ids)` respects vocabulary
  sharding. The q weights are detached.
- The term is active whenever `correction is None` and the top-k arrays are in the
  batch. With an ISCorrection the original weighted score is kept unchanged.
- No configuration changes are needed.

### Residual error

- Truncation: with a top-k of K tokens, the missing tail mass tau = 1 - sum_top q
  adds a bias of order tau (q_tail - tau p) times A. For LLM samplers with K = 20
  this is typically small. With a full distribution (as in the mini setup, where the
  vocabulary is 12 tokens) the bias is exactly zero.
- Advantage dependence: the argument above holds for a constant A at the position.
  In general the sequence advantage varies with the next token, so the exact target
  is E_p[Q(s,y)(e_y - p)] and the fix gives E_q[Q(s,y)(e_y - q)], whose difference is
  O(|q - p| |Q|). The baseline removes the part of the bias that does not depend on the
  advantage, and this is the part that compounds with the constant-advantage
  component. The leftover O(|q - p|) term is much smaller than the original A(q - p),
  and it is not amplified by IS weights.

## 4. Empirical check (short, dev only)

Single-position identity: with q a random perturbation of p over 7 tokens, the
exact expected gradient of the original loss matched -(q - p) to machine precision,
and the expected gradient of L_cv was 0 to 1e-17.

Mini setting (`dev/mini.py`, default mismatch noise 0.3, 600 steps, seeds 0 to 2):
the original loss ends with eval reward 0.66 to 0.68, and the control-variate loss
ends with 0.83 to 0.86. At noise 0.6 (seed 0) the original collapsed to zero eval
reward, while the control variate reached 0.60 eval reward (its training reward
drifted down from a peak of 0.52 to about 0.35 over the run, so it is slowed but
not fully stable there). With 40 extra tail tokens in the vocabulary (noise 0.3,
seed 0) the original collapsed to zero eval reward, and the control variate ended at
0.84. These are short dev runs, not a full evaluation.

## 5. Summary

The mismatched REINFORCE update has expected logit direction A (q - p), which is
zero on-policy and biased under mismatch. Importance weights remove the bias only in
expectation and add variance, and clipping brings the bias back. Subtracting the
score's expectation under the sampler, b(s) = E_q[log p], removes the bias exactly
for constant advantages, without any weight. This is what `losses.py` implements.

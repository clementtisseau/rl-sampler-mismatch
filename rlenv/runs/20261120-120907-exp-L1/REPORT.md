# Sampler-mismatch bias in REINFORCE, and a sampler-centred score

## What is wrong with the current loss
Notation: at a state s the trainer has policy p(.|s), the sampler q(.|s). Tokens are drawn from q.
The loss uses the gradient estimator  g = sum_t A * grad log p(y_t|s_t),  y_t ~ q.
With y ~ p the score has zero mean, E_p[grad log p] = sum_v grad p_v = 0, so REINFORCE needs no
baseline to be unbiased. Under mismatch this is false:

  E_{y~q}[grad log p(y)] = sum_v q_v grad log p_v = -grad_theta CE(q, p) =: -grad_theta H(q,p)  != 0.

Write A = Abar + (A - Abar). The bias is therefore
  b_bias = sum_i sum_t A_i * (-grad H(q_t, p_t)).
It acts as a "regulariser" on the cross-entropy between sampler and trainer on the visited states, with weight A_i:
 * Sequences with A<0 (most of them when reward is sparse / plus_minus; or the longer ones, since the loss
   sums over tokens and group-centering only zeroes the mean over sequences, not over tokens) *maximise*
   H(q,p): p is pushed away from wherever q puts mass, mostly into the low-probability tail the noisy/quantised
   sampler under- or over-weights. Positive-A sequences minimise it, but they are rarer or shorter.
 * It does not average out: it is a deterministic drift, not zero-mean noise (E over y of the term is exactly
   the expression above, independent of the sample size).
 * It compounds: the sampler is refreshed from the trainer, q = q(p), so the drift moves p, which moves q, and
   the mismatch (and thus the drift) grows; eventually p collapses to a degenerate policy (reward 0,
   gradient ~0), as observed.
Importance weights w = p/q remove the drift only if unclipped (sum_v q_v (p_v/q_v) grad log p_v = 0), but they
have heavy tails (tiny q, larger p); clipping/dropping tokens with w outside a band restores a non-zero
sum_v q_v w_v grad log p_v, asymmetrically, i.e. re-creates a bias (stronger clipping = larger bias). That is why
IS/clipping only delays the collapse.

## The fix: centre the score under the sampler distribution
Subtract from each sampled token's log-prob its expectation under the sampler:
  score_t = log p(y_t|s_t) - b_t,   b_t = sum_v c_v log p_v,  c ~ q, sum_v c_v = 1,
  loss = - sum_t A * score_t / N.
Since b_t does not depend on y_t, E_{y~q}[A_const * (grad log p(y) - grad b_t)] = 0 exactly when c = q:
the bias term above is cancelled for every advantage, including its correlation with sequence length and
with the mean of A (any baseline offset, plain +-1 rewards included). What remains is only
sum_v q_v (A_v - Abar) grad log p_v, the usual REINFORCE signal evaluated with q instead of p, an O(|p-q|)
error that is not accumulated in a fixed direction. b is also a variance-reducing baseline (it depends on
the state only).

Computing it: the full q is not stored, but the sampler returns its top-k logprobs (k=20). Take c_v = q_v on
the top-k set K, and spread the leftover sampler mass q_tail = 1 - sum_K q_v over the remaining tokens in
proportion to p. All weights are stop-gradient. Using sum_v p_v grad log p_v = 0 the gradient of b reduces to
  grad b = sum_{v in K} ( q_v - q_tail p_v / (1 - P_K) ) grad log p_v,   P_K = sum_K p_v,
needing only a gather of log p at the k ids (plus sum_v p log p, already computed for the entropy). It is exactly
the q-expectation when the tail of q has the shape of the tail of p (or is empty); otherwise only the tail part
is approximate. If the batch has no top-k keys the loss falls back to plain REINFORCE. It is active with
correction=None; if a correction is given its weight w multiplies log p (the baseline is left unweighted).
Multiplying the whole centred score by w is harmful (w(log p - b) is no longer centred under q); in my
CPU experiments adding IS on top of the centring made training collapse, the centring alone did not.

Experiments (dev/mini.py, 1000 extra tail tokens, 4 seeds/noise levels): plain PG and token IS/TIS collapse
to reward 0 in some runs; the centred loss collapsed in none (final eval 0.33-0.86), and also helped without tail tokens.

## Configuration
rl.is.level=null

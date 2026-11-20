# Score control variate for REINFORCE under a sampler mismatch

## What is wrong with the current loss

Write the trainer's next-token distribution at a prefix s as p(.|s) = softmax(z(s)), the sampler's as q(.|s),
and the per-sequence advantage as A (group-centered). With no correction, `loss_fn_rl` returns
L = -(1/N) sum_i A_i sum_t log p(x_it | s_it), and the sampled tokens x_it are drawn from q, not p.
Its expected gradient is therefore

    E_q[grad L] ~ -sum_t E_q[ A * grad log p(x_t | s_t) ].

The score identity that makes REINFORCE work is E_p[grad log p(x|s)] = 0, which holds because
grad log p(x|s) = sum_v (1[x=v] - p_v) grad z_v and sum_x p_x (1[x=v] - p_v) = 0. The same identity
is what lets any state-only baseline b(s) be subtracted for free: E_p[b(s) grad log p] = 0.

Under the sampler it fails. For any q != p,

    E_q[grad log p(x|s)] = sum_v (q_v - p_v) grad z_v =: beta(s)   (nonzero in general).

So a state-only baseline no longer removes anything, and the constant part of the advantage
contributes a bias c(s) * beta(s), where c(s) is the advantage averaged over the sampler's
continuations at s. In logit space this bias is a push of the trainer's logits along (q - p),
independent of reward. Every step the sampler is refreshed from the trainer, so the same
push is re-applied to the new trainer, which is why it compounds rather than averaging out.
Clipping or truncating the token-level ratio removes the correction exactly on the tokens where
q is much smaller than p (the tail), so stronger clipping moves the bias rather than removing it.

## The modification

Replace the per-token score grad log p(x_t|s_t) by its centered version

    psi_t = grad log p(x_t|s_t) - E_{v ~ q(.|s_t)} [ grad log p(v|s_t) ].

For any p and q, E_q[psi_t | s_t] = 0 exactly, so psi_t is a valid zero-mean score term under the
sampler, and E_q[b(s) psi_t] = 0 for any state-only b(s). On-policy (q = p) the subtracted term has
expectation E_p[grad log p] = 0, so the modification does not change the on-policy target; it only
removes the sampler-dependent part. In `losses.py` the expectation is computed with q approximated
by the sampler's top-K log-probabilities (`sampler_topk_logprobs`, renormalized over the K tokens,
which the trainer already receives), and the loss uses

    score_t = w_t log p(x_t|s_t) - sum_{v in topK} qhat(v|s_t) log p(v|s_t),

where w_t is the existing IS weight. The control variate is applied when no token-level correction is
set (`correction=None`, the default); with a token-level IS weight, w_t p/q already gives E_q[w grad log p] = E_p[grad log p] = 0 and the
subtraction is skipped. If the sampler returns no top-K fields, the loss falls back to the plain score.

Because the subtracted term has zero conditional mean under q and p alike, the change is
exactly unbiased for the constant-advantage part of the gradient and leaves the on-policy expectation
unchanged. It is a variance-changing control variate, not a reweighting, so it does not inherit the
variance blow-up or the clipping bias of importance weights.

## Limits

The advantage A is sequence-level, so it depends on tokens after s_t. The term
sum_v (q_v - p_v) (Q(s,v) - c(s)) grad z_v, where Q(s,v) = E[A | s, x_t = v], is not removed by
the control variate. Removing it exactly needs sequence-level importance weights, which the
existing `rl.is.*` options approximate with their known variance and clipping bias. The top-K
renormalization also leaves an error proportional to the sampler's tail mass outside the top 20.

## Evidence

These are dev-run checks on `dev/mini.py` only (600 steps, lr 0.2, plus-minus rewards), not Qwen3 runs:

- noise 0.0: final eval 0.929 original, 0.932 modified (no harm on-policy);
- noise 0.3: final eval 0.66 (2 seeds) original vs 0.84 (2 seeds) modified;
- noise 0.5: original collapses to eval 0 (peak 0.34); modified holds eval 0.72;
- noise 0.8: original collapses to eval 0 by step 75; modified stays at eval about 0.3 to 0.4, but
  `kl_samp_train` keeps growing (to about 17), so it slows the drift rather than removing it.

## Configuration

No configuration changes are needed; the modification is active with default arguments.

# Sampler-mismatch bias in REINFORCE, and a sampler-centred score

## 1. What is wrong with the current loss

`loss_fn_rl` with `correction=None` minimises

    L(theta) = -(1/N) sum_i A_i sum_t logp_theta(y_it | y_i,<t),

where the y_it are sampled from the sampler q, not the trainer p. The
gradient it estimates is therefore

    g_q = E_q[ A ∇ logp ],          while the intended gradient is  g_p = E_p[ A ∇ logp ].

Their difference is the mismatch bias, `g_q - g_p`. Nothing in the loss
accounts for it.

The bias is not zero-mean noise, and the cause is concentrated in one term.
Write the advantage as A = b + (A - b), where b is a baseline that depends only
on the prefix (a constant, or a group mean). On-policy the baseline term
vanishes in expectation, because the score has zero mean under the distribution
that generated it:

    E_p[ ∇ logp(y|h) ] = sum_y p(y|h) ∇ log p(y|h) = ∇ sum_y p(y|h) = 0.

Under the sampler it does not:

    E_q[ b ∇ logp(y|h) ] = b · sum_y q(y|h) ∇ log p(y|h) = -b ∇_theta H(q_h, p_h),

where H(q,p) = -sum_y q log p is the cross-entropy. So every non-zero advantage
offset injects a cross-entropy gradient that pulls the trainer's next-token
distribution toward or away from the sampler's. The size is proportional to
|b| times the sampler/trainer difference, so it is systematic. Because the
sampler is refreshed from the trainer at each step, the pull is re-applied to
the new trainer and accumulates.

This term is large in practice. Advantages are not zero-mean in general:
group-centring removes the group mean, but a group can be all-unsolved, and
token-level normalisation weights longer completions more. Any leftover mean
is b.

Evidence from the CPU stand-in (`dev/mini.py`, with the same loss interface,
over-sized vocabulary of unlikely tokens, sampler weight noise 0.3 to 0.6):
plain REINFORCE with unshifted (+1/-1) advantages collapses to zero reward
within about 50 steps, and the sampler's mass on never-correct tokens goes to
zero. The same run with zero-mean (group-centred) advantages is stable. The
collapse is driven by the offset, not by the mismatch alone.

Importance-sampling corrections only reweight the samples. They change the
covariance term, but they do not remove the cross-entropy term from an offset,
and token-level clipping or dropping removes exactly the tokens whose ratio is
far from one. That is why clipping slows the degradation but does not stop it,
and why stronger token-level clipping can make it worse.

## 2. Modification: sampler-centred score

Use the sampler's own distribution as a control variate. For each position h,
define the centre

    c_h = E_{v~q(.|h)} [ w_v ∇ log p(v|h) ],

with w_v = w(p_v/q_v) the correction weight, or w_v = 1 when there is no
correction. Replace the per-token score by

    s(y|h) = w(y) ∇ log p(y|h) - c_h,

and use the loss

    L'(theta) = -(1/N) sum_i A_i sum_t [ w(y_it) logp(y_it) - sum_v q̃_v w_v logp(v) ],

with the centre, the sampler probabilities, and the weights detached and the
gradient taken through logp(v).

Why this is the fix. For any prefix-measurable offset b_h, the centred term has
zero mean under the sampler:

    E_q[ b_h (w(y)∇logp(y) - c_h) ] = b_h ( E_q[w ∇logp] - c_h ) = 0.

So the cross-entropy term from §1 is cancelled exactly, for any constant offset,
including the one that group-centring leaves. For a general advantage the
subtracted term is E_q[Ā_h c_h], where Ā_h = E_q[A | h]. That is the
cross-entropy gradient weighted by the expected advantage at that prefix, and
it is precisely the bias from §1. The remaining part is the covariance term
E_q[(A - Ā_h) w ∇logp], which is what the importance weights are for.

Numerical check. With a 6-way categorical, sampler logits z+d, trainer logits z,
A = -1 constant: the true gradient is 0, the uncentred estimator under q has
norm 0.25 (bias), and the centred estimator under q is exactly 0.

Correction interaction. If the centre were not weighted by the same w, the
constant-offset cancellation would fail when w is clipped (e.g. TIS). Weighting
the centre by w_v makes the identity hold for any w: with full importance
weights, w_v q_v = p_v and c_h = sum p ∇logp = 0, which recovers the usual
unbiased IS estimator.

Approximation. The sampler's full distribution is not in the batch. The loss
uses the sampler's top-K logprobs (K=20 in `train_rl.py`, already returned by
`sampling.generate`), with q̃ = softmax over the top-K. Tail mass outside
top-K is ignored. This is exact only when the top-K covers the sampler's mass; the
residual is the tail mass outside top-K, which is small when the sampler is a
near-copy of the trainer, but is not bounded in general.

## 3. Why it helps, in one sentence

Any mean offset of the advantage leaks sampler-trainer cross-entropy into the
gradient; subtracting the sampler's expectation of the same score removes the
leak, so the only mismatch left in the update is the covariance term that
importance weights correct.

## Configuration

No `rl.is.*` changes are needed. The modification is active with the defaults
(`correction=None`) and also works with a correction.

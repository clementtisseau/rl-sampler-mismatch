# Sampler-expected score baseline for REINFORCE under sampler mismatch

## What is wrong with the current loss

Notation: at a position with prefix s, the trainer is p(x|s) = softmax(z)(x) and the
sampler is q(x|s). The advantage is A (constant over the tokens of a sequence for this
argument). The loss `loss_fn_rl` (no correction) is `-A * log p(x_t)` summed over tokens,
with x_t drawn from q.

Its gradient at one position, averaged over x ~ q, is

    E_q[ -A grad log p(x) ] = -A * sum_x q(x) grad log p(x).

Using the softmax identity grad_z log p(x) = e_x - p, we get sum_x q(x)(e_x - p) = q - p, so
in logit space the expected update is

    E_q[ -A (e_x - p) ] = -A (q - p).

On-policy (q = p) this vanishes, since sum_x p(x) grad log p(x) = grad sum_x p(x) = 0. Under
mismatch (q != p) it does not. Gradient descent then gives the logit step
Delta z = eta * A * (q - p): for A > 0 the trainer is pushed toward the sampler's
distribution, for A < 0 it is pushed away from it. This is a deterministic drift, not
zero-mean noise. It does not average out over a batch or over steps. The sampler is
refreshed from the trainer every step, so q_{t+1} is built from p_t, and the drift
compounds. The term is -A grad_theta CE(q, p), a cross-entropy pull toward q.

Unclipped per-token IS (w = p/q) removes the bias at a single position, since
E_q[(p/q) grad log p] = sum_x p grad log p = 0. But it multiplies the variance by
E_q[(p/q)^2], which is large where the sampler under-samples tokens the trainer likes.
Clipping removes the variance but reintroduces bias, which grows as the clip tightens.
That matches the known behaviour (IS and clipping slow the drift but do not remove it).

## The modification

Subtract from each token's score the expectation of the score under the sampler's own
distribution, i.e. a baseline that is a function of the prefix only:

    b(s) = sum_y q(y|s) log p(y|s),     grad_theta b = sum_y q(y|s) grad log p(y|s).

The per-token term becomes `-A (log p(x_t|s) - b(s))`. In `losses.py` the sum over y runs
over the sampler's top-K logprobs, which `train_rl.py` already passes as
`sampler_topk_ids` and `sampler_topk_logprobs` (K = 20). The baseline is stop-gradient on q
and differentiated through log p only. It is active with the default arguments and needs no
configuration.

## Why it works

Expected update at one position, averaged over x ~ q, with A constant:

    E_q[ -A (grad log p(x) - sum_y q(y) grad log p(y)) ]
      = -A ( sum_x q(x) grad log p(x) - sum_y q(y) grad log p(y) ) = 0.

In logit space: E_q[(e_x - p) - (q - p)] = (q - p) - (q - p) = 0. The expected update is zero for
every q, and the on-policy value (zero) is unchanged. The baseline removes exactly the
mismatch drift -A(q - p). It does not change the on-policy estimator: when q = p the
baseline gradient is sum_y p grad log p = 0 at every position, so the loss is identical in
expectation to REINFORCE. The baseline also does not depend on the sampled token, so it
adds no sampling bias on its own.

For an advantage that varies across tokens, write Abar_q = E_q[A | s]. The expected update is

    E_q[ A(x)(grad log p(x) - c) ] = sum_x q(x) A(x) grad log p(x) - Abar_q c,   c = sum_y q(y) grad log p(y).

Subtracting the on-policy target sum_x p(x) A(x) grad log p(x) leaves the residual bias

    sum_x (q(x) - p(x)) (A(x) - Abar_q) grad log p(x).

The uncorrected loss has bias sum_x (q - p)(x) A(x) grad log p(x) = Abar_q c + residual. The
baseline removes the Abar_q c part, which is the part that does not depend on how A varies
across tokens and is the part that compounds. What remains is the mismatch (q - p) times the
centred advantage. It vanishes when q = p, and it vanishes when A is constant across tokens.

## Caveats

- Top-K truncation: the sum over y uses K = 20 sampler logprobs. The tail mass outside the
  top-K is not in the baseline, which leaves a small residual of order the tail mass. The
  sampled token is always in the sum when it is in the top-K.
- The alignment of sampler_topk_* with the targets is assumed to match
  sampling_token_logprobs (position t holds the distribution of token t+1), as in dev/mini.py.
- Combining the baseline with per-token IS weights (p/q, TIS, etc.) was tried in the CPU
  harness and collapsed again. The baseline is therefore used alone in the default loss, and
  the correction, if any, is applied on top of it without changing its form.

## Configuration

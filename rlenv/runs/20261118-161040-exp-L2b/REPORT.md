# REINFORCE under sampler mismatch: a score taken under the sampler

## 1. What is wrong with the current loss

Write the trainer's next-token distribution at prefix s as p(.|s) = softmax(z(s)), and the sampler's as q(.|s). Ignoring the correction (`correction=None`, so w = 1), `loss_fn_rl` is

    L(theta) = -(1/N) sum_i A_i sum_t log p_theta(x_it | s_it),

with x_it ~ q. Its ascent direction is g = (1/N) sum_i A_i sum_t grad log p(x_it | s_it).

Take one position and condition on the prefix s. For the sampled token x ~ q, the gradient of log p with respect to the logits is e_x - p, so the expected per-position score under the sampler is

    E_{x~q}[ grad_z log p(x|s) ] = sum_x q(x) (e_x - p) = q - p.

Now let the advantage be A(x) at this position (it depends on the sampled token and on the rest of the sequence). The expected per-position update is

    E_q[ A(x) (e_x - p) ]  =  sum_x q(x) A(x) (e_x - p)  =  (q * A) - p <q, A>        (logit space),

where * is elementwise. On-policy (q = p) the same expression is the true policy gradient, E_p[A (e_x - p)]. The "-p" term is what makes the on-policy case work: it is the score baseline, and E_p[e_x - p] = 0. Under mismatch, the baseline no longer has zero mean, since E_q[e_x - p] = q - p != 0.

Consider the part of A that does not depend on the sampled token, A(x) = a(s). This is the common case of a group-centred reward that is nearly constant across the tokens of a sequence, and it is the case that drives the bias. Its expected update is

    a(s) (q - p)    (logit space),

which is nonzero whenever q != p. For a > 0 the update is supervised learning toward the sampler's distribution q; for a < 0 it is unlearning q. This is a systematic bias, not zero-mean noise. It has the same direction at every step, and since the sampler is refreshed from the trainer, q is the trainer plus its noise or staleness, so the bias pulls the trainer toward its own perturbed copy and compounds.

Importance weights do not fix this by themselves. The per-token weight w = p(x)/q(x) makes the expectation exact, E_q[w (e_x - p)] = p - p = 0, but it is unbounded. Clipping it to a bound c gives E_q[min(p/q, c) (e_x - p)] != 0, which reintroduces a bias of the same kind. This is consistent with clipping slowing the drift down without removing it, and with stronger clipping being worse.

## 2. The modification: a score under the sampler

Replace the sampled-token score by a control variate that uses the sampler's full distribution at the same position:

    S(x|s) = grad log p(x|s) - sum_y q(y|s) grad log p(y|s).

The second term does not depend on x, so it is a baseline that depends only on the prefix. It is applied by a surrogate loss, with q stop-gradiented and the gradient flowing through log p:

    score(x|s) = log p(x|s) - sum_y sg(q(y|s)) log p(y|s).

Because the subtracted term is an expectation under q, its contribution is zero in expectation under the sampler:

    E_{x~q}[ S(x|s) ] = sum_x q(x) grad log p(x|s) - sum_y q(y|s) grad log p(y|s) = 0    for every p and q.

So for any advantage that is constant in x at this position, the expected update is exactly zero, whatever the mismatch.

In logit space the modification is exact and easy to read. Since sum_y q_y (e_y - p) = q - p,

    grad_z S(x|s) = (e_x - p) - (q - p) = e_x - q.

The on-policy score e_x - p is replaced by e_x - q. The two agree when q = p, so nothing changes on-policy, and the mean of e_x - q under q is zero for any q. The same argument gives the expected update of the full loss: the constant-advantage part of the update is a(s) E_q[e_x - q] = 0, so the bias a(s)(q - p) of Section 1 is gone.

With a correction (`rl.is.*`), the IS weight w is kept and multiplies S, so the weight and the baseline compose.

## 3. What remains, and why it is a smaller error

The advantage is not constant in x. Its dependence on x is part of the signal. Let A(x) = a + B(x), where a is constant and B is the part of the advantage that depends on the token. Then the expected update is

    E_q[ A(x) S(x|s) ] = E_q[ B(x) S(x|s) ].

The on-policy gradient is E_p[B(x)(e_x - p)] = p*B - p<p,B>. The modified loss under the sampler gives E_q[B(x)(e_x - q)] = q*B - q<q,B>. Their difference is

    (q - p)*B - (q<q,B> - p<p,B>),

which is first order in q - p and multiplied by the advantage contrast B, not by the full advantage A. This is the usual off-policy error. It does not accumulate into a fixed-direction drift toward the sampler, because it has no constant component: it is the advantage-weighted part, which changes sign with the reward. Importance weights address exactly this remaining term, so the two can be combined.

## 4. Why this is the right fix

The mismatch enters the loss only through the sampling distribution of the score. The modified loss uses the sampler's distribution to cancel the score's mean under the sampler. With the same sampler, the constant-advantage bias is zero for every p, q, noise level, staleness, quantization or KV-cache setting, and the update reduces to the on-policy REINFORCE update when q = p. The fix is therefore a sampler-side control variate, not a clipping rule, so it does not trade bias for stability.

## 5. Cost and conditions

The baseline needs the sampler's full next-token distribution at each completion position, q(.|s). Without it (`sampler.vocab_logprobs=null`) the loss has only the sampled token's logprob, and the code falls back to the original score log p(x|s). In that case the fix is inactive. This is the one condition of the modification, and it is a data condition, not a flag of the loss. The configuration below makes it active. The cost is a [B, T, V] fp32 array on the sampler side, which is resharded to the trainer's (data, model) layout. Its memory scales with the vocabulary, as the trainer's own full-vocabulary log-softmax already does.

## Configuration

sampler.vocab_logprobs=full

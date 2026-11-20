# Sampler-centered score function

## What is wrong
Samples come from q, the model being trained is p, and the loss is
L = -(1/N) sum_t A * log p(y_t), with y_t ~ q. In logit space z_t, the gradient of
log p(y) is e_y - p. Take advantage A of the sequence and the prefix s_t. Then

E_q[A (e_y - p) | s] = E_q[A (e_y - q) | s] + E[A | s] (q - p).

* The first term is the policy gradient. It is the desired signal, evaluated at q.
* The second term is pure bias, `V(s)(q - p)`, with V(s) = E[A | s] the prefix
  value. On-policy (q = p) it vanishes, because E_p[e_y - p] = 0: the score
  function has zero mean, which is why REINFORCE needs no per-token baseline.
  When q != p, E_q[grad log p] = -grad KL(q||p) != 0. The update then moves p
  *toward* q on good prefixes (V > 0) and *away* from q on bad prefixes (V < 0).
  Away-from-q moves widen the mismatch, and the group-centered advantage does not
  cancel V(s) at the token level. Sequence lengths differ and N normalises by
  tokens. The sampler is rebuilt from p at each step, so the bias is systematic
  and compounds. It is not zero-mean noise: its size scales with |q - p|, and
  |q - p| grows as it acts.
* IS weights w = p/q restore E_q[w(e_y - p)] = p - p = 0 exactly. Clipping,
  dropping or banding w breaks this identity, so clipped corrections leave a bias
  of order E_q[(1-w_clip) (e_y - p)]. Tighter clipping gives a larger bias, which
  is why stronger clipping can worsen the collapse. Unclipped w has huge variance.

## Fix
Make the per-token score zero-mean under the sampler, not under p. In logit
space use e_y - q instead of e_y - p. Then E_q[A(e_y - q) | s] is exactly the
policy gradient, with no V(s)(q - p) term for any q, and no importance weights
or clipping are needed. The sampler returns its top-20 logprobs, so q is
nearly known. I use the surrogate

  ell_t = log p(y_t) - b_t,   b_t = sum_v qhat_v log p(v)  (detached qhat),

where qhat = q on the sampler's top-k tokens, and the leftover mass
1 - sum_topk q is spread over the other tokens in proportion to p. Since
grad_z log p(v) = e_v - p and sum qhat = 1, grad_z ell = (e_y - p) - (qhat - p)
= e_y - qhat. This costs no extra vocabulary pass beyond what is already
computed in the chunk. With q = p we get qhat = p up to the top-k truncation, so
b_t is constant in expectation and the loss reduces to ordinary REINFORCE. The
loss is -(1/N) sum A * w * ell_t, where w is 1 by default (correction=None) and
is the usual IS weight if a correction is passed. If top-k is absent from the
batch, it falls back to plain log p. The residual bias is only from tail
mass whose q-distribution is not known (assumed ~p there).

In the CPU dev setup (noise 0.3) this gives eval reward rising to ~0.82, against
a plateau near 0.7 for the old loss.

## Configuration

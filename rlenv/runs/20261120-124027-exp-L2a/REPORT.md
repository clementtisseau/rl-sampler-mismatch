# Sampler-centered REINFORCE

## 1. What is wrong with the current loss

For one completion y = (y_1..y_T) with advantage A, the loss gives the update
(per token, in the logits z_t of the trainer at context x_t = (prompt, y_<t))

    g = A * sum_t grad log p(y_t | x_t),      d log p(y|x)/dz = e_y - p     (1)

with tokens drawn from the **sampler** q, not from p.

**On-policy (q = p).** The score has zero mean at every context:

    E_{y~p}[grad log p(y|x)] = sum_v p(v) grad log p(v) = grad sum_v p(v) = 0.   (2)

So any part of the advantage that does not depend on the token y_t
(the group mean, the +-1 offset of `plus_minus`, the value V(x_t) of the
current context) contributes nothing in expectation: only the part of A that
is correlated with the choice of y_t moves the policy. Writing
Q(x,v) = E[A | x_t = x, y_t = v] and V(x) = sum_v q(v) Q(x,v),

    E_p[g] = sum_t E_x[ sum_v p(v) Q(x,v) (e_v - p) ] = sum_t E_x[ p ⊙ (Q - V) ]   (3)

i.e. the logit of v rises iff v is better than average. This is the policy
gradient.

**Mismatched (q != p).** (2) no longer holds:

    m(x) := E_{y~q}[grad log p(y|x)] = sum_v q(v) grad log p(v)
          = -grad_theta CE(q(.|x), p(.|x)),     in logits: m = q - p != 0.    (4)

Then

    E_q[g] = sum_t E_{x~q}[ sum_v q(v) (Q(x,v) - V(x)) grad log p(v) ]   (policy-improvement part)
           + sum_t E_{x~q}[ V(x) * m(x) ]                                 (bias)          (5)

The second term is a **reward-independent drift**: in logit space it is
V(x)(q - p). It does not care which token is better; it only depends on the
sign of the (offset) value of the context:

* V(x) > 0 (e.g. good contexts, or `plus_zero`): p is pulled towards q,
  i.e. SFT on the sampler's mistakes (dp moves along the perturbation).
* V(x) < 0 (bad contexts, failed rollouts, `plus_minus` early in training,
  any context whose value is below the group mean): gradient **ascent on
  CE(q, p)**, which pushes p away from q. CE(q,p) is unbounded above: it is
  maximized by moving p's mass off the tokens q likes onto tokens q
  dislikes, in particular onto the long tail of unlikely tokens (where noise
  makes q < p or q > p systematically).

Group-centering does not remove it. With A_i = r_i - mean_j r_j,
sum_i A_i S_i = sum_i r_i (S_i - S_bar) and

    E_q[sum_i A_i S_i] = (G-1) Cov_q(r, S) = (G-1) (E_q[r S] - E_q[r] E_q[S]),

and E_q[S] = sum_t E[m(x_t)] != 0; moreover the group mean is a baseline for
the prompt only, while V(x_t) - b varies along the sequence, so every
later position carries the drift V(x_t) - b times m(x_t).

**Why it compounds.** The mismatch is a fixed map θ -> q_θ (fixed weight
noise, quantization, a stale copy). The drift (5) has the same direction
(q_θ - p_θ, signed by V) at every step, and after the update the sampler is
rebuilt from the new trainer, so the mismatch -- and the drift -- are
regenerated rather than consumed. The bias is therefore integrated over
training (linear accumulation in θ), not averaged out like zero-mean noise.
Once p puts mass on tail tokens, q (a perturbation of p) samples them, those
rollouts fail (V < 0), and the drift away from q gets stronger: collapse.

**Why IS/clipping only slow it.** A token weight w(y) changes the drift to
E_q[w grad log p] = sum_v q(v) w(v) grad log p(v). This is zero only for the
exact ratio w = p/q (then it is sum_v grad p(v) = 0); this is the only
weight of the form f(y) that zeroes it for all directions, since the
condition is q(v) f(v) / p(v) = const. Any truncation, clamping or dropping
(TIS, MIS, PPO, DPPO ...) reintroduces a nonzero
sum_v q(v)(w(v) - p(v)/q(v)) grad log p(v), and stronger clipping means a
larger departure from p/q, hence a larger drift -- which is why it can be
worse. Exact IS removes it but at a variance cost, and sequence-level
weights don't center the per-token score at all.

## 2. The modification: center each token's score under the sampler

Replace the per-token score by its **q-centered** version

    c_t = w(y_t) grad log p(y_t|x_t) - sum_v q(v|x_t) w(v) grad log p(v|x_t)       (6)

(w = 1 without a correction; w = the token IS weight when a token-level
`ISCorrection` is given). The surrogate loss is

    L = - (1/N) sum_i A_i sum_t [ w(y_t) log p(y_t) - sum_v sg(q(v) w(v)) log p(v) ],

with q and w detached. Properties:

1. **Zero-mean under q for any q:** E_{y_t~q}[c_t | x_t] = 0 by construction.
   Hence every token-independent part of the advantage (group mean,
   reward offset, V(x_t)) has zero expected effect again, exactly as
   on-policy, and the bias term in (5) vanishes:

       E_q[A sum_t c_t] = sum_t E_{x~q}[ sum_v q(v) (Q(x,v) - V(x)) w(v) (grad log p(v) - m_w(x)) ].

   With w = 1 in logit space this is sum_t E_x[ q ⊙ (Q - V) ]: compare (3).
   The logit of a token goes up iff it is better than average (under the
   sampler's own advantage). It is a policy-improvement direction with no
   component along q - p that does not depend on reward, so there is
   nothing systematic to accumulate. The remaining off-policy error
   (q instead of p in the weighting q ⊙ (Q-V), and states drawn from q) is
   second order: it rescales the improvement direction per token but never
   has a sign independent of the reward; the fixed point "all advantages
   zero" is still a fixed point.

2. **Exactly REINFORCE on-policy:** if q = p, sum_v p(v) grad log p(v) = 0,
   so the extra term has zero gradient (verified numerically: gradients are
   identical to the original loss to 3e-8 when q = p).

3. **Composes with IS:** with exact w = p/q the subtracted term is
   sum_v grad p(v) = 0 (nothing changes); with clipped/truncated weights it
   removes exactly the drift sum_v q(v)(w(v) - p/q) grad log p(v) that the
   clipping creates.

4. **Not a KL penalty.** It does not pull p to q (that would be the
   biased term itself with a fixed sign); its expected contribution is
   zero, it only removes the conditional mean of the score.

Interpretation: the centering is a per-context, per-token control variate,
equivalent to replacing the logit-gradient e_y - p of the sampled token by
e_y - q. Plain REINFORCE uses p as the "expected one-hot", which is right
only when the tokens come from p.

## 3. Computing E_q with what the sampler returns

The sampler returns q only on its top-k tokens (k = 20, `sampler_topk_ids`,
`sampler_topk_logprobs`, present in both `train_rl.py` and `dev/mini.py`).
Let K be that set, q_tail = 1 - sum_{v in K} q(v), p_tail = 1 - sum_{v in K} p(v).
The tail is treated as one outcome, approximating q on the tail by
q_tail p(v)/p_tail:

    sum_v q(v) w(v) grad log p(v)  ≈  sum_{v in K} q(v) w(v) grad log p(v) + q_tail w_tail grad log p_tail.

In logits this is e_y - q~ with q~ = q on K and q_tail * p/p_tail on the
tail, so E_q[e_y - q~] = q - q~ is non-zero only *inside* the tail and sums
to zero there: the residual drift cannot move mass between the top-k and the
tail, or change the tail's total mass; it is bounded by the tail mass of q
(typically ~1e-2). When q = p it is exact. The top-k logprobs are gathered
from the trainer's log-softmax with the same `take_vocab` helper (TP-safe),
everything in float32, inside the existing rematerialized chunks; the cost is
a [B, chunk, 20] gather. Numerics: log p_tail uses a double `where` so that
positions where the top-k covers the whole vocabulary give zero, finite
gradients; non-finite sampler logprobs are treated as q = 0. If the batch has
no top-k logprobs the loss falls back to plain REINFORCE.

The token weight for the top-k and tail outcomes is computed with the
existing `is_weight` (detached), so `rl.is.*` token-level corrections stay
valid; sequence-level corrections still multiply the advantage as before.
Metrics add `sampler_tail_mass` (mean q_tail).

## 4. Sanity checks (dev/mini.py, 600 steps; not needed for the argument)

Mean reward of the last 50 steps, original loss -> centered loss:

| setting | original | centered |
| --- | --- | --- |
| default (noise 0.3, plus_minus), seed 0 | 0.58 | 0.87 |
| default, seed 1 | 0.58 | 0.85 |
| 50 extra tail tokens, seeds 0/1 | 0.00 / 0.00 (collapse) | 0.78 / 0.86 |
| 200 extra tokens, seed 2 | 0.00 (collapse) | 0.73 |
| 50 extra, noise 0.5 | 0.00 (collapse) | 0.52 (rising) |
| 50 extra, group_centered | 0.71 | 0.72 |
| 200 extra, TIS (high=2, clamp) | 0.59 (eval dips to 0.20) | 0.62 (no dip) |

## Configuration

No configuration changes are needed: the centering is active with the
default `correction=None`.

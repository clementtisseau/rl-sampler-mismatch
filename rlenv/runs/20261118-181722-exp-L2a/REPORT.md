# Sampler mismatch in REINFORCE: the score has a nonzero mean under q

## 1. What is wrong with the current loss

Notation. At completion position t of rollout i, the prefix is s = (prompt, y_<t) and the trainer
distribution is p(v|s), the sampler distribution is q(v|s) (q is the sampler's own distribution,
independent of the trainer parameters theta). The advantage A_i is a function of the whole rollout.
Let Q(s,v) = E[A | prefix s, token v at t] be the expected advantage given that the token is v
(the future is drawn from the sampler).

`loss_fn_rl` with `correction=None` uses weight 1 on every token:

    L = -(1/N) sum_i A_i sum_t log p(y_it | s_it),   y_it ~ q.

Its expected gradient at one position is therefore

    E_q[A grad log p(y_t|s)] = sum_v q(v|s) Q(s,v) grad log p(v|s).          (1)

On-policy (q = p) the same expression is the policy gradient
sum_v p(v|s) Q(s,v) grad log p(v|s), which is what REINFORCE should follow.
The mismatch enters (1) in two places:

(a) Weighting. The tokens are weighted by q instead of p. The difference from the target is
first order in delta = q - p:

    E_q[A grad log p] - E_p[A grad log p] = sum_v (q(v)-p(v)) Q(s,v) grad log p(v|s).

(b) The score has a nonzero mean under q. On-policy, the score has zero mean,
sum_v p(v|s) grad log p(v|s) = grad sum_v p(v|s) = 0, so any constant advantage gives no
update. Under q,

    sum_v q(v|s) grad log p(v|s) = grad sum_v q(v|s) log p(v|s) = -grad KL(q(.|s) || p(.|s)),

which is nonzero. In logit space (grad_z log p(v) = e_v - p) a constant advantage c moves the
trainer logits by c (q - p). A constant advantage is pure noise on-policy, but under mismatch
it is a systematic push of p toward q (c > 0) or away from q (c < 0).

Group-centered rewards remove only the batch-mean part of the advantage. They do not remove
the per-prefix mean E_q[A | s], which is what multiplies the score in (b).

The sign of (b) explains the observed collapse. With plus_minus rewards most early rollouts
have A = -1. The update then pushes p away from q. The sampler is refreshed from p at every step,
so the push compounds. In the dev runs (mini setting with a tail of never-correct tokens) the
sampler's tail mass falls to zero and the reward goes to zero.

Clipping or truncating p/q (TIS, MIS, PPO-style bands) keeps (b) because the weight w is not
p/q, so sum_v q(v) w(v) grad log p(v) != sum_v p(v) grad log p(v) = 0. A stronger clip makes
w deviate more from p/q, so the bias grows. This matches the observed "slows but does not
remove" behaviour.

## 2. The modification: centre the score by the sampler's expected score

Keep the trainer's per-token score, but subtract its expectation under the sampler at the same
position. With the full sampler distribution q(.|s) available, define the centred score

    g~_t = grad log p(y_t|s_t) - sum_v q(v|s_t) grad log p(v|s_t)
         = grad_theta [ log p(y_t|s_t) - sum_v sg(q(v|s_t)) log p(v|s_t) ].

The loss is L = -(1/N) sum_i A_i sum_t [log p(y_it|s_it) - sum_v sg(q_it(v)) log p(v|s_it)],
the sum over v is over the full vocabulary, and sg is stop-gradient. The sum over v is
already computed inside the rematerialized chunks as the log-softmax, so the extra cost is one
[B, chunk, V] product per chunk.

Why this works.

1. Zero mean under q for any constant advantage. Since E_q[g~_t | s_t] = m - m = 0 with
   m = sum_v q(v|s_t) grad log p(v|s_t), a prefix-constant advantage c gives
   E_q[c g~_t | s_t] = 0 exactly. The mismatch term (b) is removed exactly.

2. Exact no-op on-policy. If q = p then sum_v p(v|s) grad log p(v|s) = 0 identically, so
   g~_t = grad log p(y_t|s_t) for every sample. The estimator is the unmodified REINFORCE
   estimator on-policy, and is unbiased there.

3. Residual off-policy bias is (q-p) times the within-prefix advantage spread. Write
   Q(s,v) = Abar(s) + D(s,v), with Abar(s) = sum_v q(v|s) Q(s,v) and sum_v q(v|s) D(s,v) = 0,
   and let m(s) = sum_v q(v|s) grad log p(v|s). Then

       E_q[A g~_t | s] = sum_v q(v) Q(s,v) grad log p(v) - Abar m
                       = sum_v q(v) D(s,v) grad log p(v),

   and the on-policy target is sum_v p(v) Q(s,v) grad log p(v) = sum_v p(v) D(s,v) grad log p(v)
   (the constant part integrates to zero under p). So the bias is exactly

       Bias(s) = sum_v (q(v|s) - p(v|s)) D(s,v) grad log p(v|s),

   which is first order in q - p and vanishes when the advantage is constant within a prefix
   (D = 0). It is not the systematic drift of point (b): that drift is c (q - p) with a constant
   c, and the centring removes it exactly.

Why the change is in the loss and not in the weights. Replacing the token weight by p/q
removes the bias (E_q[(p/q) A grad log p] = E_p[A grad log p] per token), but the variance
is then set by the sampler/trainer ratio and it was the worse option in the tests below.
Combining the centred score with p/q weights collapsed in our tests, and so did combining it
with TIS. Both break the cancellation in point 1, since the weight multiplies the centring term
too. The modification is therefore applied only when `correction is None`. With a correction
given, the loss is unchanged.

Limitation. The centring needs the full sampler distribution q(.|s), so the configuration sets
`sampler.vocab_logprobs=full`. A top-k truncation of q (k=8 in our test) dropped the tail mass
from the centring term and collapsed. Without the full distribution the loss reverts to plain
REINFORCE and gets no protection.

## 3. Evidence from the dev setting (not required, for orientation only)

Mini setting: 600 steps, plus_minus rewards, weight noise 0.3 or 0.6, 200 extra never-correct
tokens, mean reward over the last 50 steps:

| loss | noise 0.3 | noise 0.6 |
| --- | --- | --- |
| plain REINFORCE (seed 0 / seed 1) | 0.00 / 0.00 (collapsed) | not run |
| TIS (high=2, clamp) | 0.59 | 0.01 |
| naive p/q weight | 0.60 | 0.05 |
| centred score, full q (this submission) | 0.84 | 0.38 (seeds 0, 1, 2: 0.38, 0.34, 0.30) |

With no extra tokens and noise 0.3, the centred score gives 0.87 (last 50 steps).

## Configuration

sampler.vocab_logprobs=full

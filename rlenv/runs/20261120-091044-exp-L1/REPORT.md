# Sampler-centred REINFORCE under sampler/trainer mismatch

## 1. What is wrong with the current loss

Let p_θ be the trainer and q the sampler. The on-policy REINFORCE gradient is

    g_p = E_{y~p}[ A(y) Σ_t ∇θ log p(y_t | y_<t) ].

`loss_fn_rl` with `correction=None` estimates instead

    g_q = E_{y~q}[ A(y) Σ_t ∇θ log p(y_t | y_<t) ],

i.e. the same score function, averaged over the sampler's tokens. The
estimator is unbiased only when q = p.

Take one position with prefix s and write the logits as z. For a softmax head,

    ∇z log p(y|s) = e_y − p(·|s),

so, for a fixed advantage c that does not depend on the token drawn at this
position,

    E_{y~q}[ c (e_y − p) ] = c (q − p).

The baseline identity E[∇log p] = 0, which group-centring relies on, holds only
under p = q. Under mismatch, every constant part of the advantage contributes
c·(q − p) to the expected logit update. This term is:

- systematic: it is linear in the mismatch δ = q − p and does not average out
  over samples or steps;
- sign-dependent: for c > 0 it pulls the trainer toward the sampler, for c < 0
  it pushes the trainer away from it;
- strongest when many advantages share a sign, for example in all-failed
  groups (every rollout has the same reward and so the same advantage).

It compounds because the sampler is refreshed from the trainer at every step:
the bias is re-applied to the same trainer each step, and the sampler moves
with it, so the mismatch does not vanish. The bias is never removed by the
zero-mean argument that makes REINFORCE work on-policy.

Importance-sampling corrections only reweight the same term. With w = p/q the
constant part is exact: E_q[(p/q)(e_y − p)] = Σ_y p_y (e_y − p) = p − p = 0. Clipping replaces w by
w̃_v = min(p_v/q_v, B) or clamps it to a band, and then

    E_q[ c w̃ (e_y − p) ] = c ( Σ_v q_v w̃_v e_v − p Σ_v q_v w̃_v ) ≠ 0 in general.

The clipped mass is exactly the bias that remains, which is why stronger token
clipping can make things worse, and why TIS and MIS only slow the drift down.

## 2. The modification: a sampler-centred score

Replace the per-token score log p(y_t) by its centred version under the sampler:

    s_t = f(y_t) − Σ_{v} q_t(v) f(v),    f(v) = w_t(v) · log p_t(v),

with w_t(v) the detached IS weight of token v (w = 1 when `correction=None`).
The loss is

    L = − Σ_i Σ_t A_i · s_t / N_tokens.

Here the sum over v runs over the sampler's top-K tokens. K = 20 is already
returned by the sampler (`logprobs_topk=20` in `train_rl.py`), and the
trainer's log-softmax is computed anyway, so the cost is O(B·T·K).

**Why this is unbiased for constant advantages.** For any position with prefix s,

    E_{y~q}[ s_t | s ] = Σ_y q(y) f(y) − Σ_v q(v) f(v) = 0.

So for any advantage c that is constant over the token drawn at that position,
E_q[c · s_t] = 0 exactly, whatever the mismatch. Group-mean offsets and
all-failed batches therefore no longer inject c(q − p).

**Logit form (w = 1).** Since ∇z log p(v) = e_v − p, we get

    s_t = (e_y − p) − Σ_v q(v)(e_v − p) = (e_y − p) − (q − p) = e_y − q.

So the update is A · (e_y − q): the score of the sampled token, with the sampler
distribution as the centring distribution in place of the trainer's.

**On-policy check.** If q = p, then Σ_v p(v) ∇log p(v) = Σ_v ∇p(v) = 0, so the
expected gradient is unchanged; the modification only changes variance. It
does not change the on-policy estimator, so it cannot cost anything when there
is no mismatch.

**Scope.** The identity E_q[s_t | prefix] = 0 removes exactly the part of the
bias that comes from an advantage that is constant over the token drawn at a
position, and any part of the advantage that depends only on the prefix.
What remains is the dependence of A on the sampled token itself and on later
tokens. That dependence is the reward signal, and it enters only at first order
in the mismatch. Two approximations apply: the sum over v is truncated to the
sampler's top-K, so mass outside the top-K gives a residual term
Σ_{v∉topK} q(v)(e_v − p) that is small when the top-20 covers most of q; and
with a correction the weights w_t(v) are used on both terms, so the centring
stays exact for the weighted estimator.

## 3. Evidence (short; empirical results are not required)

Dev setting `dev/mini.py`, 600 steps, plus_minus rewards, lr 0.2, 1 seed unless
noted. "Original" is the loss before this change. `MINI_EXTRA_TOKENS` adds
tokens that never appear in a correct answer; the sampler's tail mass
collapses under the original loss.

| setting | original | sampler-centred |
| --- | --- | --- |
| noise 0.0, no extra tokens | 0.93 (last 10) | 0.93 |
| noise 0.3, 20 extra tokens (seed 0) | 0.00 (collapse) | 0.84 |
| noise 0.3, 20 extra tokens (seeds 1, 2) | 0.00 | 0.83, 0.83 |
| noise 0.3, 40 extra tokens | 0.00 | 0.81 |
| noise 0.5, 40 extra tokens | 0.00 | 0.52 |
| noise 0.3, 20 extra, TIS clamp high=2 | 0.56 (orig. TIS) | 0.60 |

With the original loss, the runs that collapse end with reward 0, gradient norm
near 0, sampler/trainer KL near 0 and sampler tail mass near 0 (observed in the
logs; the mechanism is the c(q − p) term above).

## 4. Configuration

No configuration changes are needed. The modification is on by default in
`loss_fn_rl`, uses only the batch keys `sampler_topk_ids` and
`sampler_topk_logprobs` that `train_rl.py` already supplies, and falls back to
the plain score if those keys are absent.

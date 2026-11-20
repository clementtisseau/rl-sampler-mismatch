# Sampler-mismatch stability: a sampler-centred score

## 1. What is wrong with the current loss

With `correction=None`, `loss_fn_rl` minimises

    L = -(1/N) sum_i A_i sum_t sg(1) * log p_theta(x_it | h_it),   x_it ~ q,

where p is the trainer, q the sampler, A_i the (group-centred) advantage of rollout i, h_it the prefix and N the number of completion tokens. Its gradient is an estimate of

    E_{x~q}[ A * grad log p(x) ],

but the policy gradient is E_{x~p}[ A * grad log p(x) ]. The two differ whenever q != p, and the difference is not zero-mean noise. Take logits z of the trainer at one context, so grad_z log p(x) = e_x - p, with e_x the one-hot vector. Hold the advantage Q(h,x) := E[A | prefix h, token x] fixed and write V = sum_x p(x) Q(h,x). The expected gradient on the logits at context h is

    U(h)  = sum_x q(x) Q(h,x) (e_x - p)          (what the code estimates)
    T(h)  = sum_x p(x) Q(h,x) (e_x - p)          (the true gradient)

Let Qbar be any constant. Since sum_x p(x)(e_x - p) = 0, a constant advantage contributes exactly zero to T, but not to U:

    U - T = Qbar (q - p) + (first order in the variation delta(x) = Q(h,x) - Qbar) * (q - p).

So the uncorrected loss contains a systematic term Qbar * (q - p) per context. If the mean advantage E[A | h] is positive at a context, the update pulls p toward q there; if negative, it pushes p away from q, toward the tokens where p > q. Group-centring makes the advantage average zero over a group, but not per prefix: E[A | h] is large at the answer positions, where the prefix already determines most of the outcome. The term therefore does not cancel.

Why it compounds and collapses: q is refreshed from p every step, so the bias is applied again to a new q. With plus/minus rewards most rollouts are unsolved (A = -1), so Qbar < 0 and the bias pushes mass toward tokens where p > q. In the runs this sharpened p onto a wrong answer. Once p is nearly deterministic, grad log p = e_x - p is close to 0, so the gradient vanishes and the run is absorbing. In the dev runs (dev/mini.py) this is visible as tail_mass_q and grad_norm falling to 0 while reward goes to 0.

Importance sampling removes the bias on the variation term only through the weight p/q: E_q[(p/q) A (e_x - p)] = T exactly. That is unbiased, but at strong mismatch the weights have very high variance and the run still collapses (naive token IS collapses; so do hybrids that multiply only the advantage part by p/q). Clipping (TIS, PPO-style bands) truncates these weights, which makes the estimator biased again, and it does not remove the Qbar (q - p) term in general.

## 2. The modification: subtract the sampler's expected score

Use the score relative to the sampler's expectation. In place of log p(x_t) use

    s_t = log p(x_t) - E_{y~q}[ log p(y) ] = log p(x_t) - sum_y q(y) log p(y).

The second term is a function of the context only. Its gradient in logit space is sum_y q(y)(e_y - p) = q - p, so the corrected score is grad_z s_t = e_{x_t} - q. The loss is

    L_cen = -(1/N) sum_i A_i sum_t [ log p(x_it|h_it) - sum_y q_t(y) log p(y|h_t) ],

with q_t the sampler's distribution at position t.

Derivation of unbiasedness for the constant part. Under x ~ q at a fixed context,

    E_q[ grad_z s_t ] = E_q[e_x - q] = q - q = 0   for any p,

so for constant A the expected update is exactly zero, whereas the uncorrected estimator gave A (q - p). Writing it with Q(h,x):

    U_cen(h) = sum_x q(x) Q(h,x) (e_x - q),
    U_cen - T = sum_x (q - p)(x) delta(x) e_x - (sum_x q delta) q + (sum_x p delta) p,

with delta = Q - Qbar. The constant-advantage bias Qbar (q - p) is gone exactly. What remains is proportional to the variation delta of the advantage across tokens at the same context, times the mismatch q - p. It vanishes unless the mismatch is nonzero and the advantage depends on the token choice, which is exactly the signal learning uses; it is not a constant push toward q. Group centring cannot remove the per-prefix mean; the centring of the score does.

Why this uses no IS weight: the centring replaces the weight by a zero-mean baseline built from q, which the sampler already provides. It does not need p/q, so it does not inherit the variance of IS, and it is exact in expectation for any q, including stale weights, quantised weights and quantised KV caches, because it only uses the sampler distribution that was actually used for sampling.

Implementation. The sampler returns its top-k logprobs per position (`sampler_topk_ids`, `sampler_topk_logprobs`, k = 20 in train_rl.py). We renormalise them to q_k = softmax over the top-k, and compute

    E_q[log p] ~= sum_k q_k log p(id_k),

which is exact when the top-k holds almost all sampler mass. The residual tail mass epsilon adds a context-only term of order epsilon * E[log p | tail], which is small for realistic k. If the top-k arrays are absent, the loss falls back to the plain score.

## 3. Correction and configuration

- With `correction=None` (the default) the sampler-centred score is used.
- With any `ISCorrection` the loss is the unchanged IS loss. Centring plus a detached weight is not mean-zero under q (E_q[w (log p - E_q log p)] != 0), and this combination collapsed in tests, so the centring is turned off when a correction is supplied.

No configuration changes are needed; the loss is active under the default configuration.

## Configuration

None. The modification is active with the default `rl.is.*` settings (no correction).

## 4. Evidence (dev/mini.py, 600 steps, plus_minus reward)

Mean reward over the last 10 steps (peak in parentheses); "COLL" means the dev summary verdict was collapse.

| setting | plain PG | plain token-TIS | sampler-centred |
| --- | --- | --- | --- |
| noise 0.3, default vocab, seed 0 | 0.578 (0.675) | 0.617 (0.694) | 0.837 (0.944) |
| noise 0.3, default vocab, seed 1 | 0.606 (0.672) | - | 0.821 (0.916) |
| noise 0.3, +200 tail tokens | COLL (0.33) | 0.588 (0.667) | 0.848 (0.905) |
| noise 0.6, default vocab, seed 0 | COLL (0.25) | - | 0.424 (0.569) |
| noise 0.6, +200 tail tokens, seed 0 | COLL (0.04) | - | COLL (0.04) |
| noise 0.6, +200 tail tokens, seeds 1 / 2 | not run / COLL (0.16) | - | 0.256 / 0.272 |

Under group_centered rewards at noise 0.6 with 200 tail tokens: plain 0.281, centred 0.344 (mean of last 10). Tested and rejected: centring combined with TIS (collapses at noise 0.3 and 0.6), naive token IS at noise 0.6 with 200 tail tokens (collapses), and an unbiased IS-plus-centring hybrid (collapses).

The centring is not a complete fix: at noise 0.6 with a 200-token tail, one seed still collapses. The second-order residual above remains there, and the runs remain sensitive to the mismatch size.

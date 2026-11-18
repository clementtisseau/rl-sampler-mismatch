# Drift-free REINFORCE under sampler/trainer mismatch

## 1. Setup

Rollouts y_1..y_T are drawn from the sampler q; the trainer is p = p_theta.
For a completion with (group-centered) advantage A, the current loss is

    L = -(1/N) sum_i A_i sum_t w_{i,t} log p(y_{i,t} | h_{i,t})

(w = 1 without correction, w = w(p/q) detached with an `rl.is.*` correction).
Its per-token update direction is A * s_y, with the score
s_v = grad_theta log p(v|h).

## 2. What is wrong: a drift term that only exists off-policy

Fix a prefix h and take the expectation over the next token y ~ q(.|h).
Write Abar(h) = E_q[A | h] and Abar(h, v) = E_q[A | h, y = v] (the continuation
after v is also drawn from q). Then

    E_q[A s_y | h] = sum_v q(v|h) Abar(h,v) s_v
                   = Abar(h) * sbar(h)  +  Cov_q(A, s_y | h),
    sbar(h) = sum_v q(v|h) grad log p(v|h).

On-policy (q = p): sbar(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h)
= grad 1 = 0, so the update is pure covariance: it moves the probability of v
only according to whether v is better or worse than average at h.

Off-policy, sbar(h) != 0. Because q is held fixed (detached),

    sbar(h) = grad_theta sum_v q(v|h) log p(v|h) = - grad_theta KL(q(.|h) || p(.|h)).

So the first term is Abar(h) * (-grad KL(q||p)):

- at prefixes whose expected outcome is good (Abar > 0) the update pulls p
  toward the sampler q;
- at prefixes whose expected outcome is bad (Abar < 0) it pushes p away from
  q, i.e. increases KL(q||p) without bound (tokens the sampler likes more than
  the trainer are pushed further down, which further increases the mismatch).

This term has nothing to do with which token was better: it is present even
if the reward is independent of the token choice. It is a bias of the
expected update, not zero-mean noise, so it does not average out over a batch.

**Logit picture.** At one position with logits z, grad_z log p(v) = e_v - p.
Hence, in logit coordinates,

    E_q[A grad_z log p(y) | h] = Abar(h) (q - p)  +  [q_v (Abar(h,v) - Abar(h))]_v .
                                 ^ drift            ^ signal

The signal is weighted by how much the token choice at h changes the outcome.
For most positions of a long completion, especially late ones, the outcome
is already nearly decided: Abar(h,v) ~ Abar(h) for every likely v, so the
signal is ~0 while |Abar(h)| ~ |A| is large. There the drift
Abar(h)(q - p) is essentially the whole update, and it sums over all
completion tokens. Group centering only makes Abar = 0 at the prompt
(first token); every later prefix is unique to one completion, so
Abar(h) ~ A there.

**Why it compounds.** The sampler is refreshed from the trainer every step:
q = q_theta is a fixed deterministic map of p (weights + noise,
quantization, staleness). The drift therefore points in a consistent,
theta-dependent direction at every step instead of a random one; it
accumulates over steps, KL(q||p) grows, the drift (proportional to q - p)
grows with it, and training collapses.

## 3. Why IS corrections and clipping only slow it down

With a detached weight w_v = w(p_v/q_v) the update is A w_y s_y and

    E_q[A w_y s_y | h] = Abar(h) sbar_w(h) + Cov_q(A, w_y s_y | h),
    sbar_w(h) = sum_v q_v w_v s_v = sum_v m_v s_v,   m_v = q_v w(p_v/q_v),

(m is exactly `log_mass` in losses.py). In logits: sbar_w = m - (sum_v m_v) p.

- Untruncated token IS (w = p/q) gives m = p and sbar_w = 0: the drift vanishes
  in expectation, but it is cancelled by a single-sample estimate with weights
  p/q, which are heavy-tailed exactly where the mismatch is large (tokens
  with q << p), so variance is large.
- Any clamp or drop gives m != p, so sbar_w != 0 and the drift
  Abar(h)(m - |m| p) remains. Dropping tokens outside a narrow band removes
  their mass from m, so m - |m| p can be *larger* than q - p: this is why
  stronger token-level clipping can make things worse. PPO-style sign-dependent
  bands use different m for Abar > 0 and Abar < 0, so the two sides do not even
  cancel each other.
- Sequence-level weights multiply A by a scalar; they rescale but do not
  remove the per-position term Abar(h) sbar(h).

## 4. The fix: center the score under the sampler, exactly

Replace the score of the sampled token by its sampler-centered version:

    s_y  ->  w_y s_y - sbar_w(h),     sbar_w(h) = sum_v m_v(h) grad log p(v|h),

with m_v = q(v|h) w(p_v/q_v) (just m = q with no correction). In loss form
(this is what `losses.py` implements, all weights detached):

    L = -(1/N) sum_i A_i sum_t [ w_{i,t} log p(y_{i,t}|h_{i,t})
                                 - sum_v sg(m_v(h_{i,t})) log p(v|h_{i,t}) ].

**Derivation.** sbar_w(h) depends only on the prefix, not on y, so

    E_q[A (w_y s_y - sbar_w(h)) | h]
      = sum_v q_v w_v Abar(h,v) s_v - Abar(h) sum_v q_v w_v s_v
      = sum_v m_v (Abar(h,v) - Abar(h)) s_v.

Without correction this is exactly Cov_q(A, s_y | h): the drift term
Abar(h) sbar(h) is removed exactly, at every prefix, for every advantage
sign, with no sampling variance (it is computed as an exact expectation over
the vocabulary, a Rao-Blackwellized control variate, instead of being cancelled
by noisy importance weights). In logits the update becomes
q_v (Abar(h,v) - Abar(h)), which is nonzero only through differences of
advantage between tokens.

Properties:

1. **On-policy it is plain REINFORCE.** With q = p and no correction,
   sbar = 0 identically, so the gradient is unchanged (checked numerically:
   relative difference ~1e-7). The modification only acts on the mismatch.
2. **No update from uninformative rewards.** If A is independent of the token
   chosen at h, Abar(h,v) = Abar(h) and the expected update at h is zero for
   *any* q. Plain REINFORCE would still move p by Abar(h)(q - p). The term that
   compounded through the sampler refresh is gone, so there is no systematic
   direction left to accumulate; KL(q||p) is no longer pushed up at bad
   prefixes.
3. **What remains** is the covariance taken under q instead of p:
   tokens are weighted by q_v instead of p_v and Abar(h,v) is evaluated with
   q-continuations. This is a multiplicative, first-order reweighting of the
   *signal* (a descent direction for q close to p), and it vanishes together
   with the signal; it does not create motion on its own. If unbiasedness
   with respect to p is wanted, an IS correction can be added on top: the
   centering then uses m = q w and still removes the drift of the clipped
   estimator (Section 3), which is why it composes with any `rl.is.*` point.
4. **Cost.** One extra weighted sum over the vocabulary per position on the
   already materialized log-softmax chunk; no extra forward pass.

## 5. What the sampler must record, and approximations

sbar needs the sampler's next-token distribution, which the sampled-token
logprob alone does not give. The loss reads (and forwards via `sampler_keys`):

- `sampling_vocab_logprobs` [B,T,V] (`sampler.vocab_logprobs=full`): exact.
- `sampler_topk_ids` / `sampler_topk_logprobs` [B,T,k]
  (`sampler.vocab_logprobs=k`): exact on the top-k set K. The rest
  R = V \ K is one bucket with q(R) = 1 - sum_K q and p(R) = 1 - sum_K p, and
  score grad log p(R) = sum_{v in R} p_v s_v / p(R). Hence

      sbar(h) ~ sum_{v in K} q_v s_v + q(R) grad log p(R),

  exact whenever q is proportional to p inside R; the error is
  q(R) (E_{q|R}[s] - E_{p|R}[s]), a product of the tail mass and the mismatch
  inside the tail. K is chosen by the sampler from h alone, so the estimate
  is still independent of y and remains a valid control variate.
  The bucket matters: tokens that the sampler over-weights relative to the
  trainer (e.g. junk tokens under weight noise) are exactly where the drift
  pushes p(R) down at bad prefixes.
- Nothing recorded: the only per-prefix estimate of sbar available from the
  sampled token is the single-sample one, E_q[(1 - p_y/q_y) s_y | h] = sbar(h),
  and plugging it in gives A (p_y/q_y) s_y, i.e. untruncated token IS. The loss
  falls back to that (when `correction=None`), which removes the drift in
  expectation but with IS variance; the recorded distribution is what makes
  the removal exact. (The estimate must not use the sampled token in any other
  way: a y-dependent "baseline" is not a control variate and would bias the
  signal.)

## 6. Check on the CPU dev setting (dev/mini.py, noise 0.3, 600 steps, seed 0)

| loss | last eval reward | mean train reward, last 10 steps |
| --- | --- | --- |
| original REINFORCE | 0.663 (peak 0.718) | 0.578 |
| centered, `vocab_logprobs=full` | 0.830 (peak 0.858) | 0.845 |
| centered, `vocab_logprobs=4` (of 12 tokens; tail bucket) | 0.818 (peak 0.849) | 0.812 |

With 50 extra never-correct tokens (`MINI_EXTRA_TOKENS=50`, vocabulary 62,
seed 1; eval reward every 50 steps):

| loss | eval reward, steps 0..599 | mean train reward, last 50 |
| --- | --- | --- |
| original REINFORCE | 0.38, 0.19, 0.00, ..., 0.00 (collapse) | 0.00 |
| original + TIS (high=2, clamp) | 0.39 -> 0.76 | 0.61 |
| fallback, nothing recorded (= token IS) | 0.38 -> 0.71, dips to 0.27, 0.72 | 0.64 |
| centered, top-8 + tail bucket | 0.38 -> 0.83, nearly monotone | 0.85 |
| centered, top-8, seed 2 | 0.43 -> 0.86 | 0.84 |
| centered, top-8, with TIS | 0.38 -> 0.83 | 0.68 |

Plain REINFORCE collapses (the drift pushes the trainer away from the sampler
at bad prefixes), the noisy single-sample cancellation (token IS) is unstable,
and the exact centering trains steadily without collapse.

## Configuration

sampler.vocab_logprobs=64

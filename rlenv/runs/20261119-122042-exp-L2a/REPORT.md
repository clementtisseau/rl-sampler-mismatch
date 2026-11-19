# Loss under sampler/trainer mismatch

## What is wrong
Rollouts y ~ q, trainer is p. The loss is -sum_t A logp(y_t), whose expected gradient is
E_q[A ∇logp]. For one position, in logit space ∇_z logp(y) = e_y - p, so
E_{y~q}[∇_z logp(y)] = q - p.

- On-policy (q=p): E_p[∇logp] = sum_v ∇p(v) = 0. The score is mean-zero, so any
  baseline b (group centering, +-1 rewards, length-weighting of tokens) multiplies zero:
  E_p[(R-b)∇logp] = E_p[R ∇logp].
- Off-policy (q≠p): the same baseline term becomes -b·E_q[∇logp] = b·(p-q) per token
  (in logit space), i.e. b·∇KL(q||p)-like drift. It is a systematic bias, not noise: its
  sign is set by b (negative-advantage samples dominate, b<0 on average once tokens are
  weighted by length, since failures are long), it pushes p toward moving mass away from q's
  support (sampled tokens are pushed down, the mass goes to unsampled/tail tokens), and as
  the sampler is rebuilt from p each step it accumulates -> collapse.
- Clipping/dropping IS weights (TIS, MIS, PPO...) uses w = min(p/q, c) etc., and then
  E_q[w ∇logp] = -sum_{v: p/q>c}(p - c q)∇logp(v) ≠ 0: the zero-mean-score property is
  broken exactly on the tokens where mismatch is largest, so the bias remains (stronger
  clipping = larger residual).

## Fix
Make the expected update under q equal to the on-policy one, i.e. keep E_q[score]=0.

1. Exact, unclipped token-level importance weight w_t = p(y_t)/q(y_t) (detached; only
   log-ratio clipped to e^{±20} for overflow) applied by default (`correction=None` is
   turned into `ISCorrection()`: level=token, band [0,inf), no drop). Then
   E_q[w ∇logp] = sum_v q (p/q) ∇logp = sum_v ∇p(v) = 0 exactly, as on-policy, so all
   baselines are again harmless, and E_q[A w ∇logp] = E_p[A ∇logp] at each position given the prefix.
2. Token-weighted centering of advantages (sum over completion tokens of A = 0) so that the
   residual baseline term b·(q-p) that remains (prefix mismatch, ratio clipping at e^±20,
   finite-sample) averages out. This also removes dependence on the reward mode
   (plus_minus vs group_centered).

The cost is variance (unbounded ratios), traded for zero bias; centering tames it in practice.

## Configuration
rl.is.level=token
rl.is.high=inf
rl.is.low=0

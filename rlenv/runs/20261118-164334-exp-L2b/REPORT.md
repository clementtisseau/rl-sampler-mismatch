# Sampler/trainer mismatch in REINFORCE

## What is wrong
At one position, given the prefix, let p be the trainer's softmax with logits z, and q the sampler's distribution. With
group-centered advantages, a sampled token x gets advantage A(x) (it is the same A for every token of a rollout). The current loss
(correction=None) has expected logit-gradient
E_{x~q}[A(x) ∇_z log p(x)] = Σ_x q(x) A(x)(e_x − p).
Write A(x) = c + Ã(x), where c is the q-weighted mean advantage at this position (given the prefix).
The c term gives c·Σ_x q(x)(e_x − p) = c (q − p).

- On-policy (q = p), this is zero, because E_p[∇log p] = 0. This is why baselines and group centering are free.
- When q ≠ p it does not vanish. It is c·∇_z(−KL(q‖p)), so it adds a systematic force of size c(q−p). It pulls p toward q
  when c>0 and pushes it away from q when c<0. This is a bias, not zero-mean noise. Group centering makes the mean advantage
  zero over the whole group, but not for each prefix or token position, so c ≠ 0 in general.
- The sampler is rebuilt from the trainer every step. The bias therefore moves p, which moves q, which changes the bias.
  It compounds, and training can collapse.

## Fix
Weight each token by the detached ratio w = p(x)/q(x), computed from the stored sampler logprobs.
E_q[w A (e_x − p)] = Σ_x p(x) A(x)(e_x − p).
This is exactly the on-policy expectation. The baseline term becomes c Σ_x p(x)(e_x − p) = 0 for every q, so the bias is gone.
Equivalently, E_q[(p/q) A ∇log p] = Σ_x A ∇p, and Σ_x ∇p = 0.

Clipping or dropping tokens (TIS, MIS, PPO, DPPO) replaces w by w̃(x) = min(p,Kq)/q. Then
Σ_x q w̃ (e_x − p) = Σ_x min(p, Kq)(e_x − p) ≠ 0. The c term comes back, as c·(clipped mass defect), so these methods leave a residual
bias. Tighter clipping gives a larger residual, which is why stronger token-level clipping can make things worse.

So `loss_fn_rl` with the default `correction=None` now uses an unbiased token IS weight p/q. Its clamp is only at a ratio of 20, for numerical safety.
An explicitly passed correction still behaves as before.

Caveat. The per-position analysis is conditional on the prefix. The prefix distribution itself is q-distributed, and the code does
not correct for that. This is a smaller second-order effect than the baseline leak above, and the leak is the compounding term.

## Configuration

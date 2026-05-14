# GRPO design sketch — news-conditioned TimesFM 2

> Stretch piece per Singh's Apr-28 conversation: "this you can do through fine
> tuning or this you can do through GRPO." Parts 1–8 of the plan delivered
> the supervised fine-tuning route end-to-end (frozen TimesFM 2 + small
> adapter head, MSE loss in return space). This document outlines what the
> same problem looks like as RL with Group Relative Policy Optimization, and
> sketches the minimal code skeleton to make it runnable later.

## Why GRPO at all

Supervised MSE on adjusted returns is a strong baseline but it has two
problems Singh flagged implicitly:

1. **The reward is non-trivially shaped.** A forecast error of 0.5% means
   different things in a 0.5%-vol week vs a 5%-vol week. The desirable
   loss surface is closer to "did the conditioned forecast outperform the
   vanilla one on this day, given regime?" than to a flat MSE.
2. **Single-day MSE has no notion of trajectory consistency.** A forecast
   that gets the path roughly right but the sign wrong on one step gets the
   same penalty as one that's slightly more diffuse but directionally
   correct. We care more about the second.

GRPO's reward-and-baseline framing handles both: we score whole forecast
trajectories with a reward function we control, and the baseline is the
*group mean* of rewards from sampled rollouts at the same prompt — so the
adapter learns to produce forecasts that are *better than its own samples*
rather than to hit a fixed target.

## Mapping the problem to GRPO

| GRPO concept            | News-conditioning equivalent                                                  |
|-------------------------|-------------------------------------------------------------------------------|
| Prompt / state `s_t`    | (price history, news embedding, history features) at time `t`                 |
| Policy `π_θ(a \| s_t)`  | The adapter, but stochastic — outputs a distribution over H-step adjustments  |
| Action `a_t`            | An H-step return adjustment vector, sampled from π                            |
| Group of rollouts       | K samples of the adjustment per state, scored independently                   |
| Reward `r(s_t, a_t)`    | A function of the resulting forecast vs the actual prices over `[t, t+H]`     |
| Reference policy `π_0`  | The supervised-trained adapter from Parts 5–6                                 |
| Update                  | Group-relative advantage × log-prob, with KL pull toward π_0                  |

The crucial difference from supervised: π is now a *distribution*, not a
deterministic mapping. The cleanest implementation is to make the adapter
output `(μ, σ)` for each horizon step (so 2H outputs instead of H) and
sample from `N(μ, σ)`, soft-clipping the sample with the same tanh bound
the supervised adapter already uses.

## Reward function

Pick something more informative than MSE-vs-actual. A good first reward is
**directional gain on top of vanilla**:

```
r(a) = α · sign_match_bonus(a) + β · mae_reduction(a) − γ · ||a||²
```

where:

- `sign_match_bonus(a)` = mean over horizon steps of
  `1[sign(adjusted_return) == sign(actual_return)] − 1[sign(vanilla_return) == sign(actual_return)]`
  — we pay only the *delta* over vanilla, so the adapter is rewarded for
  improving directional accuracy, not for absolute correctness it inherits
  from TimesFM.
- `mae_reduction(a)` = MAE(vanilla, actual) − MAE(adjusted, actual). Same
  delta-over-vanilla framing.
- `||a||²` is the standard adjustment-magnitude penalty, identical to the
  one in the supervised loss.

Hyperparameters `α, β, γ` are dimensionless ratios; we'd start with
`α=1.0, β=10.0, γ=0.001` so the MAE term dominates but directional accuracy
gets nontrivial weight.

## GRPO update

For each prompt `s_t`, sample `K` adjustments `a_1, …, a_K`. Compute
rewards `r_1, …, r_K`. Group baseline:

```
b = mean(r_k for k in 1..K)
A_k = r_k − b      # group-relative advantage
```

Policy gradient with KL penalty against the reference (supervised) policy:

```
L = − E_k [ A_k · log π_θ(a_k | s_t) ]  +  λ_KL · KL(π_θ || π_0)
```

This is the DeepSeek-R1 GRPO objective applied to a continuous-action
adapter. Because `K` is small (we'd start with 8) and the policy is a tiny
MLP, the wall-clock cost per gradient step is roughly `K ×` the supervised
cost — which is fine.

## Why initialize from the supervised checkpoint

Two reasons:

1. **Variance reduction.** A fresh-init stochastic policy will produce
   wild adjustments that mostly hurt; group-relative advantages still
   work, but the early KL penalty essentially keeps the model still while
   it explores noise. Starting from the supervised optimum collapses the
   "dumb-exploration" phase.
2. **Stability of the KL anchor.** π_0 needs to be a reasonable policy or
   the KL term anchors to nonsense. The supervised adapter already
   produces near-identity adjustments at small magnitude — the right prior.

## Code skeleton (minimal, illustrative)

```python
# backend/conditioning/grpo.py — sketch only, not implemented yet
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributions as D


class StochasticAdapter(nn.Module):
    """Policy head that outputs (mu, log_sigma) for each horizon step."""
    def __init__(self, base_cfg):
        super().__init__()
        self.cfg = base_cfg
        in_dim = base_cfg.total_input_dim()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, base_cfg.hidden), nn.GELU(),
            nn.Linear(base_cfg.hidden, base_cfg.hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(base_cfg.hidden, base_cfg.horizon)
        self.log_sigma_head = nn.Linear(base_cfg.hidden, base_cfg.horizon)

    def dist(self, vanilla, news, hist):
        x = torch.cat([vanilla, news, hist], dim=-1)
        h = self.trunk(x)
        mu = self.cfg.adjust_clip * torch.tanh(self.mu_head(h))
        log_sigma = self.log_sigma_head(h).clamp(-5, 1)
        return D.Independent(D.Normal(mu, log_sigma.exp()), 1)


def grpo_step(policy, ref_policy, batch, K=8, kl_coef=0.05):
    vanilla, news, hist, target = batch
    pi = policy.dist(vanilla, news, hist)        # current
    pi_ref = ref_policy.dist(vanilla, news, hist) # frozen supervised

    # K samples per prompt
    samples = pi.rsample((K,))                    # (K, B, H)
    log_probs = pi.log_prob(samples)              # (K, B)

    # Reward: delta over vanilla on the realized target
    adjusted = vanilla.unsqueeze(0) + samples
    abs_err_v = (vanilla.unsqueeze(0) - target.unsqueeze(0)).abs().mean(-1)
    abs_err_a = (adjusted - target.unsqueeze(0)).abs().mean(-1)
    sign_match_v = (vanilla.unsqueeze(0).sign() == target.unsqueeze(0).sign()).float().mean(-1)
    sign_match_a = (adjusted.sign() == target.unsqueeze(0).sign()).float().mean(-1)
    reward = (sign_match_a - sign_match_v) * 1.0 \
           + (abs_err_v - abs_err_a) * 10.0 \
           - samples.pow(2).mean(-1) * 0.001
    # group-relative
    baseline = reward.mean(0, keepdim=True)
    advantage = reward - baseline                  # (K, B)

    pg_loss = -(advantage.detach() * log_probs).mean()
    kl = D.kl_divergence(pi, pi_ref).mean()
    return pg_loss + kl_coef * kl
```

That's the whole RL loop in ~30 lines. The infrastructure we need to wrap
around it is minimal because Parts 1–8 already handle data alignment,
encoding, vanilla forecasting, and replay scoring.

## What to do first

When we revisit this:

1. Implement `StochasticAdapter` and a checkpoint loader that hot-starts it
   from the supervised adapter (initialize `mu_head` from the supervised
   output layer, initialize `log_sigma_head` to a small constant).
2. Wire `grpo_step` into the existing training harness (the cache from
   Part 6 already provides `vanilla`, `news`, `hist`, `target`).
3. Run a 1k-step GRPO pass on top of the supervised checkpoint and re-run
   the diagnostics HTML. Gate on whether the win rate moves up vs the
   supervised model.

If the supervised baseline is already strong enough to satisfy Singh, this
work is a win-rate bump, not a structural change. Worth doing when the
data side scales beyond SPY-only.

## Risks & open questions

- **Reward hacking.** With our reward shape, the policy could learn to
  shrink σ to zero and replay vanilla exactly (`reward → 0` deterministically
  but no penalty either). Mitigations: an entropy bonus, a floor on σ, and
  the KL pull toward the ref policy.
- **Compute.** `K=8` rollouts × supervised cache size × 1k steps is ~30
  minutes on Sam's M-class Mac. Acceptable. GRPO at K=32 over a multi-channel
  universe will need a small GPU budget.
- **Reference for KL.** Reusing the supervised adapter is fine while the
  policy parameter shape matches. If we change the head structure (e.g.,
  add (μ, σ)), we need to copy the supervised weights into μ-head only and
  initialize σ-head fresh — handled in step 1 above.

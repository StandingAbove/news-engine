"""News-conditioning adapter (Part 5).

Architecture
------------
TimesFM 2 is treated as a frozen forecaster. The conditioning happens in a
small trainable head that takes:

    (vanilla_forecast, news_embedding, history_summary)  →  adjustment

and returns an additive adjustment in *return space* that we apply to the
vanilla forecast path. Singh's framing: "synthesize information from the news
feeds to something that can adjust the behavior of TimesFM 2."

We adjust in return-space (not raw price) so the magnitude of the adjustment
is invariant to the absolute price level, which makes training stable across
the 33-year SPY history.

Why "LoRA-style"?
-----------------
We don't actually inject low-rank adapters into TimesFM's transformer (that
would require touching the foundation-model internals on the 2.5 main-branch
API, which is brittle). Instead we keep TimesFM frozen and learn a small
*post-hoc* adjustment network. Same spirit — small trainable head, frozen
foundation — without the integration risk.

Inputs to the adapter
---------------------
* ``vanilla_returns`` — H-step return path predicted by frozen TimesFM 2.
* ``news_emb``        — D-dim vector from ``StatsEncoder`` or ``TextEncoder``.
* ``hist_features``   — small set of history summary stats (vol, recent
  drift, last 1d/5d returns) to give the adapter local market context.

Output
------
An H-dim adjustment that gets added to ``vanilla_returns``. The adjusted
return path is then compounded forward from the seed price (close on day t)
to produce the conditioned price forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore


# --- history feature builder --------------------------------------------------
HIST_FEATURE_NAMES: List[str] = [
    "ret_1d",            # 1-day return at t-1
    "ret_5d",            # 5-day cumulative return
    "ret_20d",           # 20-day cumulative return
    "vol_5d",            # std of last 5 daily returns
    "vol_20d",           # std of last 20 daily returns
    "drawdown_60d",      # drawdown vs trailing 60-day max
    "trend_slope_60d",   # OLS slope on last 60 closes (normalized)
    "log_close_z",       # last close z-scored over last 252 days
]


def history_features(history: np.ndarray) -> np.ndarray:
    """Compute a small fixed-size summary of price history. Pure numpy so the
    adapter can evaluate cheaply during training and inference."""
    h = np.asarray(history, dtype=np.float64)
    if h.size < 60:
        # Not enough context for the 60-day features. Return zeros so the
        # walker's own min-history guard is the source of truth, not this.
        return np.zeros(len(HIST_FEATURE_NAMES), dtype=np.float32)

    rets = np.diff(h) / h[:-1]
    last = h[-1]
    out = np.zeros(len(HIST_FEATURE_NAMES), dtype=np.float64)
    out[0] = rets[-1]
    out[1] = h[-1] / h[-6] - 1.0      # 5-day cum
    out[2] = h[-1] / h[-21] - 1.0     # 20-day cum
    out[3] = float(rets[-5:].std(ddof=0))
    out[4] = float(rets[-20:].std(ddof=0))
    out[5] = last / h[-60:].max() - 1.0
    # OLS slope on last 60 days, normalized by mean price so it's scale-free.
    x = np.arange(60, dtype=np.float64)
    y = h[-60:]
    slope = float(np.polyfit(x, y, 1)[0])
    out[6] = slope / max(1e-9, float(y.mean()))
    # Last close z-score over last 252 days.
    window = h[-252:] if h.size >= 252 else h
    mu, sd = float(window.mean()), float(window.std(ddof=0))
    out[7] = (last - mu) / max(1e-9, sd)

    return out.astype(np.float32)


# --- adapter network ---------------------------------------------------------
@dataclass
class AdapterConfig:
    horizon: int = 20
    news_dim: int = 16
    hist_dim: int = len(HIST_FEATURE_NAMES)
    hidden: int = 64
    n_hidden_layers: int = 2
    dropout: float = 0.10
    # We softly clip the adjustment so the adapter can't fly off — the
    # conditioned forecast at any horizon step is bounded to within
    # ``adjust_clip`` (in return units) of the vanilla path.
    adjust_clip: float = 0.02   # 2% of vanilla return per step, soft-tanh
    # Multi-channel: K * 8 when the multi-channel pipeline is active.
    # Zero means single-channel (SPY-only); backward-compatible default.
    multi_channel_dim: int = 0

    def total_input_dim(self) -> int:
        return self.horizon + self.news_dim + self.hist_dim + self.multi_channel_dim


if _TORCH_OK:
    class ConditioningAdapter(nn.Module):
        """Tiny MLP head that produces an H-step return adjustment."""

        def __init__(self, cfg: AdapterConfig):
            super().__init__()
            self.cfg = cfg
            in_dim = cfg.total_input_dim()
            layers: List[nn.Module] = []
            prev = in_dim
            for _ in range(cfg.n_hidden_layers):
                layers += [nn.Linear(prev, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
                prev = cfg.hidden
            layers += [nn.Linear(prev, cfg.horizon)]
            self.net = nn.Sequential(*layers)
            # Initialize the final layer to ~0 so the adapter is a near-identity
            # at training start (i.e., conditioned ≈ vanilla initially).
            with torch.no_grad():
                self.net[-1].weight.mul_(0.01)
                self.net[-1].bias.zero_()

        def forward(
            self,
            vanilla_returns: "torch.Tensor",
            news_emb: "torch.Tensor",
            hist_feats: "torch.Tensor",
            multi_channel_feats: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            """Inputs:
              vanilla_returns    : (B, H)
              news_emb           : (B, D_news)
              hist_feats         : (B, D_hist)
              multi_channel_feats: (B, K*8)  optional; only when multi_channel_dim > 0
            Returns:
              adjustment: (B, H), bounded by ±cfg.adjust_clip
            """
            parts = [vanilla_returns, news_emb, hist_feats]
            if multi_channel_feats is not None and self.cfg.multi_channel_dim > 0:
                parts.append(multi_channel_feats)
            x = torch.cat(parts, dim=-1)
            raw = self.net(x)
            # Soft-clip with tanh — keeps gradients alive at the boundary.
            return self.cfg.adjust_clip * torch.tanh(raw)

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


else:  # pragma: no cover
    class ConditioningAdapter:  # type: ignore[no-redef]
        """Stub for environments without torch — instantiation fails clearly."""
        def __init__(self, *_a, **_kw):
            raise RuntimeError("PyTorch not available; install torch to use the adapter.")


# --- inference helper (pure numpy) -------------------------------------------
def apply_adjustment(
    vanilla_returns: np.ndarray,
    adjustment: np.ndarray,
    seed_price: float,
) -> np.ndarray:
    """Compose vanilla + adjustment in return space and return the conditioned
    price path of length H.

    All inputs are 1-D length-H numpy arrays. Used during inference / replay
    when we don't want to spin up a torch tensor for a single forecast.
    """
    cond_ret = vanilla_returns + adjustment
    # Compound forward from seed_price. Step k price = seed * Π(1 + ret_i).
    growth = np.cumprod(1.0 + cond_ret)
    return (seed_price * growth).astype(np.float32)

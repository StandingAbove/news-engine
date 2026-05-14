"""Supervised fine-tuning loop for the conditioning adapter (Part 6).

Pipeline
--------
1. Take an aligned dataset (Part 2 output) and a vanilla forecaster (Part 4
   wrapper around frozen TimesFM 2 — or NaiveLast for sanity).
2. For every sample t in the train half: get the vanilla return path, the
   news embedding (Part 3), and the history features (Part 5). Cache them.
3. Train the adapter (Part 5) to minimize MSE between
   ``vanilla_returns + adjustment`` and the actual forward returns.
4. Validate on a held-out tail of the train half. Early stop when val loss
   stops improving. Persist the best checkpoint.

Why we cache the vanilla forecasts
----------------------------------
The expensive step is calling TimesFM 2 — ~1 s per sample on CPU, ~250 ms on
GPU. With ~7000 train samples that's hours per epoch if recomputed. We
compute it *once* up front, write it to disk, and the training loop is then
pure-tensor and fast.

Calling convention
------------------
The training entry point is :func:`train_adapter`, which takes the path to
the .npz dataset, an instance of a ``Forecaster`` (used only for caching),
an instance of a ``NewsEncoder`` (also used only for caching), and a
``TrainConfig``. It writes the best checkpoint to disk and returns a
:class:`TrainResult` with per-epoch val/train losses.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False
    torch = None  # type: ignore

from .adapter import (
    AdapterConfig, ConditioningAdapter, HIST_FEATURE_NAMES, history_features
)
from .alignment import AlignedSample, NewsRecord, walk_aligned, AlignmentConfig
from .baseline import Forecaster, NaiveLastForecaster
from .embed import NewsEncoder, StatsEncoder
from .data import load_cached, load_spy
from ..core.config import PROJECT_ROOT

log = logging.getLogger(__name__)


# --- cache --------------------------------------------------------------------
@dataclass
class TrainCache:
    """Pre-computed tensors keyed by sample index."""
    t: np.ndarray              # (N,) int64 ns
    vanilla_returns: np.ndarray  # (N, H) frozen-forecast return paths
    news_emb: np.ndarray       # (N, D_news)
    hist_feats: np.ndarray     # (N, D_hist)
    target_returns: np.ndarray # (N, H) actual close-to-close returns
    # Optional: (N, K*8) multi-channel history features.  None = single-channel.
    multi_channel_feats: Optional[np.ndarray] = None

    def save(self, path: Path) -> None:
        arrays: dict = dict(
            t=self.t,
            vanilla_returns=self.vanilla_returns,
            news_emb=self.news_emb,
            hist_feats=self.hist_feats,
            target_returns=self.target_returns,
        )
        if self.multi_channel_feats is not None:
            arrays["multi_channel_feats"] = self.multi_channel_feats
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "TrainCache":
        npz = np.load(path, allow_pickle=False)
        mc = npz["multi_channel_feats"] if "multi_channel_feats" in npz.files else None
        return cls(
            t=npz["t"],
            vanilla_returns=npz["vanilla_returns"],
            news_emb=npz["news_emb"],
            hist_feats=npz["hist_feats"],
            target_returns=npz["target_returns"],
            multi_channel_feats=mc,
        )


def build_cache(
    samples: Sequence[AlignedSample],
    forecaster: Forecaster,
    encoder: NewsEncoder,
    *,
    channel_matrix: Optional["pd.DataFrame"] = None,
) -> TrainCache:
    """One-shot pre-computation of all training inputs.

    Each sample gets:
      vanilla_returns      : the forecaster's H-step return path (close-to-
                             close, seeded from ``sample.target[0]``).
      news_emb             : encoder.encode(sample.news, sample.t).
      hist_feats           : history_features(sample.history).
      target_returns       : sample.forward_returns (already in return space).
      multi_channel_feats  : (N, K*8) per-channel summary stats when
                             ``channel_matrix`` is supplied, else None.
    """
    if not samples:
        raise ValueError("build_cache: no samples")
    H = samples[0].forward_returns.size

    N = len(samples)
    vanilla = np.zeros((N, H), dtype=np.float32)
    target_ret = np.zeros((N, H), dtype=np.float32)
    hist = np.zeros((N, len(HIST_FEATURE_NAMES)), dtype=np.float32)
    t_arr = np.zeros(N, dtype=np.int64)

    # News embeddings — batched if the encoder supports it.
    news_lists = [list(s.news) for s in samples]
    t_dts = [s.t.to_pydatetime() for s in samples]
    news_emb = encoder.encode_batch(news_lists, t_dts)
    if news_emb.shape != (N, encoder.dim):
        raise RuntimeError(
            f"encoder returned shape {news_emb.shape}, expected ({N}, {encoder.dim})"
        )

    # Multi-channel features (optional).
    mc_feats: Optional[np.ndarray] = None
    if channel_matrix is not None and not channel_matrix.empty:
        from .multi_channel import multi_channel_features
        first_mc = multi_channel_features(channel_matrix, samples[0].t)
        mc_feats = np.zeros((N, first_mc.size), dtype=np.float32)

    for i, s in enumerate(samples):
        t_arr[i] = s.t.value
        seed = float(s.target[0])
        path = forecaster.forecast(s.history, H)
        prev = np.concatenate([[seed], path[:-1]])
        vanilla[i] = path / prev - 1.0
        target_ret[i] = s.forward_returns
        hist[i] = history_features(s.history)
        if mc_feats is not None and channel_matrix is not None:
            from .multi_channel import multi_channel_features
            mc_feats[i] = multi_channel_features(channel_matrix, s.t)

    return TrainCache(
        t=t_arr,
        vanilla_returns=vanilla.astype(np.float32),
        news_emb=news_emb.astype(np.float32),
        hist_feats=hist.astype(np.float32),
        target_returns=target_ret.astype(np.float32),
        multi_channel_feats=mc_feats,
    )


# --- dataset wrapper ----------------------------------------------------------
if _TORCH_OK:
    class CacheDataset(Dataset):
        """Thin wrapper turning a TrainCache into a torch Dataset."""

        def __init__(self, cache: TrainCache):
            self.vanilla = torch.from_numpy(cache.vanilla_returns)
            self.news = torch.from_numpy(cache.news_emb)
            self.hist = torch.from_numpy(cache.hist_feats)
            self.target = torch.from_numpy(cache.target_returns)
            self.mc = (
                torch.from_numpy(cache.multi_channel_feats)
                if cache.multi_channel_feats is not None
                else None
            )

        def __len__(self) -> int:
            return self.vanilla.shape[0]

        def __getitem__(self, idx):  # type: ignore[override]
            mc = self.mc[idx] if self.mc is not None else None
            return self.vanilla[idx], self.news[idx], self.hist[idx], self.target[idx], mc


# --- training -----------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    seed: int = 0
    early_stop_patience: int = 5
    # Loss is MSE on adjusted-vs-actual returns. We add a small L2 penalty on
    # the adjustment magnitude so the adapter prefers the smallest deviation
    # that explains the data.
    adjust_l2_lambda: float = 1e-3


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float       # MAE in return space (model)
    val_mae: float
    val_baseline_mae: float  # MAE if we just used vanilla (no adjustment)
    elapsed_s: float


@dataclass
class TrainResult:
    best_val_loss: float
    best_epoch: int
    log: List[EpochLog]
    n_train: int
    n_val: int
    checkpoint_path: Path
    config: TrainConfig
    adapter_cfg: AdapterConfig

    def to_json(self) -> str:
        return json.dumps({
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "checkpoint_path": str(self.checkpoint_path),
            "config": asdict(self.config),
            "adapter_cfg": asdict(self.adapter_cfg),
            "log": [asdict(e) for e in self.log],
        }, indent=2)


def train_adapter(
    cache: TrainCache,
    adapter_cfg: AdapterConfig,
    cfg: TrainConfig = TrainConfig(),
    *,
    checkpoint_path: Optional[Path | str] = None,
    device: str = "cpu",
) -> TrainResult:
    """Fit the conditioning adapter.

    Loss
    ----
    L = MSE(vanilla + adjustment, target) + λ · ||adjustment||²

    The vanilla path is *not* a learnable parameter; it's the frozen TimesFM 2
    forecast cached in ``cache.vanilla_returns``. The adapter's job is just to
    learn the residual.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    H = cache.vanilla_returns.shape[1]
    if adapter_cfg.horizon != H:
        raise ValueError(
            f"adapter_cfg.horizon={adapter_cfg.horizon} but cache horizon={H}"
        )
    if adapter_cfg.news_dim != cache.news_emb.shape[1]:
        raise ValueError(
            f"adapter news_dim={adapter_cfg.news_dim} but cache news_emb is {cache.news_emb.shape[1]}-dim"
        )

    ds = CacheDataset(cache)
    n_val = max(1, int(len(ds) * cfg.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = ConditioningAdapter(adapter_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out = Path(checkpoint_path) if checkpoint_path else \
        PROJECT_ROOT / "data" / "conditioning" / "adapter_best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    log_rows: List[EpochLog] = []
    patience = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        # ---- train ----
        model.train()
        train_losses: List[float] = []
        train_maes: List[float] = []
        for vanilla, news, hist, target, mc in train_loader:
            vanilla = vanilla.to(device); news = news.to(device)
            hist = hist.to(device); target = target.to(device)
            mc_dev = mc.to(device) if mc is not None else None
            adjust = model(vanilla, news, hist, mc_dev)
            adjusted = vanilla + adjust
            mse = ((adjusted - target) ** 2).mean()
            penalty = (adjust ** 2).mean()
            loss = mse + cfg.adjust_l2_lambda * penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
            train_maes.append(float((adjusted - target).abs().mean().item()))

        # ---- val ----
        model.eval()
        val_losses: List[float] = []
        val_maes: List[float] = []
        baseline_maes: List[float] = []
        with torch.no_grad():
            for vanilla, news, hist, target, mc in val_loader:
                vanilla = vanilla.to(device); news = news.to(device)
                hist = hist.to(device); target = target.to(device)
                mc_dev = mc.to(device) if mc is not None else None
                adjust = model(vanilla, news, hist, mc_dev)
                adjusted = vanilla + adjust
                mse = ((adjusted - target) ** 2).mean()
                penalty = (adjust ** 2).mean()
                loss = mse + cfg.adjust_l2_lambda * penalty
                val_losses.append(float(loss.item()))
                val_maes.append(float((adjusted - target).abs().mean().item()))
                baseline_maes.append(float((vanilla - target).abs().mean().item()))

        elapsed = time.time() - t0
        row = EpochLog(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=float(np.mean(val_losses)),
            train_mae=float(np.mean(train_maes)),
            val_mae=float(np.mean(val_maes)),
            val_baseline_mae=float(np.mean(baseline_maes)),
            elapsed_s=round(elapsed, 2),
        )
        log_rows.append(row)
        log.info(
            "epoch %2d  train=%.5f  val=%.5f (vs baseline %.5f)  elapsed=%.1fs",
            row.epoch, row.train_loss, row.val_loss, row.val_baseline_mae, row.elapsed_s,
        )

        if row.val_loss < best_val - 1e-7:
            best_val = row.val_loss
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "adapter_cfg": asdict(adapter_cfg),
                "epoch": epoch,
                "val_loss": best_val,
            }, out)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log.info("early stop at epoch %d (no improvement for %d)", epoch, patience)
                break

    return TrainResult(
        best_val_loss=best_val,
        best_epoch=best_epoch,
        log=log_rows,
        n_train=n_train,
        n_val=n_val,
        checkpoint_path=out,
        config=cfg,
        adapter_cfg=adapter_cfg,
    )


# --- inference helper ---------------------------------------------------------
def load_adapter_checkpoint(path: Path | str, *, device: str = "cpu"):
    """Load an adapter checkpoint and return (model, adapter_cfg)."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = AdapterConfig(**ckpt["adapter_cfg"])
    model = ConditioningAdapter(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

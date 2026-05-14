#!/usr/bin/env python3
"""End-to-end driver for the news-conditioned TimesFM 2 plumbing.

Walks the SPY history, builds the alignment dataset, caches vanilla
forecasts, trains the conditioning adapter, replays both vanilla and
conditioned forecasters on the holdout, and renders the diagnostics HTML.

Defaults are tuned for Sam's setup — feel free to tweak via CLI flags.

Example
-------
    python scripts/run_conditioning.py --train-start 2018-01-01 \\
        --holdout-start 2025-03-07 --holdout-end 2026-03-06 \\
        --forecaster timesfm

Pass ``--forecaster naive`` to skip the TimesFM 2 download and use the
naive-last fallback (useful for harness sanity checks).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `backend` importable regardless of where this script is invoked from.
# `python scripts/run_conditioning.py` puts `scripts/` on sys.path, not the
# repo root — so we add it explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.conditioning.adapter import AdapterConfig
from backend.conditioning.alignment import AlignmentConfig, walk_aligned, fetch_news_rows
from backend.conditioning.baseline import NaiveLastForecaster, replay
from backend.conditioning.data import load_spy
from backend.conditioning.diagnostics import render_html
from backend.conditioning.embed import StatsEncoder, make_encoder
from backend.conditioning.replay import ConditionedForecaster, compare, replay_conditioned
from backend.conditioning.train import (
    TrainConfig,
    build_cache,
    load_adapter_checkpoint,
    train_adapter,
)
from backend.core.config import PROJECT_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-start", default="2018-01-01")
    p.add_argument("--train-end", default="2025-03-06")
    p.add_argument("--holdout-start", default="2025-03-07")
    p.add_argument("--holdout-end", default="2026-03-06")
    p.add_argument("--lookback", type=int, default=512)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--news-lookback-days", type=int, default=7)
    p.add_argument("--forecaster", choices=["timesfm", "naive"], default="timesfm")
    p.add_argument("--encoder", choices=["stats", "text"], default="stats")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--use-multi-channel", action="store_true",
                   help="Download ~20-50 channel ETF universe and use per-channel "
                        "history features as additional adapter inputs. "
                        "Adds ~K*8 dims to the adapter input (K = channels loaded). "
                        "First run downloads and caches the price series via yfinance.")
    p.add_argument("--mc-tickers", default=None,
                   help="Comma-separated custom ticker list for multi-channel mode. "
                        "Defaults to HOUSE_UNIVERSE env var or the built-in ~40-ticker list.")
    p.add_argument("--quiet", action="store_true", help="WARN-level logging only")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    log = logging.getLogger("run_conditioning")

    # ---- 1. data ----
    log.info("Loading SPY history…")
    df = load_spy()
    log.info("SPY %d rows %s → %s", len(df), df.index.min().date(), df.index.max().date())

    cfg_align = AlignmentConfig(
        history_lookback=args.lookback,
        news_lookback_days=args.news_lookback_days,
        horizon=args.horizon,
    )

    # Pre-fetch news once spanning the whole train+holdout window.
    log.info("Fetching news from DB…")
    from datetime import datetime, timezone, timedelta
    import pandas as pd
    news_start = (pd.Timestamp(args.train_start) - pd.Timedelta(days=args.news_lookback_days)) \
        .to_pydatetime().replace(tzinfo=timezone.utc)
    news_end = (pd.Timestamp(args.holdout_end) + pd.Timedelta(days=1)) \
        .to_pydatetime().replace(tzinfo=timezone.utc)
    try:
        all_news = fetch_news_rows(news_start, news_end)
    except Exception as e:
        log.warning("news fetch failed (%s) — proceeding with empty news", e)
        all_news = []
    log.info("news rows in window : %d", len(all_news))

    train_samples = list(walk_aligned(df, cfg_align,
                                       start=args.train_start, end=args.train_end,
                                       news=all_news))
    holdout_samples = list(walk_aligned(df, cfg_align,
                                         start=args.holdout_start, end=args.holdout_end,
                                         news=all_news))
    log.info("train samples=%d, holdout samples=%d", len(train_samples), len(holdout_samples))
    if not train_samples or not holdout_samples:
        raise SystemExit("walk_aligned produced empty splits — check your date windows")

    # ---- 2. forecaster + encoder ----
    if args.forecaster == "timesfm":
        from backend.conditioning.baseline import TimesFMBaseline
        log.info("Loading TimesFM 2 (first call downloads the checkpoint, ~900MB)…")
        vanilla = TimesFMBaseline(context_len=args.lookback)
    else:
        vanilla = NaiveLastForecaster()
    log.info("vanilla forecaster: %s", vanilla.name)

    encoder = make_encoder(args.encoder)
    log.info("news encoder: kind=%s, dim=%d", args.encoder, encoder.dim)

    # ---- 2b. optional multi-channel universe ----
    channel_matrix = None
    mc_dim = 0
    if args.use_multi_channel:
        from backend.conditioning.multi_channel import (
            load_universe, align_universe, build_channel_manifest,
        )
        mc_tickers = (
            [t.strip() for t in args.mc_tickers.split(",") if t.strip()]
            if args.mc_tickers
            else None
        )
        log.info("Loading multi-channel universe (yfinance)…")
        universe = load_universe(
            tickers=mc_tickers,
            start=args.train_start,
            end=args.holdout_end,
        )
        channel_matrix = align_universe(universe)
        manifest = build_channel_manifest(channel_matrix, args.train_start, args.holdout_end)
        mc_dim = manifest.feature_dim
        log.info("\n%s", manifest.to_text())
        manifest.save()

    # ---- 3. cache + train ----
    log.info("Building train cache (vanilla forecasts + news embs)…")
    cache = build_cache(train_samples, vanilla, encoder, channel_matrix=channel_matrix)
    log.info("cache: vanilla%s, news_emb%s, hist%s, target%s",
             cache.vanilla_returns.shape, cache.news_emb.shape,
             cache.hist_feats.shape, cache.target_returns.shape)

    ad_cfg = AdapterConfig(horizon=args.horizon, news_dim=encoder.dim, multi_channel_dim=mc_dim)
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    log.info("Training adapter (%d params)…",
             ad_cfg.total_input_dim() * ad_cfg.hidden)  # rough
    result = train_adapter(cache, ad_cfg, train_cfg)
    log.info("best val_loss=%.6f at epoch %d", result.best_val_loss, result.best_epoch)

    # ---- 4. holdout replay (vanilla + conditioned) ----
    log.info("Replaying vanilla forecaster on holdout…")
    vanilla_replay = replay(holdout_samples, vanilla)
    log.info("Replaying conditioned forecaster on holdout…")
    adapter_model, _ = load_adapter_checkpoint(result.checkpoint_path)
    cond_fc = ConditionedForecaster(
        name=f"{vanilla.name}+adapter", vanilla=vanilla, adapter=adapter_model, encoder=encoder,
    )
    cond_replay = replay_conditioned(holdout_samples, cond_fc)

    cmp = compare(vanilla_replay, cond_replay)
    print(); print(cmp.to_text()); print()

    # ---- 5. diagnostics ----
    out_dir = PROJECT_ROOT / "data" / "conditioning"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = render_html(
        vanilla_replay, cond_replay, cmp, holdout_samples,
        out_path=out_dir / "diagnostics.html",
        vanilla_name=f"Vanilla {vanilla.name}",
        cond_name=f"Conditioned ({encoder.dim}-dim {args.encoder})",
    )
    log.info("diagnostics → %s", html_path)


if __name__ == "__main__":
    main()

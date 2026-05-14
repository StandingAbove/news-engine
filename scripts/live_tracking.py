#!/usr/bin/env python3
"""Day-to-day live tracking of vanilla vs. conditioned TimesFM 2.

Per Singh's guidance: "TimesFM 2 can run in the background in some ways, all
the way from last year or five years ago. You can see on a day-to-day basis
how well it is tracking."

This script walks from --start-date to --end-date (default: today), replaying
both the vanilla TimesFM 2 baseline and, if a checkpoint exists, the
conditioned adapter. Each run appends per-day rows to a persistent tracking
CSV at data/conditioning/live_tracking.csv.

Run it once a day (cron, launchd, etc.) to keep the log current. Use
--incremental to skip days already in the log.

Examples
--------
    # Walk from a year ago to today (initial backfill)
    python scripts/live_tracking.py --start-date 2025-05-12

    # Incremental: only replay days not yet in the log
    python scripts/live_tracking.py --incremental

    # Skip the ~900MB TimesFM download and use naive_last instead
    python scripts/live_tracking.py --forecaster naive --start-date 2025-05-12
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.conditioning.alignment import AlignmentConfig, walk_aligned, fetch_news_rows
from backend.conditioning.baseline import NaiveLastForecaster, replay
from backend.conditioning.data import load_spy
from backend.conditioning.embed import make_encoder
from backend.conditioning.replay import ConditionedForecaster, replay_conditioned
from backend.conditioning.train import load_adapter_checkpoint
from backend.core.config import PROJECT_ROOT

TRACKING_LOG = PROJECT_ROOT / "data" / "conditioning" / "live_tracking.csv"


def _effective_window(
    requested_start: str,
    requested_end: str,
    *,
    incremental: bool,
) -> tuple[str, str]:
    """Return the (start, end) to actually replay, honouring --incremental."""
    if not incremental or not TRACKING_LOG.exists():
        return requested_start, requested_end

    log_df = pd.read_csv(TRACKING_LOG, parse_dates=["date"])
    if log_df.empty:
        return requested_start, requested_end

    last_logged = log_df["date"].max()
    new_start = (last_logged + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if pd.Timestamp(new_start) > pd.Timestamp(requested_end):
        return "", ""
    return new_start, requested_end


def _append_log(vanilla_replay, cond_replay) -> None:
    """Append one row per day to the tracking CSV and deduplicate."""
    vanilla_t = pd.to_datetime(vanilla_replay.t, unit="ns")
    rows = []
    for i, t in enumerate(vanilla_t):
        vf = vanilla_replay.forecast_returns[i]
        vt = vanilla_replay.target_returns[i]
        row: dict = {
            "date": t.date().isoformat(),
            "vanilla_mae": float(np.abs(vf - vt).mean()),
            "vanilla_dir_acc": float((np.sign(vf) == np.sign(vt)).mean()),
        }
        if cond_replay is not None:
            cf = cond_replay.forecast_returns[i]
            ct = cond_replay.target_returns[i]
            row["cond_mae"] = float(np.abs(cf - ct).mean())
            row["cond_dir_acc"] = float((np.sign(cf) == np.sign(ct)).mean())
            row["win"] = int(row["cond_mae"] < row["vanilla_mae"])
        rows.append(row)

    new_df = pd.DataFrame(rows)
    if TRACKING_LOG.exists():
        existing = pd.read_csv(TRACKING_LOG)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df

    TRACKING_LOG.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TRACKING_LOG, index=False)


def main() -> None:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    one_year_ago = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-date", default=one_year_ago,
                   help="First day to include in the replay window.")
    p.add_argument("--end-date", default=today,
                   help="Last day (inclusive). Defaults to today.")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--lookback", type=int, default=512)
    p.add_argument("--news-lookback-days", type=int, default=7)
    p.add_argument("--forecaster", choices=["timesfm", "naive"], default="timesfm",
                   help="Use 'naive' to skip the TimesFM 2 download.")
    p.add_argument("--encoder", choices=["stats", "text"], default="stats")
    p.add_argument("--checkpoint", default=None,
                   help="Path to adapter .pt checkpoint. "
                        "Defaults to data/conditioning/adapter_best.pt.")
    p.add_argument("--incremental", action="store_true",
                   help="Only replay days not yet in live_tracking.csv.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )
    log = logging.getLogger("live_tracking")

    start, end = _effective_window(
        args.start_date, args.end_date, incremental=args.incremental
    )
    if not start:
        print("Tracking log is already up to date — nothing to do.")
        return

    log.info("tracking window: %s → %s", start, end)

    # ---- data ----
    df = load_spy()

    cfg_align = AlignmentConfig(
        history_lookback=args.lookback,
        news_lookback_days=args.news_lookback_days,
        horizon=args.horizon,
    )

    news_start = (
        pd.Timestamp(start) - pd.Timedelta(days=args.news_lookback_days)
    ).to_pydatetime().replace(tzinfo=timezone.utc)
    news_end = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).to_pydatetime().replace(tzinfo=timezone.utc)

    try:
        all_news = fetch_news_rows(news_start, news_end)
    except Exception as exc:
        log.warning("news fetch failed (%s) — replaying with empty news", exc)
        all_news = []

    samples = list(walk_aligned(df, cfg_align, start=start, end=end, news=all_news))
    if not samples:
        log.warning(
            "no samples in %s → %s (need ≥%d days of prior history)",
            start, end, args.lookback,
        )
        return
    log.info("%d samples to replay", len(samples))

    # ---- vanilla forecaster ----
    if args.forecaster == "timesfm":
        from backend.conditioning.baseline import TimesFMBaseline
        log.info("loading TimesFM 2 (may download ~900MB on first run)…")
        vanilla = TimesFMBaseline(context_len=args.lookback)
    else:
        vanilla = NaiveLastForecaster()
    log.info("forecaster: %s", vanilla.name)

    encoder = make_encoder(args.encoder)

    # ---- vanilla replay ----
    log.info("replaying vanilla…")
    vanilla_replay = replay(samples, vanilla)

    # ---- conditioned replay (optional, skipped if no checkpoint) ----
    cond_replay = None
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else PROJECT_ROOT / "data" / "conditioning" / "adapter_best.pt"
    )
    if ckpt_path.exists():
        log.info("loading adapter checkpoint from %s…", ckpt_path)
        try:
            adapter_model, _ = load_adapter_checkpoint(ckpt_path)
            cond_fc = ConditionedForecaster(
                name=f"{vanilla.name}+adapter",
                vanilla=vanilla,
                adapter=adapter_model,
                encoder=encoder,
            )
            log.info("replaying conditioned…")
            cond_replay = replay_conditioned(samples, cond_fc)
        except Exception as exc:
            log.warning("adapter load failed (%s) — logging vanilla only", exc)
    else:
        log.info("no adapter checkpoint at %s — logging vanilla only", ckpt_path)

    # ---- persist ----
    _append_log(vanilla_replay, cond_replay)

    # ---- summary ----
    log_df = pd.read_csv(TRACKING_LOG)
    n = len(log_df)
    print(f"\nTracking log : {TRACKING_LOG}")
    print(f"Logged days  : {n}")
    if "cond_mae" in log_df.columns:
        win_rate = log_df["win"].mean()
        n_wins = int(log_df["win"].sum())
        print(f"Win rate     : {win_rate:.1%}  ({n_wins}/{n} days conditioned < vanilla)")
        print(f"Vanilla MAE  : {log_df['vanilla_mae'].mean():.6f}  (mean over log)")
        print(f"Cond.   MAE  : {log_df['cond_mae'].mean():.6f}  (mean over log)")
    else:
        print(f"Vanilla MAE  : {log_df['vanilla_mae'].mean():.6f}  (mean over log)")
    print()


if __name__ == "__main__":
    main()

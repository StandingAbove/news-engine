# News-conditioned TimesFM 2

This is the headline sub-component going forward, per Prof. Singh's April 28
guidance: news arrives day-by-day and acts as a *signal that adjusts the
parameters of TimesFM 2 itself*, not just a feature predicted alongside it.

## High-level flow

```
                 ┌──────────────────────┐
                 │ FactSet SPY daily    │  data/historical/SPY.csv
                 └─────────┬────────────┘
                           │ 33 yrs of closes, returns, vol
                           ▼
┌──────────────────────┐         ┌────────────────────────────────┐
│ News DB (sub-comp B) │         │ AlignedSample per trading day  │
│ providers + sentiment│────────▶│  history(L) + news[t-K..t]     │
│ + ticker mapping     │         │  + target[t..t+H]              │
└──────────────────────┘         └─────────────┬──────────────────┘
                                               │
                                               ▼
                       ┌──────────────────────────────────────────┐
                       │ Encoder  +  Frozen TimesFM 2  +  Adapter │
                       │  (16-dim       (vanilla path)   (LoRA-   │
                       │   stats /       in return-space  style   │
                       │   text)                          MLP)    │
                       └─────────────────────┬────────────────────┘
                                             │ adjusted = vanilla + Δ
                                             ▼
                                      conditioned forecast
                                             │
                                             ▼
                              walk-forward replay → diagnostics.html
```

## Package layout

| File | Purpose |
|---|---|
| `data.py`       | FactSet SPY loader + train/holdout split + pickle cache |
| `alignment.py`  | Walks the calendar producing (history, news, target) samples; builds `.npz` dataset |
| `embed.py`      | News → fixed-size vector. `StatsEncoder` (16-dim, no deps) + optional `TextEncoder` (sentence-transformers) |
| `baseline.py`   | Vanilla forecaster wrapper + walk-forward replay |
| `adapter.py`    | Tiny MLP head: (vanilla_returns, news_emb, hist_feats) → return-space adjustment, soft-tanh-clipped |
| `train.py`      | Pre-cache vanilla forecasts, train adapter (MSE on adjusted vs actual) with early stopping |
| `replay.py`     | Conditioned walk-forward replay + head-to-head comparison |
| `diagnostics.py`| Self-contained HTML report (Plotly via CDN) |

## Decoupling rule

`backend/conditioning/` is the *only* place where sub-component A (forecast/)
and sub-component B (news/) meet. The two original packages still don't
import each other; this one imports from both. Verified via grep on every
push.

## Running it

```bash
# default: TimesFM 2 + 16-dim StatsEncoder, 1y holdout
python scripts/run_conditioning.py

# fast harness sanity (no TimesFM download, naive_last forecaster)
python scripts/run_conditioning.py --forecaster naive --epochs 10

# different windows / horizon
python scripts/run_conditioning.py \
  --train-start 2018-01-01 --train-end 2025-03-06 \
  --holdout-start 2025-03-07 --holdout-end 2026-03-06 \
  --horizon 20 --news-lookback-days 7
```

Output:

- `data/conditioning/adapter_best.pt` — the trained adapter checkpoint.
- `data/conditioning/diagnostics.html` — KPI strip + per-day MAE + per-horizon
  step MAE + top-10 win/loss days with the relevant news headlines.

## Multi-channel time series

Per Singh: "you should have maybe 100 channel time series data, or 50
channels, whatever you can afford. You should know where they are coming
from, what is the source."

`backend/conditioning/multi_channel.py` extends the pipeline:

- Default universe: ~40 liquid ETFs (HOUSE_UNIVERSE + additional sector/
  factor/macro ETFs). Source: yfinance (free, ~15-min delayed).
- Each channel contributes 8 history-summary statistics (same feature set as
  the single-ticker adapter). At K channels the adapter input grows by K×8.
- Activate with `--use-multi-channel` in `run_conditioning.py`. First run
  downloads and caches the price series; subsequent runs use the file cache.
- For Bloomberg-quality delayed data, see
  `backend/market/perplexity_client.py` (Perplexity API, ~$20/month).

```bash
python scripts/run_conditioning.py --use-multi-channel --forecaster naive --epochs 5
```

## Live day-to-day tracking

Per Singh: "TimesFM 2 can run in the background, from last year or five years
ago, and you can see on a day-to-day basis how well it is tracking."

`scripts/live_tracking.py` walks from a start date to today, appending
per-day vanilla vs. conditioned MAE and directional accuracy to
`data/conditioning/live_tracking.csv`.

```bash
# Initial backfill (1 year)
python scripts/live_tracking.py --start-date 2025-05-12

# Incremental daily update
python scripts/live_tracking.py --incremental
```

## What's not done yet

- **GRPO.** Designed in `docs/GRPO_DESIGN.md`, not implemented. The
  supervised fine-tune is the baseline; GRPO is the follow-up when the
  supervised model plateaus.
- **Text-encoder default.** Stats encoder is the default (no extra dep).
  Pass `--encoder text` to use sentence-transformers (~25MB MiniLM).
- **News backfill.** The pipeline reads from the existing `NewsItem` DB,
  which only has live-ingest data. Backfill 2018–2025 from EODHD/NewsAPI.ai;
  the alignment walker accepts whatever's in the DB.

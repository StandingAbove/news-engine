# News-Engine — News-driven allocation research

A research codebase organized around the two sub-components Dr. Singh asked
for in his Feb-6 email:

> *"To get started, (1) some forecasting model and (2) source of text feeds
> which will be synthesized. They don't need to talk right now. a) Get
> timesFM and a few other forecasting models on your own. b) Feed them
> through some data, see that you get some mechanism operational — where the
> data is toy. Then move to some real data. Let the model fail and/or
> establish some minimum performance."*

The two sub-components live in separate packages, have separate evaluators,
and don't depend on each other. An earlier integrated stack (dashboard,
paper trader, blended strategy) is still in the repo as **v2 / future
integration** — it was built in the first pass before this guidance
arrived, and is intentionally kept off the critical path.

---

## Sub-component A — Forecasting models

Location: `backend/forecast/`

A uniform `Forecaster` interface with these implementations:

| name            | family               | deps                 |
|-----------------|----------------------|----------------------|
| `naive_last`    | baseline             | numpy                |
| `naive_mean`    | baseline             | numpy                |
| `naive_drift`   | baseline             | numpy                |
| `ewma`          | exp. smoothing       | numpy                |
| `holt_winters`  | exp. smoothing       | statsmodels          |
| `arima`         | classical            | statsmodels          |
| `mlp`           | small neural net     | torch                |
| `lstm`          | small neural net     | torch                |
| `timesfm`       | Google foundation    | timesfm + torch      |

Toy data generators (`backend/forecast/toy.py`) cover sine / sine+trend /
AR(1) / random walk / regime switch / noisy price. The harness
(`backend/forecast/harness.py`) runs any subset of the models on any subset
of the series and produces a comparison DataFrame with MAE / RMSE / MAPE /
MASE / directional-accuracy.

**Toy first, then real:**

```bash
# Toy only — fast, no network.
./scripts/eval_forecast.sh

# Toy + real tickers (fetches via yfinance).
./scripts/eval_forecast.sh SPY QQQ AAPL

# Fine-grained control.
python -m backend.forecast.run_eval --models naive_drift,arima,timesfm \
       --horizon 20 --real SPY,QQQ --real-period 2y
```

Heavy-dep models (`statsmodels`, `torch`, `timesfm`) are **skipped cleanly**
with a per-model reason when the dep isn't installed, so the eval always
produces output. To enable them:

```bash
pip install -r requirements-forecast.txt
```

The first TimesFM run downloads ~800MB of checkpoint from HuggingFace;
subsequent runs are cached in-process.

## Sub-component B — Text feed synthesis

Location: `backend/news/`

Pipeline: provider fetch → dedupe → finance-tuned VADER sentiment →
ticker/sector/region mapping (Hong Kong → EEM, Fed → TLT, etc.) →
per-ticker aggregation with half-life decay and tanh squashing.

**Toy-data harness** at `backend/news/toy.py`: 18 hand-written headlines
with expected sentiment direction and expected ticker tags, driven through
the same pipeline that the live feed uses. The evaluator reports:

* sentiment direction accuracy,
* ticker-tag recall,
* aggregate-sign accuracy on the squashed scores.

```bash
./scripts/eval_text_feed.sh
```

Floors are asserted in `backend/tests/test_news_toy.py` so regressions are
caught by pytest.

**Move-to-real:** once toy accuracy looks right, the same pipeline runs
against the three live providers (Marketaux, NewsAPI.ai, EODHD) via the
scheduler. That integration is already wired; see `backend/news/aggregator.py`.

> Caveat (and known gap vs Dr. Singh's ask): sentiment here is
> **VADER + a finance lexicon**, not a proper forecasting model. It's
> a deterministic placeholder adequate for wiring and toy eval, but the
> real-data performance floor is expected to be low. The planned replacement
> is a small fine-tuned language model; parked until sub-component A is
> standing on its own.

---

## v2 / integrated stack (future integration)

The following existed before the sub-component split and is **off the
critical path** per Dr. Singh's "they don't need to talk right now":

* FastAPI backend + vanilla-JS dashboard (`backend/api/`, `frontend/`)
* Paper-trading engine (`backend/portfolio/`)
* Vectorized backtester (`backend/backtest/`)
* News → signal → weight allocator (`backend/signals/`)
* Cloudflare-tunnel launcher (`scripts/run.sh --share`)

Kept, runnable, and tested — but explicitly not the current milestone.
Once both sub-components individually pass their toy evals and clear some
minimum real-data bar, this is where they'd be wired together.

---

## Quickstart

```bash
# core (text-feed sub-component, v2 integration, tests)
pip install -r requirements.txt

# forecasting sub-component (opt-in — ARIMA / Holt-Winters / MLP / LSTM)
pip install -r requirements-forecast.txt

# TimesFM — installed separately because PyPI caps it at Python <3.12.
# Python ≤3.12: pip install "timesfm>=1.2"
# Python 3.13:  pip install "git+https://github.com/google-research/timesfm.git@main"
# See docs/FORECAST.md for the fallback (parallel 3.12 venv).

# tests — use scripts/test.sh rather than bare `pytest`.
# If conda base is active alongside .venv, bare `pytest` will pick up the
# conda interpreter and miss venv-installed deps. scripts/test.sh calls
# .venv/bin/python -m pytest directly to avoid that.
./scripts/test.sh

# text-feed sub-component, toy eval
./scripts/eval_text_feed.sh

# forecasting sub-component, toy eval
./scripts/eval_forecast.sh

# forecasting sub-component, toy + real
./scripts/eval_forecast.sh SPY QQQ

# v2 integrated dashboard (off-path)
./scripts/run.sh --share
```

---

## Project layout

```
news-engine/
├── backend/
│   ├── forecast/         # sub-component A (Dr. Singh's sub-component 1)
│   │   ├── base.py       # Forecaster protocol + Forecast dataclass
│   │   ├── metrics.py    # MAE / RMSE / MAPE / MASE / dir-acc
│   │   ├── toy.py        # deterministic toy series
│   │   ├── harness.py    # evaluate(models, series) → DataFrame
│   │   ├── real_data.py  # yfinance loader for the "then real" step
│   │   ├── run_eval.py   # python -m backend.forecast.run_eval
│   │   └── models/
│   │       ├── naive.py
│   │       ├── ewma.py
│   │       ├── arima.py
│   │       ├── mlp.py
│   │       ├── lstm.py
│   │       └── timesfm_model.py
│   ├── news/             # sub-component B (Dr. Singh's sub-component 2)
│   │   ├── providers.py
│   │   ├── aggregator.py
│   │   ├── sentiment.py
│   │   ├── mapping.py
│   │   ├── toy.py        # toy headlines + expected signals
│   │   └── toy_eval.py
│   ├── signals/          # v2 — signal → weight allocator
│   ├── backtest/         # v2 — vectorized backtester
│   ├── portfolio/        # v2 — paper trader
│   ├── market/           # v2 — yfinance wrapper
│   ├── api/              # v2 — FastAPI + WebSocket
│   ├── scheduler.py      # v2
│   ├── main.py           # v2
│   └── tests/            # unit tests (all sub-components)
├── frontend/             # v2 dashboard
├── scripts/
│   ├── eval_forecast.sh       # sub-component A runner
│   ├── eval_text_feed.sh      # sub-component B runner
│   ├── run.sh                 # v2 dashboard launcher
│   └── run.bat                # Windows equivalent
├── docs/
│   ├── FORECAST.md            # sub-component A design doc
│   └── TEXT_FEED.md           # sub-component B design doc
├── data/
├── requirements.txt           # core
├── requirements-forecast.txt  # sub-component A opt-in deps
├── pytest.ini
└── README.md
```

---

## Testing

```bash
pytest -q
```

Covers:

* Forecast metrics (MAE / RMSE / MAPE / MASE / directional accuracy)
* Every `Forecaster` on toy data (with skip markers for missing deps)
* Harness end-to-end (per-series ranking, skip/error reporting)
* Text-feed toy accuracy floors (sentiment direction, ticker recall,
  aggregate-sign agreement)
* Signal aggregator + water-filling weight cap (v2)
* Backtester turnover / cost accounting (v2)
* Metric formulas: Sharpe, Sortino, drawdown, beta, alpha (v2)

---

## Status vs Dr. Singh's Feb-6 ask

| Ask                                                     | Status |
|---------------------------------------------------------|--------|
| (1) A forecasting model (TimesFM + a few others)        | ✅ Sub-component A — 9 models, uniform interface |
| (2) A source of text feeds, synthesized                 | ✅ Sub-component B — 3 providers live; toy harness with accuracy floors |
| Toy data first, then real                               | ✅ Both sub-components: toy harness wired; real-data hook in place (yfinance for A, live providers for B) |
| "Let the model fail" / min performance                  | ✅ Harness reports status=error and keeps going — no hidden passes |
| They don't need to talk                                 | ✅ No cross-imports between `backend/forecast/` and `backend/news/` |
| No formal report                                        | ✅ This README + `docs/FORECAST.md` + `docs/TEXT_FEED.md` in place of a report |

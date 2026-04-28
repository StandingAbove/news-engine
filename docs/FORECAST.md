# Sub-component A — Forecasting models

This is Dr. Singh's sub-component (1): a set of forecasting models,
including TimesFM, standing up in isolation so they can be characterized
before anything is wired to the text-feed side.

## Goals

1. **Uniform interface.** Every model — naive, ARIMA, EWMA, Holt-Winters,
   MLP, LSTM, TimesFM — implements the same `Forecaster` protocol:
   `fit(history)` then `forecast(horizon) -> Forecast`. Nothing model-
   specific leaks into the harness.
2. **Toy before real.** A deterministic toy-series catalog (sine, sine+trend,
   AR(1), random walk, regime switch, noisy price) is run first. Real
   tickers are a second pass, opted into via `--real` on the eval script.
3. **Fail loudly.** When a model's dependency is missing, the harness
   marks the row `skipped`. When a fit crashes, the row is marked `error`
   with a reason. No silent passes.
4. **Decoupled.** This package must not import from `backend.news`,
   `backend.signals`, `backend.portfolio`, or any of the v2 stack.

## Interface

```python
# backend/forecast/base.py
@dataclass
class Forecast:
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None

class Forecaster(Protocol):
    name: str
    def fit(self, history: np.ndarray) -> None: ...
    def forecast(self, horizon: int) -> Forecast: ...
```

Static method `is_available()` returns `False` when heavy deps aren't
installed; the harness uses it to decide whether to skip or run.

## Model roster

| `name`          | Family              | Key hyperparameters                   | Intended floor role                |
|-----------------|---------------------|----------------------------------------|-------------------------------------|
| `naive_last`    | flat                | —                                     | Random-walk champion                |
| `naive_mean`    | flat                | —                                     | Mean-reversion champion             |
| `naive_drift`   | linear extrapolation| —                                     | Trend champion                      |
| `ewma`          | exp. smoothing      | `alpha`                               | Very low-variance baseline          |
| `holt_winters`  | exp. smoothing      | `trend`, `seasonal`, `seasonal_periods`| Captures level+trend+seasonality    |
| `arima`         | classical           | auto-selected `(p, d, q)` by AIC     | Competent stochastic-process model  |
| `mlp`           | neural              | `lookback`, `hidden`, `epochs`        | Learnable baseline                  |
| `lstm`          | neural              | `lookback`, `hidden`, `epochs`        | Stronger neural baseline            |
| `timesfm`       | foundation          | `repo_id`, `context_len`, `use_v2`    | **Target upper bound**              |

## Evaluation

Metrics reported per (series, model):

* **MAE** / **RMSE** — absolute and squared error.
* **MAPE** — percentage error (skips zero-truth points).
* **MASE** — MAE normalized by the in-sample naive error. MASE < 1 → the
  model beats seasonal-naive on the training window.
* **Directional accuracy** — fraction of forecast steps whose direction
  vs. the last observed value matches the truth direction.

Run:

```bash
./scripts/eval_forecast.sh                   # toy only
./scripts/eval_forecast.sh SPY QQQ           # toy + real
python -m backend.forecast.run_eval --help   # full flag list
```

The CLI prints per-series rankings (ordered by MAE) plus a final
cross-dataset leaderboard averaged over series.

## TimesFM specifics

### Installing on Python 3.13

The `timesfm` PyPI package pins `python_requires` to `<3.12`. If you're on
3.13 (which we are, for Apple Silicon wheel compatibility), `pip install
timesfm` fails — that's why `requirements-forecast.txt` doesn't list it.

Pick one:

| Path | Command | Notes |
|------|---------|-------|
| Python ≤3.12    | `pip install "timesfm>=1.2"`                                                 | Easiest |
| Python 3.13 from source | `pip install "git+https://github.com/google-research/timesfm.git@main"` | `main` has dropped the 3.12 cap. Requires `git-lfs` installed locally (`brew install git-lfs && git lfs install`). |
| …same, without git-lfs | `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/google-research/timesfm.git /tmp/timesfm && pip install /tmp/timesfm` | Works on locked-down machines. LFS-stored files are artifacts, not source — the HF download at model-load time supplies the weights. |
| Parallel 3.12 venv | `python3.12 -m venv .venv-forecast` → activate → install                 | Recommended if (2) fails |

If none of those work, the harness cleanly skips TimesFM and the other 8
models still run — this is exactly what `is_available()` is for.

### Wrapper implementation

The wrapper (`models/timesfm_model.py`):

* Defaults to `google/timesfm-1.0-200m-pytorch` (smaller, ~800MB).
* `use_v2=True` switches to `google/timesfm-2.0-500m-pytorch` (~2GB, 50
  layers, positional-embedding disabled per the published config).
* Tries two init APIs in order (`TimesFmHparams + TimesFmCheckpoint`, then
  the older positional init) so it survives minor `timesfm` version bumps.
* Caches the loaded model on the class so repeat fits don't re-download.
* Clips forecast length to `horizon` (TimesFM emits up to
  `horizon_len=128` by default).

## Known caveats

* MLP / LSTM are intentionally tiny. Making them big is not the point of
  this comparison; the foundation-model slot is TimesFM's to lose.
* MASE is NaN when the training history is constant (division by zero) —
  the harness surfaces it as NaN rather than 0, so it's visible.
* Neural baselines train on `epochs` Adam steps with no early stopping —
  good enough for short toy series, overkill for long ones. Pass
  `model_kwargs={"mlp": {"epochs": 50}}` to `evaluate()` to trim.

## What "failing" looks like here

Per Dr. Singh: *"Let the model fail and/or establish some minimum
performance."* Concretely, we expect:

* TimesFM to beat naive on trend-plus-sine, lose or tie on pure random walk.
* ARIMA to dominate AR(1) and lose on regime switches.
* Naive-last to be the champion on random-walk-style series (by design).
* MLP/LSTM to be mediocre at this scale; they're here as sanity anchors.

When any of those assumptions breaks, the harness surfaces the MAE/RMSE
directly so the diagnosis is fast.

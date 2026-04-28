"""Evaluation harness.

Runs a set of models over a set of series and produces a comparison table.
Series are split train/test with a contiguous holdout window (no leakage).

Typical usage
-------------
```python
from backend.forecast.harness import evaluate, print_report
from backend.forecast import toy

results = evaluate(
    series=toy.all_series(400),
    models=["naive_last", "naive_drift", "ewma", "arima", "mlp", "lstm", "timesfm"],
    horizon=32,
)
print_report(results)
```

Every row of the returned DataFrame is `(series, model, mae, rmse, mape, mase,
dir_acc, elapsed_s, status)`. Unavailable models (missing dep, crashed fit)
are surfaced with an explicit `skipped` / `error` status rather than silently
dropped — that's half the value when standing up sub-components separately.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import metrics as mx
from . import models as M
from .base import as_array


@dataclass
class RunResult:
    series: str
    model: str
    status: str                 # "ok" | "skipped" | "error"
    elapsed_s: float
    metrics: Dict[str, float]
    detail: str = ""

    def row(self) -> Dict[str, object]:
        out = {
            "series": self.series,
            "model": self.model,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s, 3),
        }
        out.update({k: self.metrics.get(k, float("nan")) for k in ("mae", "rmse", "mape", "mase", "dir_acc")})
        if self.detail:
            out["detail"] = self.detail
        return out


def _split(series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    arr = as_array(series)
    if len(arr) <= horizon + 8:
        raise ValueError("series too short for this horizon")
    return arr[:-horizon], arr[-horizon:]


def evaluate(
    series: Dict[str, np.ndarray],
    models: Iterable[str],
    *,
    horizon: int = 32,
    model_kwargs: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    model_kwargs = model_kwargs or {}
    rows: List[Dict[str, object]] = []

    for sname, s in series.items():
        try:
            train, test = _split(s, horizon)
        except Exception as e:  # noqa: BLE001
            for mname in models:
                rows.append(RunResult(sname, mname, "error", 0.0, {}, f"split failed: {e}").row())
            continue

        for mname in models:
            if mname not in M.ALL_MODELS:
                rows.append(RunResult(sname, mname, "error", 0.0, {}, "unknown model").row())
                continue

            cls = M.ALL_MODELS[mname]
            avail = getattr(cls, "is_available", lambda: True)()
            if not avail:
                rows.append(RunResult(sname, mname, "skipped", 0.0, {}, "dependency unavailable").row())
                continue

            kwargs = model_kwargs.get(mname, {})
            model = cls(**kwargs)
            t0 = time.perf_counter()
            try:
                model.fit(train)
                fc = model.forecast(horizon)
                m = mx.summarize_forecast(test, fc.point, train)
                rows.append(RunResult(sname, mname, "ok", time.perf_counter() - t0, m).row())
            except Exception as e:  # noqa: BLE001
                rows.append(
                    RunResult(
                        sname, mname, "error", time.perf_counter() - t0, {},
                        f"{type(e).__name__}: {e}",
                    ).row()
                )

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    """Pretty-print the evaluate() DataFrame — per-series ranking on MAE."""
    if df.empty:
        print("(no results)")
        return
    for sname, grp in df.groupby("series"):
        print(f"\n=== {sname} ({int((grp['status']=='ok').sum())}/{len(grp)} models ok) ===")
        ok = grp[grp["status"] == "ok"].sort_values("mae")
        if not ok.empty:
            cols = ["model", "mae", "rmse", "mape", "mase", "dir_acc", "elapsed_s"]
            print(ok[cols].to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        skipped = grp[grp["status"] != "ok"]
        if not skipped.empty:
            print("(not run)")
            for _, row in skipped.iterrows():
                print(f"  - {row['model']}: {row['status']} — {row.get('detail','')}")

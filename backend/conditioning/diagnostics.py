"""Diagnostics artifact (Part 8).

Produces a single self-contained HTML page that visualizes the head-to-head
comparison between vanilla TimesFM 2 and the news-conditioned forecaster.
Singh's framing was "you can see on a day-to-day basis how it is tracking" —
this file is what that statement looks like as a viewable artifact.

What's in the page
------------------
1. **Headline metrics** — vanilla vs conditioned MAE, RMSE, dir-acc, win rate.
2. **Per-day MAE curves** — both replays as line series so you can see where
   the adapter is helping and where it's hurting.
3. **Per-day MAE delta** (vanilla − conditioned) — sorted descending, shaded
   green where the adapter wins and red where it loses.
4. **Per-horizon-step MAE** — bar chart of MAE at horizon step 1, 2, …, H so
   you can see if the adapter is more useful at short or long horizons.
5. **Top-10 win and loss days** with the news context — Plotly + plain HTML
   tables, no framework. Lets the reader eyeball whether news content on big
   wins/losses looks intuitive.

We use Plotly via CDN (the same approach the existing dashboard uses) so the
HTML is fully self-contained and openable from disk.
"""
from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .alignment import AlignedSample
from .baseline import ReplayResult
from .replay import CompareResult


# --- HTML scaffolding ---------------------------------------------------------
_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>Conditioning diagnostics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg:#111418; --panel:#1a1d22; --panel-2:#15181c; --line:#2a2e34;
    --line-2:#353a41; --ink:#d6d8db; --ink-soft:#8a8e94; --ink-dim:#5e6268;
    --accent:#5a8cc4; --green:#5fa572; --red:#c46060; --amber:#c89860;
  }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--ink);
         font:13px/1.45 ui-sans-serif, system-ui, sans-serif;
         font-feature-settings:"tnum"; }}
  h1 {{ font-size:18px; font-weight:600; margin:0 0 6px; }}
  h2 {{ font-size:13px; font-weight:600; color:var(--ink-soft);
        text-transform:uppercase; letter-spacing:0.5px;
        border-bottom:1px solid var(--line); padding-bottom:6px; margin:24px 0 12px; }}
  .sub {{ color:var(--ink-soft); margin:0 0 18px; font-size:12px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:4px;
            padding:14px 16px; margin-bottom:14px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px; }}
  .kpi  {{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:12px 14px; }}
  .kpi-l {{ font-size:10px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:0.5px; }}
  .kpi-v {{ font-size:20px; font-weight:500; margin-top:3px; font-variant-numeric:tabular-nums; }}
  .kpi-d {{ font-size:11px; color:var(--ink-soft); font-variant-numeric:tabular-nums; }}
  .pos {{ color:var(--green); }}  .neg {{ color:var(--red); }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
  th,td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; font-size:12px; }}
  th  {{ font-size:10px; text-transform:uppercase; letter-spacing:0.5px; color:var(--ink-soft); }}
  .num {{ text-align:right; }}
  .chart {{ width:100%; height:340px; }}
</style></head>
<body>

<h1>News-Conditioned TimesFM 2 — Diagnostics</h1>
<p class="sub">Vanilla {vanilla_name} vs {cond_name}.
   Replay window: {first_t} → {last_t} ({n} samples, horizon = {H}).</p>

<div class="kpis">
  <div class="kpi"><div class="kpi-l">MAE (returns)</div>
       <div class="kpi-v">{mae_v:.5f} → {mae_c:.5f}</div>
       <div class="kpi-d {mae_cls}">Δ {mae_delta:+.5f}</div></div>
  <div class="kpi"><div class="kpi-l">RMSE (returns)</div>
       <div class="kpi-v">{rmse_v:.5f} → {rmse_c:.5f}</div>
       <div class="kpi-d {rmse_cls}">Δ {rmse_delta:+.5f}</div></div>
  <div class="kpi"><div class="kpi-l">Directional accuracy</div>
       <div class="kpi-v">{da_v:.3f} → {da_c:.3f}</div>
       <div class="kpi-d {da_cls}">Δ {da_delta:+.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Win rate (vs vanilla)</div>
       <div class="kpi-v">{win:.2%}</div>
       <div class="kpi-d">days conditioned MAE &lt; vanilla MAE</div></div>
</div>

<h2>Per-day MAE (return space)</h2>
<div class="panel"><div id="chart-perday" class="chart"></div></div>

<h2>Per-day MAE delta (vanilla − conditioned)</h2>
<div class="panel"><div id="chart-delta" class="chart"></div></div>

<h2>MAE by horizon step</h2>
<div class="panel"><div id="chart-horizon" class="chart"></div></div>

<h2>Top-10 win days (largest MAE reduction)</h2>
<div class="panel">{wins_table}</div>

<h2>Top-10 loss days (largest MAE regression)</h2>
<div class="panel">{losses_table}</div>

<script>
  const PLOTLY_FONT = {{ color:"#d6d8db", family:"ui-sans-serif, system-ui, sans-serif", size:11 }};
  const AXIS = {{ gridcolor:"#2a2e34", linecolor:"#2a2e34", zerolinecolor:"#2a2e34" }};
  const LAYOUT_BASE = {{
    margin:{{t:10,r:20,b:35,l:60}},
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:PLOTLY_FONT,
    xaxis:AXIS, yaxis:AXIS,
    legend:{{ orientation:"h", x:0, y:1.10, font:{{size:11}} }},
  }};

  Plotly.react("chart-perday", {perday_traces}, Object.assign({{}}, LAYOUT_BASE),
               {{displayModeBar:false, responsive:true}});
  Plotly.react("chart-delta",  {delta_traces},  Object.assign({{}}, LAYOUT_BASE),
               {{displayModeBar:false, responsive:true}});
  Plotly.react("chart-horizon",{horizon_traces},Object.assign({{}}, LAYOUT_BASE),
               {{displayModeBar:false, responsive:true}});
</script>
</body></html>
"""


def _table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return '<div style="color:var(--ink-soft)">No rows.</div>'
    head = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    body = ""
    for r in rows:
        cells = ""
        for c in cols:
            v = r.get(c, "")
            klass = "num" if isinstance(v, (int, float, np.floating, np.integer)) else ""
            cells += f'<td class="{klass}">{html.escape(str(v))}</td>'
        body += f"<tr>{cells}</tr>"
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _summarize_news_for_day(s: AlignedSample, max_titles: int = 3) -> str:
    if not s.news:
        return "—"
    parts = [
        f"{n.title[:90]}{'…' if len(n.title) > 90 else ''} <span style='color:var(--ink-dim)'>({n.sentiment:+.2f})</span>"
        for n in sorted(s.news, key=lambda r: r.published_at)[-max_titles:]
    ]
    return " · ".join(parts)


def render_html(
    vanilla: ReplayResult,
    conditioned: ReplayResult,
    cmp: CompareResult,
    samples: Sequence[AlignedSample],
    *,
    out_path: Path | str,
    vanilla_name: Optional[str] = None,
    cond_name: Optional[str] = None,
) -> Path:
    """Render the diagnostics HTML and write it to disk."""
    n = cmp.n_samples
    H = cmp.horizon
    t_index = pd.to_datetime(cmp.per_day_t, unit="ns")
    first_t, last_t = t_index.min().date(), t_index.max().date()

    # --- per-day MAE chart ---
    perday_traces = json.dumps([
        dict(x=[d.isoformat() for d in t_index],
             y=cmp.per_day_mae_vanilla.tolist(),
             name=vanilla_name or vanilla.forecaster,
             line=dict(color="#8a8e94", width=1.4),
             mode="lines"),
        dict(x=[d.isoformat() for d in t_index],
             y=cmp.per_day_mae_conditioned.tolist(),
             name=cond_name or conditioned.forecaster,
             line=dict(color="#5a8cc4", width=1.6),
             mode="lines"),
    ])

    # --- delta sorted descending ---
    delta = cmp.per_day_mae_vanilla - cmp.per_day_mae_conditioned
    order = np.argsort(-delta)
    delta_sorted = delta[order]
    colors = ["#5fa572" if d >= 0 else "#c46060" for d in delta_sorted]
    delta_traces = json.dumps([
        dict(x=list(range(len(delta_sorted))),
             y=delta_sorted.tolist(),
             type="bar",
             marker=dict(color=colors),
             name="Δ MAE (vanilla − cond)"),
    ])

    # --- per-horizon-step MAE ---
    v_h = np.mean(np.abs(vanilla.forecast_returns - vanilla.target_returns), axis=0)
    c_h = np.mean(np.abs(conditioned.forecast_returns - conditioned.target_returns), axis=0)
    horizon_traces = json.dumps([
        dict(x=list(range(1, H + 1)), y=v_h.tolist(), type="bar",
             name=vanilla_name or vanilla.forecaster, marker=dict(color="#8a8e94")),
        dict(x=list(range(1, H + 1)), y=c_h.tolist(), type="bar",
             name=cond_name or conditioned.forecaster, marker=dict(color="#5a8cc4")),
    ])

    # --- win/loss tables ---
    sample_by_t = {s.t.value: s for s in samples}
    rows: List[dict] = []
    for k, idx in enumerate(order):
        s = sample_by_t.get(int(cmp.per_day_t[idx]))
        if s is None:
            continue
        rows.append({
            "rank": k + 1,
            "date": pd.Timestamp(int(cmp.per_day_t[idx])).date().isoformat(),
            "vanilla MAE": f"{cmp.per_day_mae_vanilla[idx]:.5f}",
            "cond MAE": f"{cmp.per_day_mae_conditioned[idx]:.5f}",
            "Δ MAE": f"{delta[idx]:+.5f}",
            "n news": s.n_news(),
            "news (most recent 3)": _summarize_news_for_day(s),
        })

    cols = ["rank", "date", "vanilla MAE", "cond MAE", "Δ MAE", "n news", "news (most recent 3)"]
    wins = _table(rows[:10], cols)
    # losses are the bottom of the same sort, in ascending Δ
    losses_rows = list(reversed(rows[-10:]))
    losses = _table(losses_rows, cols)

    def cls(d: float) -> str:
        return "neg" if d < 0 else ("pos" if d > 0 else "")

    page = _PAGE_TEMPLATE.format(
        vanilla_name=html.escape(vanilla_name or vanilla.forecaster),
        cond_name=html.escape(cond_name or conditioned.forecaster),
        first_t=first_t, last_t=last_t, n=n, H=H,
        mae_v=cmp.mae_vanilla, mae_c=cmp.mae_conditioned, mae_delta=cmp.mae_delta,
        mae_cls=cls(-cmp.mae_delta),
        rmse_v=cmp.rmse_vanilla, rmse_c=cmp.rmse_conditioned, rmse_delta=cmp.rmse_delta,
        rmse_cls=cls(-cmp.rmse_delta),
        da_v=cmp.dir_acc_vanilla, da_c=cmp.dir_acc_conditioned, da_delta=cmp.dir_acc_delta,
        da_cls=cls(cmp.dir_acc_delta),
        win=cmp.win_rate,
        perday_traces=perday_traces,
        delta_traces=delta_traces,
        horizon_traces=horizon_traces,
        wins_table=wins,
        losses_table=losses,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out

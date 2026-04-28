/* News-Engine dashboard logic. Single file, no build step. */
(() => {
  const fmtMoney = (n) => n == null ? "—" : "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
  const fmtPct = (n, dp = 2) => n == null ? "—" : (n * 100).toFixed(dp) + "%";
  const fmtNum = (n, dp = 2) => n == null ? "—" : Number(n).toFixed(dp);
  const fmtTime = (iso) => new Date(iso).toLocaleString();
  const relTime = (iso) => {
    const d = new Date(iso);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return Math.floor(diff) + "s ago";
    if (diff < 3600) return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
  };

  const api = {
    async get(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(path + " → " + r.status);
      return r.json();
    },
    async post(path, body) {
      const r = await fetch(path, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!r.ok) throw new Error(path + " → " + r.status);
      return r.json();
    },
  };

  // -------- tabs --------
  const views = document.querySelectorAll(".view");
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.view;
      views.forEach((v) => v.classList.toggle("hidden", v.id !== "view-" + target));
      if (target === "news") loadNews();
      if (target === "trades") loadTrades();
      if (target === "backtest") loadRuns();
      if (target === "forecast") initForecastView();
    });
  });

  // -------- websocket --------
  const dot = document.getElementById("ws-dot");
  const dotLabel = document.getElementById("ws-label");

  function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws");
    ws.onopen = () => { dot.classList.add("ok"); dot.classList.remove("err"); dotLabel.textContent = "live"; };
    ws.onclose = () => {
      dot.classList.remove("ok"); dot.classList.add("err"); dotLabel.textContent = "disconnected";
      setTimeout(connectWS, 3000);
    };
    ws.onerror = () => { dot.classList.remove("ok"); dot.classList.add("err"); };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.event === "portfolio_tick") {
          refreshLive();
        } else if (msg.event === "boot") {
          refreshLive();
        }
      } catch (_) { /* ignore */ }
    };
    // keepalive
    setInterval(() => { if (ws.readyState === 1) ws.send("ping"); }, 25000);
  }
  connectWS();

  // -------- LIVE VIEW --------
  async function refreshLive() {
    try {
      const [status, curve, news, health, stats] = await Promise.all([
        api.get("/api/portfolio/status"),
        api.get("/api/portfolio/equity-curve"),
        api.get("/api/news?limit=10"),
        api.get("/api/health"),
        api.get("/api/news/stats"),
      ]);

      document.getElementById("kpi-equity").textContent = fmtMoney(status.equity);
      const pnl = status.pnl || 0;
      const pnlEl = document.getElementById("kpi-pnl");
      pnlEl.textContent = (pnl >= 0 ? "+" : "") + fmtMoney(pnl) + " P&L";
      pnlEl.className = "kpi-sub " + (pnl >= 0 ? "pos" : "neg");

      document.getElementById("kpi-return").textContent = fmtPct(status.return_pct, 2);
      document.getElementById("kpi-bench").textContent = fmtMoney(status.benchmark_equity);
      const alpha = (status.equity - status.benchmark_equity);
      const alphaEl = document.getElementById("kpi-alpha");
      alphaEl.textContent = (alpha >= 0 ? "α +" : "α ") + fmtMoney(alpha) + " vs SPY";
      alphaEl.className = "kpi-sub " + (alpha >= 0 ? "pos" : "neg");

      document.getElementById("kpi-news24").textContent = (stats.last_24h ?? 0).toLocaleString();
      const provs = Object.entries(stats.by_source || {}).map(([k, v]) => `${k}:${v}`).join(" · ");
      document.getElementById("kpi-providers").textContent = provs || "no items yet";

      drawEquity(curve);
      drawAllocation(status.holdings || []);
      renderHoldings(status.holdings || []);
      renderSignals(news);
      document.getElementById("last-update").textContent = "updated " + new Date().toLocaleTimeString();
    } catch (e) {
      console.error("refreshLive", e);
    }
  }

  function drawEquity(curve) {
    const xs = curve.map((p) => p.ts);
    const s = curve.map((p) => p.equity);
    const b = curve.map((p) => p.benchmark_equity);

    // When the market is closed (weekends, holidays) every snapshot writes
    // essentially the same equity and Plotly's auto-range zooms into the
    // sub-cent floating-point noise — making a flat book look like a huge
    // spike. Pad the range to at least ±0.5% of the median so "flat" actually
    // reads as flat.
    const all = s.concat(b).filter((v) => Number.isFinite(v) && v > 0);
    let yrange;
    if (all.length) {
      const mid = all.sort((a, b) => a - b)[Math.floor(all.length / 2)];
      const lo = Math.min(...all);
      const hi = Math.max(...all);
      const minPad = Math.max(mid * 0.005, 500); // at least $500 or 0.5% of equity
      yrange = [Math.min(lo, mid - minPad), Math.max(hi, mid + minPad)];
    }

    const layout = {
      margin: { t: 10, r: 20, b: 35, l: 72 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d6d8db", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
      xaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34" },
      yaxis: {
        gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34",
        tickprefix: "$",
        tickformat: "~s",     // compact SI: 1M, 1.2M, 900k, etc.
        separatethousands: true,
        range: yrange,
      },
      legend: { orientation: "h", x: 0, y: 1.1, font: { size: 11 } },
    };
    const traces = [
      { x: xs, y: s, name: "Strategy", line: { color: "#5a8cc4", width: 1.6 }, hovertemplate: "$%{y:,.0f}<extra></extra>" },
      { x: xs, y: b, name: "SPY benchmark", line: { color: "#8a8e94", width: 1.4, dash: "dot" }, hovertemplate: "$%{y:,.0f}<extra></extra>" },
    ];
    Plotly.react("chart-equity", traces, layout, { displayModeBar: false, responsive: true });
  }

  function drawAllocation(holdings) {
    const data = holdings.filter((h) => h.value > 1);
    if (!data.length) {
      Plotly.react("chart-allocation", [], {
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        annotations: [{ text: "No holdings yet", showarrow: false, font: { color: "#8c98b1" } }],
      }, { displayModeBar: false, responsive: true });
      return;
    }
    Plotly.react("chart-allocation", [{
      labels: data.map((h) => h.ticker),
      values: data.map((h) => h.value),
      type: "pie", hole: 0.6,
      textinfo: "label+percent",
      textfont: { size: 11, color: "#d6d8db" },
      hovertemplate: "%{label}: $%{value:,.0f}<extra></extra>",
      // Muted, neutral palette — no neon, no gradients
      marker: {
        colors: ["#5a8cc4","#7a9bbf","#8ea69a","#b89876","#a07f6c","#9a8aa8","#6e7a86","#aab2b8","#7a8590","#5e6e7a","#90785e","#b09a82"],
        line: { color: "#1a1d22", width: 2 },
      },
    }], {
      margin: { t: 10, r: 10, b: 10, l: 10 },
      paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#d6d8db", family: "ui-sans-serif, system-ui, sans-serif" },
      showlegend: false,
    }, { displayModeBar: false, responsive: true });
  }

  function renderHoldings(holdings) {
    const tbody = document.querySelector("#holdings-tbl tbody");
    tbody.innerHTML = "";
    if (!holdings.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:#8c98b1">No positions yet — waiting for signals.</td></tr>';
      return;
    }
    for (const h of holdings) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${h.ticker}</td>
        <td>${fmtPct(h.weight, 1)}</td>
        <td>${fmtNum(h.qty, 0)}</td>
        <td>${fmtMoney(h.price)}</td>
        <td>${fmtMoney(h.value)}</td>`;
      tbody.appendChild(tr);
    }
  }

  function renderSignals(items) {
    const feed = document.getElementById("signals-feed");
    feed.innerHTML = "";
    if (!items.length) {
      feed.innerHTML = '<div style="color:#8c98b1">No news ingested yet — it will populate within a minute.</div>';
      return;
    }
    for (const it of items) {
      feed.appendChild(newsCard(it));
    }
  }

  function newsCard(it) {
    const div = document.createElement("div");
    div.className = "feed-item";
    const sClass = it.sentiment > 0.1 ? "pos" : it.sentiment < -0.1 ? "neg" : "neu";
    const sLabel = it.sentiment > 0.1 ? "bull" : it.sentiment < -0.1 ? "bear" : "neut";
    const tickers = (it.tickers || []).map((t) => `<span class="chip">${t}</span>`).join(" ");
    div.innerHTML = `
      <div class="title">${it.url ? `<a href="${it.url}" target="_blank" rel="noopener">${escapeHtml(it.title)}</a>` : escapeHtml(it.title)}</div>
      <div class="meta">
        <span class="sentiment ${sClass}">${sLabel} ${it.sentiment.toFixed(2)}</span>
        <span>${it.source}</span>
        <span>${relTime(it.published_at)}</span>
        ${tickers}
      </div>`;
    return div;
  }

  function escapeHtml(s) {
    return (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  document.getElementById("btn-tick").addEventListener("click", async () => {
    const btn = document.getElementById("btn-tick");
    btn.disabled = true; btn.textContent = "Running…";
    try {
      await api.post("/api/portfolio/tick");
      await refreshLive();
    } finally {
      btn.disabled = false; btn.textContent = "Run allocation tick";
    }
  });

  // -------- NEWS VIEW --------
  async function loadNews() {
    const ticker = document.getElementById("news-filter").value.trim().toUpperCase();
    const url = ticker ? `/api/news?limit=100&ticker=${encodeURIComponent(ticker)}` : "/api/news?limit=100";
    try {
      const items = await api.get(url);
      const feed = document.getElementById("news-feed");
      feed.innerHTML = "";
      if (!items.length) {
        feed.innerHTML = '<div style="color:#8c98b1">No items match.</div>';
        return;
      }
      items.forEach((it) => feed.appendChild(newsCard(it)));
    } catch (e) {
      console.error(e);
    }
  }
  document.getElementById("btn-refresh-news").addEventListener("click", async () => {
    await api.post("/api/news/refresh");
    await loadNews();
  });
  document.getElementById("news-filter").addEventListener("change", loadNews);

  // -------- TRADES VIEW --------
  async function loadTrades() {
    const rows = await api.get("/api/portfolio/trades?limit=200");
    const tbody = document.querySelector("#trades-tbl tbody");
    tbody.innerHTML = "";
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="color:#8c98b1">No trades yet.</td></tr>';
      return;
    }
    for (const t of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${fmtTime(t.ts)}</td>
        <td>${t.ticker}</td>
        <td class="${t.side === 'BUY' ? 'side-buy' : 'side-sell'}">${t.side}</td>
        <td>${fmtNum(t.qty, 2)}</td>
        <td>${fmtMoney(t.price)}</td>
        <td>${escapeHtml(t.reason || "")}</td>`;
      tbody.appendChild(tr);
    }
  }

  // -------- BACKTEST VIEW --------
  const btForm = document.getElementById("bt-form");
  btForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const statusEl = document.getElementById("bt-status");
    statusEl.textContent = "Running…";
    const cfg = {
      universe: document.getElementById("bt-universe").value.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
      start: document.getElementById("bt-start").value,
      end: document.getElementById("bt-end").value,
      starting_cash: +document.getElementById("bt-cash").value,
      rebalance: document.getElementById("bt-rebal").value,
      transaction_cost_bps: +document.getElementById("bt-tc").value,
      max_weight: +document.getElementById("bt-maxw").value,
      benchmark: document.getElementById("bt-bench").value.toUpperCase(),
      use_news_signals: document.getElementById("bt-news").checked,
      long_only: true,
    };
    try {
      const res = await api.post("/api/backtest/run", cfg);
      renderBacktest(res);
      statusEl.textContent = "Done · run #" + res.id;
      await loadRuns();
    } catch (e) {
      statusEl.textContent = "Failed: " + e.message;
      console.error(e);
    }
  });

  function renderBacktest(res) {
    const m = res.metrics;
    const fields = [
      ["Total return", fmtPct(m.total_return, 2), m.total_return],
      ["Annualized", fmtPct(m.annual_return, 2), m.annual_return],
      ["Annual vol", fmtPct(m.annual_vol, 2), 0],
      ["Sharpe", fmtNum(m.sharpe, 2), m.sharpe - 1],
      ["Sortino", fmtNum(m.sortino, 2), m.sortino - 1],
      ["Max drawdown", fmtPct(m.max_drawdown, 2), m.max_drawdown],
      ["Calmar", fmtNum(m.calmar, 2), m.calmar],
      ["Alpha (annual)", fmtPct(m.alpha_annual, 2), m.alpha_annual],
      ["Beta", fmtNum(m.beta, 2), 0],
      ["Info ratio", fmtNum(m.information_ratio, 2), m.information_ratio],
      ["Hit rate vs bench", fmtPct(m.hit_rate, 1), m.hit_rate - 0.5],
      ["SPY total return", fmtPct(m.benchmark_total_return, 2), m.benchmark_total_return],
    ];
    const grid = document.getElementById("bt-metrics");
    grid.innerHTML = "";
    for (const [label, val, sign] of fields) {
      const d = document.createElement("div");
      d.className = "metric";
      d.innerHTML = `<div class="metric-label">${label}</div><div class="metric-value ${sign>0?'pos':sign<0?'neg':''}">${val}</div>`;
      grid.appendChild(d);
    }

    const xs = res.equity_curve.map((p) => p.ts);
    const s = res.equity_curve.map((p) => p.strategy);
    const b = res.equity_curve.map((p) => p.benchmark);
    Plotly.react("bt-chart", [
      { x: xs, y: s, name: "Strategy",  line: { color: "#5a8cc4", width: 1.6 } },
      { x: xs, y: b, name: "Benchmark", line: { color: "#8a8e94", width: 1.4, dash: "dot" } },
    ], {
      margin: { t: 30, r: 20, b: 35, l: 72 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d6d8db", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
      xaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34" },
      yaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34", tickprefix: "$", tickformat: "~s", separatethousands: true },
      legend: { orientation: "h", x: 0, y: 1.1, font: { size: 11 } },
      title: { text: "Equity curve · strategy vs benchmark", font: { color: "#8a8e94", size: 11 } },
    }, { displayModeBar: false, responsive: true });

    const dd_xs = res.drawdown_curve.map((p) => p.ts);
    const dd_ys = res.drawdown_curve.map((p) => p.dd);
    Plotly.react("bt-dd", [
      { x: dd_xs, y: dd_ys, name: "Drawdown", fill: "tozeroy", line: { color: "#c46060", width: 1 }, fillcolor: "rgba(196,96,96,0.18)" },
    ], {
      margin: { t: 30, r: 20, b: 35, l: 60 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d6d8db", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
      xaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34" },
      yaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34", tickformat: ".0%", rangemode: "tozero" },
      title: { text: "Underwater (drawdown) curve", font: { color: "#8a8e94", size: 11 } },
    }, { displayModeBar: false, responsive: true });
  }

  async function loadRuns() {
    try {
      const rows = await api.get("/api/backtest/runs?limit=20");
      const tbody = document.querySelector("#runs-tbl tbody");
      tbody.innerHTML = "";
      if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:#8c98b1">No runs yet.</td></tr>';
        return;
      }
      for (const r of rows) {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer";
        tr.innerHTML = `
          <td>#${r.id}</td>
          <td>${fmtTime(r.created_at)}</td>
          <td>${(r.config.universe || []).slice(0, 6).join(", ")}${r.config.universe.length > 6 ? "…" : ""}</td>
          <td>${r.config.start} → ${r.config.end}</td>
          <td class="${r.metrics.total_return>=0?'side-buy':'side-sell'}">${fmtPct(r.metrics.total_return, 2)}</td>
          <td>${fmtNum(r.metrics.sharpe, 2)}</td>
          <td>${fmtPct(r.metrics.max_drawdown, 2)}</td>`;
        tr.addEventListener("click", async () => {
          const full = await api.get("/api/backtest/runs/" + r.id);
          const mock = { id: full.id, metrics: full.metrics, equity_curve: full.equity_curve,
                         drawdown_curve: full.equity_curve.map((p, i, arr) => {
                           const peak = Math.max(...arr.slice(0, i + 1).map((q) => q.strategy));
                           return { ts: p.ts, dd: p.strategy / peak - 1 };
                         }) };
          renderBacktest(mock);
        });
        tbody.appendChild(tr);
      }
    } catch (e) { console.error(e); }
  }

  // -------- FORECAST VIEW --------
  // Fast-running models (used by the "fast only" preset). Everything not in
  // this set is either neural (MLP/LSTM) or foundation (TimesFM) and can take
  // several seconds on first fit.
  const FC_FAST_MODELS = new Set(["naive_last", "naive_mean", "naive_drift", "ewma", "holt_winters", "arima"]);

  // Plotly color palette for per-model forecast lines. Long enough to cover
  // all 9 models even if a user unchecks/re-checks.
  // Restrained palette — distinguishable but no neon, no gradients.
  const FC_COLORS = [
    "#5a8cc4", "#8a8e94", "#7a9bbf", "#b89876",
    "#9a8aa8", "#5fa572", "#c89860", "#6e8590", "#a07f6c",
  ];

  let fcInitialized = false;
  let fcModelsCache = null;

  async function initForecastView() {
    if (fcInitialized) return;
    fcInitialized = true;

    // Populate ticker quick-pick from /api/forecast/universe (falls back to
    // /api/health if the former 500s for any reason).
    try {
      const u = await api.get("/api/forecast/universe");
      const pick = document.getElementById("fc-ticker-pick");
      for (const t of (u.universe || [])) {
        const opt = document.createElement("option");
        opt.value = t; opt.textContent = t;
        pick.appendChild(opt);
      }
      pick.addEventListener("change", () => {
        const v = pick.value;
        if (v) document.getElementById("fc-ticker").value = v;
      });
    } catch (e) { console.warn("universe load", e); }

    // Populate model checkbox grid.
    try {
      fcModelsCache = await api.get("/api/forecast/models");
      renderModelGrid(fcModelsCache);
    } catch (e) { console.error("models load", e); }

    // Sub-tabs.
    document.querySelectorAll(".subtab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".subtab").forEach((t) => t.classList.remove("active"));
        btn.classList.add("active");
        const target = btn.dataset.subview;
        document.querySelectorAll(".subview").forEach((v) => {
          v.classList.toggle("hidden", v.id !== "subview-" + target);
        });
      });
    });

    // Model select helpers.
    document.getElementById("fc-models-all").addEventListener("click", () => setModelSelection("all"));
    document.getElementById("fc-models-none").addEventListener("click", () => setModelSelection("none"));
    document.getElementById("fc-models-fast").addEventListener("click", () => setModelSelection("fast"));

    // Run button.
    document.getElementById("fc-form").addEventListener("submit", runForecast);

    // Leaderboard run button.
    document.getElementById("btn-run-leaderboard").addEventListener("click", runLeaderboard);
  }

  function renderModelGrid(models) {
    const grid = document.getElementById("fc-models-grid");
    grid.innerHTML = "";
    for (const m of models) {
      const id = "fc-m-" + m.name;
      const wrap = document.createElement("label");
      wrap.className = "fc-model" + (m.is_available ? "" : " unavailable");
      // Default selection: all fast models + timesfm, leaving MLP/LSTM off
      // so a fresh "Run" is quick.
      const defaultChecked = m.is_available && (FC_FAST_MODELS.has(m.name) || m.name === "timesfm");
      wrap.innerHTML = `
        <input type="checkbox" id="${id}" value="${m.name}" ${defaultChecked ? "checked" : ""} ${m.is_available ? "" : "disabled"} />
        <div class="fc-model-main">
          <div class="fc-model-name">${m.name}
            <span class="fc-family">${m.family}</span>
            ${m.is_available ? "" : '<span class="fc-unavail">unavailable</span>'}
          </div>
          <div class="fc-model-blurb">${escapeHtml(m.blurb || "")}</div>
        </div>`;
      grid.appendChild(wrap);
    }
  }

  function setModelSelection(mode) {
    document.querySelectorAll("#fc-models-grid input[type=checkbox]").forEach((cb) => {
      if (cb.disabled) { cb.checked = false; return; }
      if (mode === "all") cb.checked = true;
      else if (mode === "none") cb.checked = false;
      else if (mode === "fast") cb.checked = FC_FAST_MODELS.has(cb.value);
    });
  }

  function selectedModels() {
    return Array.from(document.querySelectorAll("#fc-models-grid input[type=checkbox]:checked")).map((cb) => cb.value);
  }

  async function runForecast(ev) {
    ev.preventDefault();
    const ticker = (document.getElementById("fc-ticker").value || document.getElementById("fc-ticker-pick").value || "").trim().toUpperCase();
    if (!ticker) { setFcStatus("Pick or type a ticker", true); return; }
    const models = selectedModels();
    if (!models.length) { setFcStatus("Select at least one model", true); return; }

    const body = {
      ticker,
      models,
      horizon: +document.getElementById("fc-horizon").value,
      eval_tail: +document.getElementById("fc-evaltail").value,
      period: document.getElementById("fc-period").value,
    };
    setFcStatus("Running… (first TimesFM call downloads 900MB if not cached)", false);
    try {
      const res = await api.post("/api/forecast/run", body);
      renderForecast(res);
      const okCount = res.forecasts.filter((f) => f.status === "ok").length;
      setFcStatus(`Done · ${okCount}/${res.forecasts.length} models ok · n=${res.history_len}`, false);
    } catch (e) {
      console.error(e);
      setFcStatus("Failed: " + e.message, true);
    }
  }

  function setFcStatus(msg, isErr) {
    const el = document.getElementById("fc-status");
    el.textContent = msg;
    el.style.color = isErr ? "var(--red)" : "var(--ink-soft)";
  }

  function renderForecast(res) {
    const history = res.history;
    const n = history.length;
    const fitLen = res.fit_len;
    const evalTail = res.eval_tail;
    const horizon = res.horizon;

    // X axis: integer trading-day offsets from the start of history.
    const xHist = Array.from({ length: n }, (_, i) => i);
    const traces = [];

    // History trace.
    traces.push({
      x: xHist, y: history,
      name: `${res.ticker} history`,
      line: { color: "#d6d8db", width: 1.4 },
      hovertemplate: "day %{x}: $%{y:,.2f}<extra></extra>",
    });

    // Held-out tail marker (if any).
    if (evalTail > 0) {
      traces.push({
        x: [fitLen - 0.5, fitLen - 0.5],
        y: [Math.min(...history) * 0.98, Math.max(...history) * 1.02],
        name: "fit / tail split",
        mode: "lines",
        line: { color: "#5e6268", dash: "dot", width: 1 },
        hoverinfo: "skip",
        showlegend: true,
      });
    }

    // Forecast x-axis starts at day `fitLen` (first held-out point / first
    // forward point depending on eval_tail).
    const xFc = Array.from({ length: horizon }, (_, i) => fitLen + i);

    // Per-model forecast lines + confidence bands.
    const metricsRows = [];
    let colorIdx = 0;
    for (const fc of res.forecasts) {
      const color = FC_COLORS[colorIdx % FC_COLORS.length];
      colorIdx += 1;

      if (fc.status !== "ok" || !fc.point) {
        metricsRows.push({ model: fc.model, family: fc.family || "?", status: fc.status, error: fc.error });
        continue;
      }

      // Confidence band (behind the point line).
      if (fc.lower && fc.upper) {
        traces.push({
          x: xFc.concat([...xFc].reverse()),
          y: fc.lower.concat([...fc.upper].reverse()),
          name: `${fc.model} band`,
          fill: "toself",
          fillcolor: hexToRgba(color, 0.13),
          line: { color: "rgba(0,0,0,0)" },
          hoverinfo: "skip",
          showlegend: false,
        });
      }

      traces.push({
        x: xFc, y: fc.point,
        name: fc.model,
        line: { color, width: 1.7 },
        hovertemplate: `${fc.model} · day %{x}: $%{y:,.2f}<extra></extra>`,
      });

      metricsRows.push({
        model: fc.model,
        family: fc.family,
        status: "ok",
        metrics: fc.metrics || null,
      });
    }

    const caption = evalTail > 0
      ? `${res.ticker} · fit on first ${fitLen} days, forecast ${horizon} days covering ${evalTail}-day held-out tail`
      : `${res.ticker} · fit on full ${n} days, forward forecast ${horizon} days (no held-out tail)`;
    document.getElementById("fc-chart-caption").textContent = caption;

    Plotly.react("fc-chart", traces, {
      margin: { t: 10, r: 20, b: 40, l: 60 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#d6d8db", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
      xaxis: {
        gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34",
        title: { text: "trading days (from start of history)", font: { size: 10, color: "#8a8e94" } },
      },
      yaxis: { gridcolor: "#2a2e34", linecolor: "#2a2e34", zerolinecolor: "#2a2e34", tickformat: "$,.2f" },
      legend: { orientation: "h", x: 0, y: 1.08, font: { size: 10 } },
      hovermode: "x unified",
    }, { displayModeBar: false, responsive: true });

    renderForecastMetrics(metricsRows);
  }

  function renderForecastMetrics(rows) {
    const tbody = document.querySelector("#fc-metrics-tbl tbody");
    tbody.innerHTML = "";

    // Sort OK rows by MAE (ascending); append skipped/error rows at the bottom.
    const ok = rows.filter((r) => r.status === "ok" && r.metrics)
                   .sort((a, b) => (a.metrics.mae ?? Infinity) - (b.metrics.mae ?? Infinity));
    const bad = rows.filter((r) => r.status !== "ok" || !r.metrics);

    for (const r of ok) {
      const m = r.metrics || {};
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${r.model}</strong></td>
        <td><span class="pill pill-${r.family}">${r.family}</span></td>
        <td>${fmtMetric(m.mae)}</td>
        <td>${fmtMetric(m.rmse)}</td>
        <td>${fmtPct(m.mape, 2)}</td>
        <td>${fmtMetric(m.mase)}</td>
        <td>${fmtPct(m.dir_acc, 1)}</td>
        <td><span class="pill pill-ok">ok</span></td>`;
      tbody.appendChild(tr);
    }
    for (const r of bad) {
      const tr = document.createElement("tr");
      const cls = r.status === "skipped" ? "pill-warn" : "pill-err";
      tr.innerHTML = `
        <td><strong>${r.model}</strong></td>
        <td><span class="pill pill-${r.family || "?"}">${r.family || "?"}</span></td>
        <td colspan="5" style="color:var(--ink-soft)">${escapeHtml(r.error || r.status)}</td>
        <td><span class="pill ${cls}">${r.status}</span></td>`;
      tbody.appendChild(tr);
    }
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="color:#8c98b1">No forecasts yet.</td></tr>';
    }
  }

  function fmtMetric(v) {
    return (v == null || !Number.isFinite(v)) ? "—" : Number(v).toFixed(4);
  }

  // -------- Leaderboard sub-view --------
  let lbInFlight = false;
  async function runLeaderboard() {
    if (lbInFlight) return;
    lbInFlight = true;
    const btn = document.getElementById("btn-run-leaderboard");
    const statusEl = document.getElementById("lb-status");
    btn.disabled = true;
    statusEl.textContent = "Running toy eval… this takes ~30–60 seconds on first TimesFM call.";
    const horizon = +document.getElementById("lb-horizon").value || 32;
    const n = +document.getElementById("lb-n").value || 400;
    try {
      const res = await api.get(`/api/forecast/toy-eval?horizon=${horizon}&n=${n}`);
      renderLeaderboard(res);
      statusEl.textContent = `Done · ${res.series.length} series × ${res.models.length} models`;
    } catch (e) {
      console.error(e);
      statusEl.textContent = "Failed: " + e.message;
    } finally {
      btn.disabled = false;
      lbInFlight = false;
    }
  }

  function renderLeaderboard(res) {
    // Overall table.
    const overallTbody = document.querySelector("#lb-overall-tbl tbody");
    overallTbody.innerHTML = "";
    if (!res.overall.length) {
      overallTbody.innerHTML = '<tr><td colspan="6" style="color:#8c98b1">No models produced any results.</td></tr>';
    } else {
      for (const r of res.overall) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${r.model}</strong></td>
          <td>${fmtMetric(r.mae)}</td>
          <td>${fmtMetric(r.rmse)}</td>
          <td>${fmtPct(r.mape, 2)}</td>
          <td>${fmtMetric(r.mase)}</td>
          <td>${fmtPct(r.dir_acc, 1)}</td>`;
        overallTbody.appendChild(tr);
      }
    }

    // Per-series panels.
    const container = document.getElementById("lb-per-series");
    container.innerHTML = "";
    for (const ps of res.per_series) {
      const block = document.createElement("div");
      block.className = "lb-series-block";
      const okRows = ps.ok.map((r) => `
        <tr>
          <td><strong>${r.model}</strong></td>
          <td>${fmtMetric(r.mae)}</td>
          <td>${fmtMetric(r.rmse)}</td>
          <td>${fmtPct(r.mape, 2)}</td>
          <td>${fmtMetric(r.mase)}</td>
          <td>${fmtPct(r.dir_acc, 1)}</td>
          <td style="color:var(--ink-soft)">${fmtMetric(r.elapsed_s)}s</td>
        </tr>`).join("");
      const skipRows = ps.skipped.map((r) => `
        <tr class="lb-skip">
          <td><strong>${r.model}</strong></td>
          <td colspan="5" style="color:var(--ink-soft)">${escapeHtml(r.detail || "")}</td>
          <td><span class="pill ${r.status === 'skipped' ? 'pill-warn' : 'pill-err'}">${r.status}</span></td>
        </tr>`).join("");
      block.innerHTML = `
        <h4 class="subhead">${ps.series} <span class="subhead-sub">${ps.ok.length}/${ps.ok.length + ps.skipped.length} ok</span></h4>
        <table class="tbl">
          <thead>
            <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>MASE</th><th>Dir-acc</th><th>Time</th></tr>
          </thead>
          <tbody>${okRows}${skipRows}</tbody>
        </table>`;
      container.appendChild(block);
    }
  }

  function hexToRgba(hex, a) {
    const h = hex.replace("#", "");
    const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    const r = parseInt(full.slice(0, 2), 16);
    const g = parseInt(full.slice(2, 4), 16);
    const b = parseInt(full.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // -------- boot --------
  refreshLive();
  setInterval(refreshLive, 30_000);
})();

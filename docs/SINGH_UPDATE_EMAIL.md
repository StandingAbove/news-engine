# Draft — May-7 update email to Prof. Singh

Subject: News-engine update before NYC

---

Hi Professor Singh,

Quick update on the news-conditioned TimesFM 2 work before I head to New York
on the 8th. Following your April 28 framing — news arriving day-by-day as a
signal that adjusts TimesFM 2's parameters/output, not just a feature
predicted alongside it.

## What's running now

The full plumbing for the supervised fine-tuning route is end-to-end. SPY-only
universe to start (FactSet daily, 1993–2026, 8.3k rows). Code lives in
`backend/conditioning/` and is decoupled from the original two sub-components
in the prescribed way — that package is the *only* place A (forecast/) and B
(news/) meet.

The pipeline:

1. **Alignment**. For each trading day t, build a sample = (512-day price
   history strictly before t, all news in [t − 7d, t), prices over [t, t+20]).
2. **Encoding**. News list → 16-dim stats vector (sentiment moments, recency
   decay, ticker/sector shares). Sentence-transformer encoder is wired but off
   by default to keep the deps minimal.
3. **Vanilla forecast**. TimesFM 2.5 (200M, frozen) produces a 20-step return
   path from the history.
4. **Adapter**. A small (~8k-param) MLP head takes
   (vanilla_returns, news_emb, history_features) and outputs an H-step
   return-space adjustment, soft-tanh-clipped to ±2%/step. Initialized
   ~zero so the conditioned forecast starts as identity to vanilla.
5. **Training**. Frozen TimesFM, supervised MSE on adjusted-vs-actual returns
   with a tiny L2 penalty on adjustment magnitude. Early stop on val loss.
6. **Replay**. Walk-forward over the last 1y holdout (252 trading days) for
   both vanilla and conditioned, log per-day MAE, RMSE, dir-acc.
7. **Diagnostics**. Self-contained HTML — KPI strip, per-day MAE curves,
   per-horizon-step MAE, top-10 win/loss days with the relevant headlines.

A driver script (`scripts/run_conditioning.py`) does all of the above end-to-
end.

On the synthetic-news harness check (no TimesFM, just to validate the
training & replay loops): 71% win rate, MAE drop from 0.00615 to 0.00606, and
directional accuracy 0.023 → 0.59. The numbers will move once I plug in
TimesFM 2 + the real news (the news side is the bottleneck — see below).

## What's on deck this week before NYC

- Run the full pipeline with TimesFM 2 + the live news stream as the
  adapter's training signal, on the 1y holdout. Replace the synthetic-news
  numbers above with real ones in the HTML report.
- Backfill historical news. The current DB only has the last few months of
  live-ingest data; for a meaningful train set I'll pull EODHD's news
  history endpoint to fill 2018–2025. (FactSet doesn't expose a clean news
  archive at my access level.)
- GRPO sketch. I wrote up the design as `docs/GRPO_DESIGN.md` — reward is
  delta-over-vanilla on directional accuracy + MAE reduction, with KL pull
  toward the supervised checkpoint as the reference. Not implemented yet;
  the supervised baseline is the gate I want to clear first.

## What I'd like your read on

1. Is SPY-only acceptable as the v1 demo, or would you rather I scope to
   ~10–20 sector ETFs (XL-_, country ETFs) before showing results?
2. Reward shape for GRPO — does the delta-over-vanilla framing make sense
   to you, or do you have a sharper one in mind?
3. The student you mentioned who's a TimesFM expert — would it be useful to
   loop them in on the adapter architecture (specifically: whether to keep
   it post-hoc on the output, or actually inject LoRA adapters into the
   foundation model)?

Repo: https://github.com/StandingAbove/news-signal (vsinghuw added)

I'll be on the East Coast May 8 → mid-June. Happy to do a Zoom anytime
that's good for you, or to leave a written status on the GitHub if email
works better while you're chasing the deadline.

Best,
Sam

# Sub-component B — Text feed synthesis

This is Dr. Singh's sub-component (2): a pipeline that takes a stream of
financial news headlines and synthesizes them into per-ticker directional
scores. Operates independently of the forecasting sub-component.

## Pipeline

```
  provider fetch ──► dedupe ──► sentiment ──► entity mapping ──► aggregation
  (3 providers)      (db)       (VADER+       (ticker /           (half-life
                                 finance       sector /            decay +
                                 lexicon)      region)             tanh squash)
```

Implementation lives in:

* `backend/news/providers.py` — Marketaux, NewsAPI.ai (Event Registry), EODHD
* `backend/news/aggregator.py` — fetch + dedupe + persist
* `backend/news/sentiment.py` — VADER with a finance-tuned lexicon
* `backend/news/mapping.py` — company / sector / region → ticker or ETF
* `backend/signals/engine.py` — per-ticker aggregation + squash

The same aggregation function that the live pipeline uses is reused in the
toy harness — one code path, two drivers.

## Toy harness

`backend/news/toy.py` contains 18 hand-written headlines (`TOY_HEADLINES`),
each tagged with:

* `expected_sentiment_sign` ∈ {-1, 0, +1}
* `expected_tickers` (direct, sector, or regional)
* a free-text `note` explaining why the case is interesting

The eval:

1. Scores every headline through VADER + finance lexicon; compares sign to
   expectation → **sentiment direction accuracy**.
2. Tags every headline via `extract_tickers()`; checks whether expected
   tags are a subset of returned tags → **ticker-tag recall**.
3. Feeds all scored items into `aggregate_signals()` spaced 1h apart, so
   half-life decay is nontrivial; squashes; checks per-ticker aggregate
   sign vs intuition → **aggregate-sign accuracy**.

Run:

```bash
./scripts/eval_text_feed.sh
```

Exits 0 only if all three metrics clear their soft floors (70% / 80% /
70%). Those floors are also enforced by `backend/tests/test_news_toy.py`.

## Moving to real data

Once toy accuracy looks right, the same pipeline is already wired to the
three live providers via `backend.news.aggregator.Aggregator`. Real-data
smoke happens on the v2 integrated stack (`./scripts/run.sh`), but the
text-feed component can also be driven directly:

```python
from backend.news.aggregator import Aggregator
ag = Aggregator()
items = await ag.fetch_all(limit_per_provider=25)
```

## Known caveats

* **Sentiment is VADER.** Dr. Singh's "forecasting model" ask is answered
  in sub-component A; the text-feed side uses VADER + a finance lexicon
  as a placeholder. Accuracy floors are chosen to reflect that — we
  expect regressions when genuinely ambiguous headlines show up. The
  planned upgrade is a small fine-tuned sentence classifier, parked until
  A stabilizes.
* **Regional spillover is coarse.** Hong Kong → EEM, ECB → EFA. Deliberately
  coarse because the universe only includes region ETFs; a finer mapping
  would need corresponding exposures.
* **Toy set is small (18 headlines).** Enough to catch regressions, not
  enough to train on. Intentional: this sub-component isn't learned from
  the toy set, it's validated against it.

## What "failing" looks like here

* Sentiment inversion on a lexicon-sensitive headline (e.g., "Apple warns
  on demand" scored positive) → the row shows `[+0.3] ... *` in the eval.
* A region spillover keyword gets added/removed and EEM/EFA tags shift →
  ticker recall drops below 80% floor and `test_ticker_recall_floor` fails.
* Aggregation inverts the sign of a clearly positive/negative ticker (bug
  in half-life or tanh logic) → `aggregate_sign_correct` drops below 66%.

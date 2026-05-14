"""News-conditioned TimesFM-2 fine-tuning experiment (per Singh, Apr-28 mtg).

Singh's framing: news arriving day-by-day is a signal that should *adjust the
parameters of TimesFM-2 itself*, not just be predicted alongside it. The
training objective is to learn, from historical (news_t, prices_{t..t+H})
pairs, a mechanism that conditions TimesFM-2 such that conditioned forecasts
beat vanilla TimesFM-2 on a held-out walk-forward window.

This package is the new headline. The existing `backend/forecast/` and
`backend/news/` packages stay where they are and feed into this one as
upstream baselines + raw news source. They do not import from each other.
"""

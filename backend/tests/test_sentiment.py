"""Unit tests for the finance-tuned sentiment layer."""
from __future__ import annotations

import pytest

from backend.news.sentiment import FinanceSentiment


@pytest.fixture(scope="module")
def sent():
    return FinanceSentiment()


def test_beat_estimates_is_positive(sent):
    assert sent.score("Apple beat estimates on strong iPhone demand") > 0.3


def test_bankruptcy_is_strongly_negative(sent):
    assert sent.score("Company files for bankruptcy amid debt default") < -0.5


def test_neutral_text_near_zero(sent):
    s = sent.score("The company held its annual shareholder meeting today.")
    assert -0.35 < s < 0.35


def test_empty_returns_zero(sent):
    assert sent.score("") == 0.0
    assert sent.score(None) == 0.0  # defensive


def test_blend_averages_with_provider(sent):
    # If both agree, result remains clearly positive.
    s = sent.blend("Strong earnings beat", provider_sentiment=0.8)
    assert s > 0.3


def test_blend_tolerates_bad_provider_value(sent):
    s1 = sent.blend("Strong earnings beat", provider_sentiment="NaN")
    s2 = sent.score("Strong earnings beat")
    assert s1 == pytest.approx(s2, abs=1e-6)


def test_blend_clamps_provider_value(sent):
    # Even if the provider sends something out of range, we cap it to [-1, 1].
    s = sent.blend("Neutral sentence.", provider_sentiment=-10.0)
    # Blended = 0.6 * neutral_own + 0.4 * (-1) → should land well below 0.
    assert s < -0.2

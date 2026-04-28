"""Background scheduler for news refresh + paper-trading ticks."""
from __future__ import annotations

import asyncio
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .core.logging import get_logger
from .news.aggregator import refresh_latest
from .news.providers import build_providers
from .portfolio.paper import get_trader

log = get_logger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None


async def news_refresh_job():
    providers = build_providers()
    try:
        n = await refresh_latest(providers, limit_per_provider=30)
        log.info("[scheduler] news refresh inserted %d rows", n)
    except Exception:  # noqa: BLE001
        log.exception("news_refresh_job failed")


async def paper_tick_job():
    try:
        trader = get_trader()
        summary = await trader.tick()
        log.info(
            "[scheduler] paper tick equity=$%.2f trades=%d signals=%d",
            summary["equity"], len(summary["trades"]), summary["signals_used"],
        )
    except Exception:  # noqa: BLE001
        log.exception("paper_tick_job failed")


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    s = AsyncIOScheduler()
    # News every 5 minutes (well under free-tier ceilings).
    s.add_job(news_refresh_job, "interval", minutes=5, id="news_refresh", next_run_time=None)
    # Paper-trade every 10 minutes.
    s.add_job(paper_tick_job, "interval", minutes=10, id="paper_tick", next_run_time=None)
    s.start()
    _scheduler = s
    log.info("Scheduler started")
    return s


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("Scheduler stopped")


async def trigger_initial_refresh():
    """Run one news refresh + paper tick immediately on startup."""
    await news_refresh_job()
    await paper_tick_job()

"""FastAPI entrypoint."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes_backtest import router as backtest_router
from .api.routes_forecast import router as forecast_router
from .api.routes_market import router as market_router
from .api.routes_news import router as news_router
from .api.routes_portfolio import router as portfolio_router
from .api.ws import broadcaster
from .core.config import PROJECT_ROOT, get_settings
from .core.db import init_db
from .core.logging import configure_logging, get_logger
from .portfolio.paper import get_trader
from .scheduler import start_scheduler, stop_scheduler, trigger_initial_refresh

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("--- News-Engine starting ---")
    await init_db()

    # Initialize the paper trader (loads last snapshot or creates fresh).
    await get_trader().initialize()

    # Kick off the scheduler but run an initial refresh in the background
    # so the first /api/news call returns something meaningful.
    start_scheduler()
    asyncio.create_task(_boot_tasks())
    try:
        yield
    finally:
        log.info("--- News-Engine shutting down ---")
        stop_scheduler()


async def _boot_tasks():
    try:
        await trigger_initial_refresh()
        await broadcaster.broadcast("boot", {"status": "ready"})
    except Exception:  # noqa: BLE001
        log.exception("boot tasks failed")


app = FastAPI(
    title="News-Driven Allocation Engine",
    version="0.1.0",
    description=(
        "Real-time news → signal → portfolio allocation research platform. "
        "Includes a vectorized backtester and a live paper-trading house model."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news_router)
app.include_router(backtest_router)
app.include_router(portfolio_router)
app.include_router(market_router)
app.include_router(forecast_router)


@app.get("/api/health")
async def health():
    s = get_settings()
    return {
        "status": "ok",
        "providers": {
            "marketaux": bool(s.marketaux_api_key),
            "newsapi_ai": bool(s.newsapi_ai_key),
            "eodhd": bool(s.eodhd_api_key),
        },
        "benchmark": s.benchmark_ticker,
        "universe": s.house_universe,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep the socket alive; we don't expect client→server messages.
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text('{"event":"pong","data":{}}')
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(websocket)


# --- Static frontend ----------------------------------------------------------
_FRONTEND = PROJECT_ROOT / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(_FRONTEND / "index.html")

    @app.get("/favicon.ico")
    async def favicon():
        f = _FRONTEND / "favicon.svg"
        if f.exists():
            return FileResponse(f)
        return FileResponse(_FRONTEND / "index.html")  # harmless fallback

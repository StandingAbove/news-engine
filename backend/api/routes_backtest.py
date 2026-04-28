"""Backtest endpoints."""
from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from ..backtest.service import run_configured_backtest
from ..core.db import BacktestRun, get_session_factory
from ..core.schemas import BacktestConfig, BacktestResult

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestResult)
async def run_backtest(cfg: BacktestConfig):
    try:
        return await run_configured_backtest(cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Backtest failed: {e}") from e


@router.get("/runs")
async def list_runs(limit: int = 20):
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit)
        rows = list((await session.execute(stmt)).scalars())
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "config": json.loads(r.config_json),
            "metrics": json.loads(r.metrics_json),
        })
    return out


@router.get("/runs/{run_id}")
async def get_run(run_id: int):
    factory = get_session_factory()
    async with factory() as session:
        r = await session.get(BacktestRun, run_id)
        if r is None:
            raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat(),
        "config": json.loads(r.config_json),
        "metrics": json.loads(r.metrics_json),
        "equity_curve": json.loads(r.equity_curve_json),
    }

"""WebSocket broadcaster for live dashboards."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set

from fastapi import WebSocket

from ..core.logging import get_logger

log = get_logger(__name__)


class Broadcaster:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("WS client connected (total=%d)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("WS client disconnected (total=%d)", len(self._clients))

    async def broadcast(self, event: str, data: Dict[str, Any]) -> None:
        payload = json.dumps({"event": event, "data": data}, default=str)
        dead = []
        async with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                await c.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(c)
        if dead:
            async with self._lock:
                for d in dead:
                    self._clients.discard(d)


broadcaster = Broadcaster()

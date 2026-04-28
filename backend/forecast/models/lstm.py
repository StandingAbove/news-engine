"""Small LSTM baseline (PyTorch).

Same recipe as the MLP — `lookback` input points, predict next, forecast
recursively — but with a single-layer LSTM encoder. Kept intentionally
small so it trains in seconds on CPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import Forecast, as_array


class LSTMForecaster:
    name = "lstm"

    def __init__(
        self,
        lookback: int = 32,
        hidden: int = 32,
        epochs: int = 200,
        lr: float = 5e-3,
        seed: int = 1234,
        device: Optional[str] = None,
    ) -> None:
        self.lookback = int(lookback)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device = device
        self._model = None
        self._mu: float = 0.0
        self._sd: float = 1.0
        self._last_window: np.ndarray | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def _build(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)

        class _Net(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: (batch, seq, 1)
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return _Net(self.hidden)

    def fit(self, history: np.ndarray) -> None:
        if not self.is_available():
            raise RuntimeError("torch not installed")
        import torch
        import torch.nn as nn

        h = as_array(history)
        if len(h) < self.lookback + 2:
            raise ValueError(
                f"need at least {self.lookback + 2} points, got {len(h)}"
            )

        self._mu = float(np.mean(h))
        self._sd = float(np.std(h) + 1e-8)
        z = (h - self._mu) / self._sd

        X = np.stack([z[i : i + self.lookback] for i in range(len(z) - self.lookback)])
        y = z[self.lookback :].reshape(-1, 1)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        Xt = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(-1)
        yt = torch.tensor(y, dtype=torch.float32, device=device)

        model = self._build().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()

        model.eval()
        self._model = model
        self._last_window = z[-self.lookback :].copy()

    def forecast(self, horizon: int) -> Forecast:
        if self._model is None or self._last_window is None:
            raise RuntimeError("fit() must be called first")
        import torch

        window = self._last_window.copy()
        preds_z: list[float] = []
        device = next(self._model.parameters()).device
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(window, dtype=torch.float32, device=device).view(1, -1, 1)
                yhat = float(self._model(x).item())
                preds_z.append(yhat)
                window = np.concatenate([window[1:], [yhat]])
        preds = np.array(preds_z, dtype=np.float64) * self._sd + self._mu
        return Forecast(point=preds)

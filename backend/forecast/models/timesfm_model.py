"""Google TimesFM foundation-model wrapper.

TimesFM is a decoder-only time-series foundation model released by Google
Research. It's the model Dr. Singh specifically named on Feb-6 ("get timesFM
and a few other forecasting models on your own"), so it gets a dedicated
wrapper here with the same `Forecaster` interface as every other model.

Deps
----
* `timesfm>=1.2` on PyPI
* `torch>=2.1` (we use the pytorch backend — jax is heavier and slower on CPU)
* First call downloads the checkpoint from HuggingFace; on CPU the 200M
  variant is ~800MB and loads in ~30s. The 500M v2 variant is ~2GB.

The wrapper is written defensively because the `timesfm` package has shipped
a few slightly different call signatures. We probe for the right one at
`fit()` time rather than failing at import.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import Forecast, as_array


# Sensible defaults. Overrideable via constructor kwargs.
_DEFAULT_CONTEXT = 512
_DEFAULT_REPO_V1 = "google/timesfm-1.0-200m-pytorch"
_DEFAULT_REPO_V2 = "google/timesfm-2.0-500m-pytorch"
# v2.5 ships under distinct HF paths; used only when Strategy C (2.5 classes)
# is the one that matches.
_REPO_V25_200M = "google/timesfm-2.5-200m-pytorch"
_REPO_V25_500M = "google/timesfm-2.5-500m-pytorch"


class TimesFMForecaster:
    name = "timesfm"

    def __init__(
        self,
        *,
        repo_id: Optional[str] = None,
        context_len: int = _DEFAULT_CONTEXT,
        backend: str = "cpu",
        per_core_batch_size: int = 1,
        num_layers: Optional[int] = None,
        use_v2: bool = False,
    ) -> None:
        self.repo_id = repo_id or (_DEFAULT_REPO_V2 if use_v2 else _DEFAULT_REPO_V1)
        self.context_len = int(context_len)
        self.backend = backend
        self.per_core_batch_size = int(per_core_batch_size)
        self.num_layers = num_layers
        self.use_v2 = bool(use_v2)
        self._history: np.ndarray | None = None
        self._loaded_model = None
        self._strategy: str | None = None
        # Cache on the class so repeat fits in the same process don't re-download.
        # Keyed by (repo_id, backend, context_len).
        self._cache_key = (self.repo_id, self.backend, self.context_len)

    # ---- availability ---------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        try:
            import timesfm  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    # ---- model loading --------------------------------------------------

    _model_cache: dict = {}

    def _load(self):
        import timesfm

        if self._cache_key in TimesFMForecaster._model_cache:
            return TimesFMForecaster._model_cache[self._cache_key]

        # The `timesfm` package API has changed shape three times across the
        # 1.x → 2.x → 2.5.x line. Rather than guess, we probe known entry
        # points in order and stop at the first one that successfully loads.
        errors: list[tuple[str, str]] = []

        # Strategy A: the 1.2 → 2.0 stable interface that shipped as
        # `timesfm.TimesFm(hparams=..., checkpoint=...)`.
        model = self._try_hparams_api(timesfm, errors)
        if model is not None:
            TimesFMForecaster._model_cache[self._cache_key] = (model, "hparams")
            return TimesFMForecaster._model_cache[self._cache_key]

        # Strategy B: the pre-Hparams positional constructor.
        model = self._try_positional_api(timesfm, errors)
        if model is not None:
            TimesFMForecaster._model_cache[self._cache_key] = (model, "positional")
            return TimesFMForecaster._model_cache[self._cache_key]

        # Strategy C: the 2.5.x main-branch interface where each variant
        # is its own class (e.g. `TimesFm_2p5_200M_torch`) with a
        # `load_checkpoint()` / `compile(ForecastConfig(...))` lifecycle.
        model = self._try_v25_api(timesfm, errors)
        if model is not None:
            TimesFMForecaster._model_cache[self._cache_key] = (model, "v25")
            return TimesFMForecaster._model_cache[self._cache_key]

        # Nothing worked. Emit what the package DID expose so the wrapper
        # can be pinned in the next round without another blind guess.
        exposed = sorted(n for n in dir(timesfm) if not n.startswith("_"))
        msg_parts = ["could not initialize timesfm via any known API."]
        for label, err in errors:
            msg_parts.append(f"  [{label}] {err}")
        msg_parts.append(
            "  installed timesfm exposes: " + ", ".join(exposed[:40])
            + (" …" if len(exposed) > 40 else "")
        )
        raise RuntimeError("\n".join(msg_parts))

    # ---- load strategies ------------------------------------------------

    def _try_hparams_api(self, timesfm, errors):
        """timesfm 1.2 → 2.0: TimesFm(hparams=..., checkpoint=...)."""
        if not hasattr(timesfm, "TimesFm") or not hasattr(timesfm, "TimesFmHparams"):
            errors.append(("hparams", "TimesFm / TimesFmHparams not present"))
            return None
        try:
            hparams_kwargs = dict(
                backend=self.backend,
                per_core_batch_size=self.per_core_batch_size,
                horizon_len=128,
                context_len=self.context_len,
            )
            if self.num_layers is not None:
                hparams_kwargs["num_layers"] = self.num_layers
            if self.use_v2:
                hparams_kwargs.setdefault("num_layers", 50)
                hparams_kwargs["use_positional_embedding"] = False
            hparams = timesfm.TimesFmHparams(**hparams_kwargs)
            ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
            return timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
        except Exception as e:  # noqa: BLE001
            errors.append(("hparams", f"{type(e).__name__}: {e}"))
            return None

    def _try_positional_api(self, timesfm, errors):
        """Older positional-init shape."""
        if not hasattr(timesfm, "TimesFm"):
            errors.append(("positional", "TimesFm not present"))
            return None
        try:
            model = timesfm.TimesFm(
                context_len=self.context_len,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=self.num_layers or 20,
                model_dims=1280,
                backend=self.backend,
            )
            model.load_from_checkpoint(repo_id=self.repo_id)
            return model
        except Exception as e:  # noqa: BLE001
            errors.append(("positional", f"{type(e).__name__}: {e}"))
            return None

    def _try_v25_api(self, timesfm, errors):
        """2.5 main-branch: per-variant class + load_checkpoint + compile.

        Classes are named like `TimesFM_2p5_200M_torch` (200M params, torch
        backend). The load_checkpoint signature varies by point-release
        (some builds take `path=`, others `checkpoint_path=`, others take a
        positional, and the newest just downloads from a class-bundled
        default with no args), so we introspect rather than guess.
        """
        # Cover the naming-style permutations we've seen: TimesFM vs TimesFm,
        # 2p5 vs 2_0, 200M vs 500M.
        candidates = []
        if self.use_v2:
            candidates += [
                "TimesFM_2p5_500M_torch",
                "TimesFm_2p5_500M_torch",
                "TimesFm_2_0_500M_torch",
            ]
        candidates += [
            "TimesFM_2p5_200M_torch",
            "TimesFm_2p5_200M_torch",
            "TimesFm_2_0_200M_torch",
            "TimesFM",
        ]
        cls = None
        picked_name = None
        for name in candidates:
            if hasattr(timesfm, name):
                cls = getattr(timesfm, name)
                picked_name = name
                break
        if cls is None:
            errors.append(
                ("v25", f"no 2.5-style class found (tried {', '.join(candidates)})")
            )
            return None

        # Pick the matching HF repo for the class we actually resolved.
        if "500M" in picked_name:
            hf_repo = _REPO_V25_500M
        elif "2p5" in picked_name:
            hf_repo = _REPO_V25_200M
        else:
            hf_repo = self.repo_id

        try:
            # Some builds expose a `from_pretrained` classmethod that does
            # construction + load in one shot. Prefer it when present.
            model = None
            if hasattr(cls, "from_pretrained"):
                try:
                    model = cls.from_pretrained(hf_repo)
                except Exception:  # noqa: BLE001
                    model = None

            if model is None:
                model = cls()
                self._load_checkpoint_dispatch(model, hf_repo)

            # 2.5 requires an explicit compile() before forecast().
            if hasattr(timesfm, "ForecastConfig") and hasattr(model, "compile"):
                self._compile_dispatch(timesfm, model)
            return model
        except Exception as e:  # noqa: BLE001
            errors.append(("v25", f"{picked_name}: {type(e).__name__}: {e}"))
            return None

    @staticmethod
    def _load_checkpoint_dispatch(model, hf_repo: str) -> None:
        """Call model.load_checkpoint with whatever kwarg it expects.

        We inspect the signature and bind the HF repo id to the first param
        name that matches a known alias. If load_checkpoint takes no
        bindable params, fall back to calling it with no args.
        """
        import inspect

        load = getattr(model, "load_checkpoint", None)
        if load is None:
            return

        try:
            sig = inspect.signature(load)
        except (TypeError, ValueError):
            # Unintrospectable — try the most common shapes in order.
            for attempt in ((hf_repo,), {"path": hf_repo}, {"checkpoint_path": hf_repo}, {}):
                try:
                    if isinstance(attempt, dict):
                        load(**attempt)
                    else:
                        load(*attempt)
                    return
                except TypeError:
                    continue
            load()  # last-ditch; let it raise if it must
            return

        # Param names load_checkpoint is known to use across versions.
        aliases = (
            "path",
            "checkpoint_path",
            "ckpt_path",
            "hf_repo_id",
            "hf_repo",
            "huggingface_repo_id",
            "repo_id",
            "repo",
            "model_id",
        )
        params = sig.parameters
        chosen = next((a for a in aliases if a in params), None)

        if chosen is not None:
            load(**{chosen: hf_repo})
            return

        # No recognized keyword — check for a positional path-like param.
        positional = [
            p for p in params.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if positional and positional[0].default is inspect.Parameter.empty:
            load(hf_repo)
            return

        # Nothing to bind — class presumably bundles a default.
        load()

    def _compile_dispatch(self, timesfm, model) -> None:
        """Build a ForecastConfig compatible with this timesfm release."""
        import inspect

        try:
            sig = inspect.signature(timesfm.ForecastConfig)
            params = set(sig.parameters)
        except (TypeError, ValueError):
            params = set()

        desired = {
            "max_context": self.context_len,
            "max_horizon": 128,
            "per_core_batch_size": self.per_core_batch_size,
        }
        cfg_kwargs = {k: v for k, v in desired.items() if not params or k in params}
        try:
            cfg = timesfm.ForecastConfig(**cfg_kwargs)
        except TypeError:
            # Fall back to a minimal construction.
            cfg = timesfm.ForecastConfig(max_context=self.context_len, max_horizon=128)
        model.compile(cfg)

    # ---- Forecaster interface -------------------------------------------

    def fit(self, history: np.ndarray) -> None:
        if not self.is_available():
            raise RuntimeError("timesfm / torch not installed")
        h = as_array(history)
        if h.size < 8:
            raise ValueError("timesfm needs at least 8 points of history")
        # Foundation models don't "fit" — the checkpoint is frozen. We just
        # stash the context and load the model lazily.
        self._history = h[-self.context_len :].copy()
        self._loaded_model, self._strategy = self._load()

    def forecast(self, horizon: int) -> Forecast:
        if self._loaded_model is None or self._history is None:
            raise RuntimeError("fit() must be called first")
        series = self._history.astype(np.float32).tolist()
        model = self._loaded_model

        point: np.ndarray | None = None
        quantiles = None

        if self._strategy == "v25":
            # 2.5 API: forecast(horizon=N, inputs=[...]) → (point, quantiles)
            # with point.shape = (batch, horizon). No freq kwarg.
            try:
                out = model.forecast(horizon=horizon, inputs=[series])
            except TypeError:
                out = model.forecast(inputs=[series], horizon=horizon)
            if isinstance(out, tuple):
                point_arr = out[0]
                quantiles = out[1] if len(out) > 1 else None
            else:
                point_arr = out
            point = np.asarray(point_arr[0], dtype=np.float64)[:horizon]
        else:
            # 1.x / 2.0 API: forecast(inputs=[...], freq=[0]).
            # freq codes: 0 = high-frequency (daily), 1 = monthly, 2 = yearly.
            try:
                point_arr, quantiles = model.forecast(inputs=[series], freq=[0])
                point = np.asarray(point_arr[0], dtype=np.float64)[:horizon]
            except TypeError:
                out = model.forecast(inputs=[series])
                if isinstance(out, tuple):
                    out = out[0]
                point = np.asarray(out[0], dtype=np.float64)[:horizon]
                quantiles = None

        lower = upper = None
        if quantiles is not None:
            try:
                q = np.asarray(quantiles[0], dtype=np.float64)
                # timesfm returns [0.1, 0.2, ..., 0.9] along the last axis.
                if q.ndim == 2 and q.shape[-1] >= 9:
                    lower = q[:horizon, 0]
                    upper = q[:horizon, 8]
            except Exception:  # noqa: BLE001
                lower = upper = None

        # TimesFM sometimes under-fills when context < patch_len. Pad out.
        if point.size < horizon:
            padding = np.full(horizon - point.size, point[-1] if point.size else 0.0)
            point = np.concatenate([point, padding])
        return Forecast(point=point, lower=lower, upper=upper)

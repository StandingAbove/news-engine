"""Application configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Values are loaded from environment variables and a .env file at the project
    root. Default values mirror `.env.example`.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- provider keys ---
    marketaux_api_key: str = Field(default="", alias="MARKETAUX_API_KEY")
    newsapi_ai_key: str = Field(default="", alias="NEWSAPI_AI_KEY")
    eodhd_api_key: str = Field(default="", alias="EODHD_API_KEY")

    # --- server ---
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    # --- storage ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/news_engine.db", alias="DATABASE_URL"
    )

    # --- house model ---
    house_universe_csv: str = Field(
        default="SPY,QQQ,DIA,IWM,EFA,EEM,TLT,GLD,USO,XLE,XLK,XLF,XLV,XLY,XLI,XLP,XLU,XLB,XLRE",
        alias="HOUSE_UNIVERSE",
    )
    paper_starting_cash: float = Field(default=1_000_000.0, alias="PAPER_STARTING_CASH")
    benchmark_ticker: str = Field(default="SPY", alias="BENCHMARK_TICKER")

    # --- logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def house_universe(self) -> List[str]:
        return [t.strip().upper() for t in self.house_universe_csv.split(",") if t.strip()]

    @field_validator("app_port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError("APP_PORT must be in 1024..65535")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Ensure data dir exists even on first launch.
(PROJECT_ROOT / "data").mkdir(exist_ok=True, parents=True)

"""
QVEX v10.7 — Канонический контракт телеметрии Core State (Audit GAP-04).
Строго соответствует разделам 8 и 9 архитектурного аудита:
- Допускает None для финансовых и рыночных полей (никаких фиктивных default-значений).
- Содержит блок DataQuality для явной фиксации актуальности данных.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PositionState(BaseModel):
    coin: str = Field(..., description="Тикер актива")
    side: str = Field(..., description="LONG / SHORT")
    size: float = Field(..., description="Размер позиции")
    entry_px: float = Field(..., description="Цена входа")
    sl_px: Optional[float] = Field(default=None, description="Биржевой стоп-лосс")
    highest_px: Optional[float] = Field(default=None)
    lowest_px: Optional[float] = Field(default=None)
    trailing_active: bool = Field(default=False)
    breakeven_active: bool = Field(default=False)
    ml_prob: Optional[float] = Field(default=None)


class ModelStatus(BaseModel):
    loaded: bool = Field(default=False)
    model_path: Optional[str] = None
    features_count: int = Field(default=0)


class DataQuality(BaseModel):
    fresh: bool = Field(..., description="Флаг актуальности ончейн-данных")
    reason: Optional[str] = Field(default=None, description="Причина неполноты/устаревания данных")


class CoreState(BaseModel):
    schema_version: int = Field(default=1, description="Версия схемы контракта")
    timestamp: float = Field(..., description="UTC timestamp генерации снимка")

    system_status: str = Field(default="ACTIVE", description="ACTIVE / PAUSED / ERROR")
    trading_enabled: bool = Field(..., description="Разрешение на открытие новых позиций")
    network: str = Field(..., description="MAINNET / TESTNET")

    account_address: Optional[str] = Field(default=None, description="L1 адрес кошелька")

    equity: Optional[float] = Field(default=None, description="Реальный капитал счета (или None)")
    free_margin: Optional[float] = Field(default=None, description="Свободная маржа (или None)")
    unrealized_pnl: Optional[float] = Field(default=None, description="Совокупный U-PnL (или None)")

    btc_price: Optional[float] = Field(default=None, description="Цена бенчмарка BTC")
    market_regime: Optional[str] = Field(default=None, description="BULL / BEAR / CHOP")

    active_slots: int = Field(default=0, description="Текущее количество открытых позиций")
    max_slots: int = Field(default=2, description="Лимит параллельных слотов")

    positions: List[PositionState] = Field(default_factory=list, description="Список активных позиций")
    model_status: Optional[ModelStatus] = Field(default=None, description="Статус ML-модели")
    data_quality: DataQuality = Field(..., description="Статус качества данных")


# Канонический путь к снимку телеметрии
CANONICAL_TELEMETRY_PATH = "data/control_plane_state.json"


# Реэкспорт модели управления для совместимости
from core.ipc.control import TradingControlState

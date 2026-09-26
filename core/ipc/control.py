import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from core.utils.helpers import atomic_write_json, safe_read_json

logger = logging.getLogger(__name__)


class TradingControlState(BaseModel):
    """Каноническая модель состояния управления торговлей из Control Plane."""
    trading_enabled: bool = Field(default=True, description="Флаг активности торговли")
    panic_requested: bool = Field(default=False, description="Запрос экстренной ликвидации позиций")
    last_command_by: str = Field(default="System", description="Инициатор последней команды")
    message: str = Field(default="Штатный режим", description="Пояснение к текущему статусу")
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Временная метка обновления"
    )


class ControlStateManager:
    """Управление состоянием торговли и перехват операторских сигналов."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._init_defaults()

    def _init_defaults(self) -> None:
        state = TradingControlState(
            trading_enabled=True,
            panic_requested=False,
            last_command_by="System",
            message="Инициализация по умолчанию",
            updated_at=datetime.now(timezone.utc).isoformat()
        )
        self.set_state(state)

    def get_state(self) -> TradingControlState:
        data = safe_read_json(self.path, fallback=None)
        if data is None:
            return TradingControlState()
        try:
            return TradingControlState.model_validate(data)
        except Exception:
            return TradingControlState()

    def set_state(self, state: TradingControlState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.loads(state.model_dump_json())
        atomic_write_json(self.path, payload)

    def pause_trading(self, admin_tag: str = "Operator") -> TradingControlState:
        state = self.get_state()
        state.trading_enabled = False
        state.last_command_by = admin_tag
        state.message = "Торговля приостановлена оператором (новые сделки заблокированы)"
        self.set_state(state)
        return state

    def resume_trading(self, admin_tag: str = "Operator") -> TradingControlState:
        state = self.get_state()
        state.trading_enabled = True
        state.last_command_by = admin_tag
        state.message = "Торговля активна (штатный режим)"
        self.set_state(state)
        return state

    def request_panic(self, admin_tag: str = "Operator") -> TradingControlState:
        state = self.get_state()
        state.panic_requested = True
        state.trading_enabled = False
        state.last_command_by = admin_tag
        state.message = "ИНИЦИИРОВАНА ПАНИКА: Закрытие всех позиций и остановка"
        self.set_state(state)
        return state

    def clear_panic(self) -> TradingControlState:
        state = self.get_state()
        state.panic_requested = False
        self.set_state(state)
        return state

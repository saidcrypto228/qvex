import json
import logging
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar
from pydantic import BaseModel
from core.utils.helpers import atomic_write_json, safe_read_json

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class PosixAtomicStateManager(Generic[T]):
    """Атомарный менеджер канонического состояния телеметрии."""

    def __init__(self, path: Path | str, schema: Type[T]):
        self.path = Path(path).resolve()
        self.schema = schema
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_atomic_state(self, state: T) -> None:
        try:
            payload = json.loads(state.model_dump_json()) if hasattr(state, "model_dump_json") else state
            atomic_write_json(self.path, payload)
        except Exception as err:
            logger.error(f"[IPC] Ошибка атомарной записи в {self.path.name}: {err}")
            raise

    def read_atomic_state(self) -> Optional[T]:
        data = safe_read_json(self.path, fallback=None)
        if data is None:
            return None
        try:
            return self.schema.model_validate(data)
        except Exception as err:
            logger.error(f"[IPC] Ошибка валидации {self.path.name}: {err}")
            return None

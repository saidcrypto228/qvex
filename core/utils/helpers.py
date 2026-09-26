import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

def atomic_write_json(
    file_path: Path | str,
    data: Any,
    max_retries: int = 25,
    base_delay: float = 0.005
) -> None:
    path = Path(file_path).resolve()
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        dir=directory,
        delete=False,
        encoding="utf-8",
        prefix=f".tmp_{path.stem}_",
        suffix=".tmp"
    )
    temp_name = temp_file.name
    try:
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_file.close()

        for attempt in range(max_retries):
            try:
                os.replace(temp_name, path)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (1.3 ** attempt))
    except Exception:
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except OSError:
                pass
        raise

def safe_read_json(
    file_path: Path | str,
    fallback: Any = None,
    max_retries: int = 20,
    base_delay: float = 0.005
) -> Any:
    path = Path(file_path).resolve()
    if not path.exists():
        return fallback
    for attempt in range(max_retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError, OSError):
            if attempt == max_retries - 1:
                return fallback
            time.sleep(base_delay * (1.3 ** attempt))
    return fallback

#!/usr/bin/env python3
"""
Конфигурация квантового торгового комплекса QVEX v12.1 Canonical.
- Архитектура: Strict DPL 4H + FR Shield + 2-Slot FIFO + Chandelier 1.75 ATR.
"""

import os
import sys
from pathlib import Path

# Сеть и учетные данные
IS_TESTNET = os.getenv("HYPERLIQUID_TESTNET", "true").strip().lower() in ("1", "true", "yes", "on")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS", "0x0000000000000000000000000000000000000000").strip()
SECRET_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "").strip()

# Windows console UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "bot_state.json"

# Вселенная ликвидных альткоинов
TARGET_COINS = [
    "SOL", "AVAX", "SUI", "APT", "DOGE",
    "NEAR", "ARB", "OP", "TIA", "INJ", "RENDER"
]

# Временные интервалы
CANDLE_TIMEFRAME = "1h"
CHECK_INTERVAL_SEC = 60
RECONCILE_INTERVAL_SEC = 15
DEADMAN_TIMEOUT_MIN = 15

# Архитектурные лимиты слотов и плеча (v12.1 Canonical)
PORTFOLIO_HARD_LEVERAGE_CAP = 2.50
MAX_PORTFOLIO_LEVERAGE = 2.50
BASE_CONCURRENT_POSITIONS = 2
EXPANDED_CONCURRENT_POSITIONS = 2  # Строгий лимит 2 слота (FIFO)
MAX_OPEN_POSITIONS = 2

# Институциональный риск-менеджмент
BASE_RISK_PER_TRADE = 0.0100       # Строгий Flat 1.0% на сделку
PORTFOLIO_STOP_RISK_CAP = 0.0800   # Максимальный совокупный риск портфеля <= 8.0%
MAX_SINGLE_POSITION_LEVERAGE = 1.10
MIN_NOTIONAL_USD = 10.0
ENTRY_SLIPPAGE = 0.005

# Макро-шлюз BTC (Strict DPL v12.1)
BTC_SLOPE_THRESHOLD = 0.08         # Порог наклона EMA50 (> +0.08% для отсечения флэта)
BTC_TREND_FILTER_MA_PERIOD = 200

# Параметры сопровождения позиций
BREAKEVEN_TRIGGER_ATR = 1.10       # Перенос в безубыток при +1.1 ATR
CHANDELIER_LOOKBACK_PERIODS = 18
CHANDELIER_ATR_MULTIPLIER = 1.75   # Трейлинг Chandelier 1.75 ATR

# Микроструктура и лимиты L1
MAX_PRICE_SIGNIFICANT_FIGURES = 5
MAX_PERP_DECIMALS = 6
STOP_BUFFER_LIMIT_RATIO = 0.15
ORDERFLOW_STALE_TIMEOUT_SEC = 15.0

# Funding Crowding Shield (v12.1)
MAX_ADVERSE_FUNDING_RATE = 0.00020  # +0.020% в час (~175% APR)
MAX_FUNDING_ZSCORE = 1.8            # Z-score перегрева розничного плеча

# Исполнение ордеров
ENABLE_MAKER_FIRST = True
MAKER_TIMEOUT_SEC = 2.0
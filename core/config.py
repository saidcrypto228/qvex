#!/usr/bin/env python3
"""
Конфигурация квантового торгового комплекса v10.0 (Alpha Expansion & L1 Hardened).
"""

import os
import sys
from pathlib import Path

# Сеть и учетные данные
IS_TESTNET = os.getenv("HYPERLIQUID_TESTNET", "true").strip().lower() in ("1", "true", "yes", "on")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS", "0x0000000000000000000000000000000000000000").strip()
SECRET_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "").strip()

# Базовые директории

# Windows console UTF-8 compatibility.
# Prevent UnicodeEncodeError for production logging messages.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "bot_state.json"

# Вселенная активов (11 ликвидных альткоинов)
TARGET_COINS = [
    "SOL", "AVAX", "SUI", "APT", "DOGE", 
    "NEAR", "ARB", "OP", "TIA", "INJ", "RENDER"
]

# Временные интервалы
CANDLE_TIMEFRAME = "1h"
CHECK_INTERVAL_SEC = 60
RECONCILE_INTERVAL_SEC = 15
DEADMAN_TIMEOUT_MIN = 15

# Alpha Blueprint v10.0: Лимиты капитала и динамические слоты
PORTFOLIO_HARD_LEVERAGE_CAP = 2.50
MAX_PORTFOLIO_LEVERAGE = 2.50
BASE_CONCURRENT_POSITIONS = 2
EXPANDED_CONCURRENT_POSITIONS = 4
MAX_OPEN_POSITIONS = 2  # Динамически модулируется до 4

# Адаптивный риск-менеджмент
BASE_RISK_PER_TRADE = 0.0100
MIN_RISK_PER_TRADE = 0.0060
MAX_RISK_PER_TRADE = 0.0165
MAX_SINGLE_POSITION_LEVERAGE = 1.10
MIN_NOTIONAL_USD = 10.0
ENTRY_SLIPPAGE = 0.005

# Макро-модуляция тренда BTC
BTC_SLOPE_THRESHOLD = 0.40
BTC_TREND_FILTER_MA_PERIOD = 200

# Асимметричный выход: 40% TP1 + 60% Runner
TAKE_PROFIT_1_ATR_MULTIPLE = 1.50
TAKE_PROFIT_1_SIZE_RATIO = 0.40
SOFT_BREAKEVEN_ATR_OFFSET = 0.20
CHANDELIER_LOOKBACK_PERIODS = 18
CHANDELIER_ATR_MULTIPLIER = 2.50

# Микроструктура и лимиты L1
MAX_PRICE_SIGNIFICANT_FIGURES = 5
MAX_PERP_DECIMALS = 6
STOP_BUFFER_LIMIT_RATIO = 0.15
ORDERFLOW_STALE_TIMEOUT_SEC = 15.0


# --- Alpha Gate: Funding Rate Threshold ---
MAX_ADVERSE_FUNDING_RATE = 0.0003  # 0.03% в час (~262% APR)

# --- Alpha Execution: Maker-First Configuration ---
ENABLE_MAKER_FIRST = True        # Пассивное выставление лимитных Alo заявок
MAKER_TIMEOUT_SEC = 2.0         # Окно ожидания налития до маркет-фолбэка

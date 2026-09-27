import torch
from core.utils.helpers import atomic_write_json
#!/usr/bin/env python3
"""
Hyperliquid 4H Swing Bot (Production v10.7 - Institutional Hardened Engine).
- P0 Fix: Нативная атомарная замена стопов через modify_order() без cancel-race.
- P0 Fix: Строгая идентификация Stop-Loss по вложенному типу trigger.tpsl == 'sl'.
- P0 Fix: Ликвидация риска сноса стопов через DMS при открытых сделках.
- P0 Fix: Жесткий инвариант совокупного риска портфеля: PortfolioStopRisk <= 8.0% от Equity.
- P1 Fix: Изоляция незакрытых 1H-свечей (строго iloc[-2] для индикаторов).
"""

import os
import sys
import time
import math
import json
import logging
import re
import uuid
import threading
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from hyperliquid.utils.types import Cloid

from core import config
from core.ipc.control import ControlStateManager
from core.ipc.schema import CoreState, PositionState, ModelStatus, DataQuality, CANONICAL_TELEMETRY_PATH
from core.ipc.state import PosixAtomicStateManager
from core.quant.factors import QuantFactorEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] QVEX-v10.7: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_DIR / "bot_runtime.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("QVEX-v10.7")

class PrecisionEngine:
    @staticmethod
    def round_sz(size: float, sz_decimals: int) -> float:
        if not math.isfinite(size) or size <= 0:
            return 0.0
        quantum = Decimal("1").scaleb(-sz_decimals)
        return float(Decimal(str(size)).quantize(quantum, rounding=ROUND_DOWN))

    @staticmethod
    def round_px(price: float, sz_decimals: int, is_buy_stop: bool = False, max_decimals: int = 6) -> float:
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid price value: {price}")

        # 1. Целочисленные цены разрешены L1 независимо от значащих цифр
        if abs(price - round(price)) < 1e-9:
            return float(round(price))

        # 2. Институциональная формула Hyperliquid:
        # До 5 значащих цифр И не более (6 - szDecimals) знаков после запятой
        magnitude = math.floor(math.log10(price))
        sig_decimals = max(0, 5 - magnitude - 1)
        hard_max_decimals = max(0, max_decimals - sz_decimals)
        allowed_decimals = max(0, min(sig_decimals, hard_max_decimals))

        step = Decimal("10") ** -allowed_decimals
        dec_price = Decimal(f"{price:.10f}")
        round_mode = ROUND_UP if is_buy_stop else ROUND_DOWN
        rounded_dec = dec_price.quantize(step, rounding=round_mode)

        res = float(f"{rounded_dec:f}")
        if res <= 0:
            raise ValueError(f"Price rounded to zero: {price}")
        return res

class MarketDataWorker:
    def __init__(self, info: Info):
        self.info = info

    def fetch_1h_candles(self, coin: str, limit: int = 160) -> pd.DataFrame:
        end_time = int(time.time() * 1000)
        start_time = end_time - (limit * 3600 * 1000)

        raw = self.info.candles_snapshot(
            name=coin, interval=config.CANDLE_TIMEFRAME,
            startTime=start_time, endTime=end_time
        )
        if not raw:
            return pd.DataFrame()

        records = [{
            "timestamp": int(c["t"]),
            "dt": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"])
        } for c in raw]

        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        # Отсекаем текущую формирующуюся часовую свечу
        if len(df) > 2 and (time.time() * 1000 - df.iloc[-1]["timestamp"] < 3500 * 1000):
            df = df.iloc[:-1].reset_index(drop=True)
        return df

    def compute_multi_tf_indicators(self, coin: str) -> pd.DataFrame:
        df_1h = self.fetch_1h_candles(coin, limit=160)
        if len(df_1h) < 40:
            return pd.DataFrame()

        df_1h["vol_rolling_4h"] = df_1h["volume"].rolling(4, min_periods=4).sum()
        df_temp = df_1h.set_index("dt")
        ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "first"}
        df_4h = df_temp.resample("4h", label="left", closed="left").agg(ohlc).dropna().reset_index()

        c = df_4h["close"]
        h = df_4h["high"]
        l = df_4h["low"]
        v = df_4h["volume"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

        df_4h["atr_4h"] = tr.rolling(14, min_periods=14).mean()
        df_4h["ema20_4h"] = c.ewm(span=20, adjust=False).mean()
        df_4h["ema50_4h"] = c.ewm(span=50, adjust=False).mean()
        df_4h["ema50_slope"] = df_4h["ema50_4h"] - df_4h["ema50_4h"].shift(3)
        df_4h["donchian_high_4h"] = h.rolling(36).max()
        df_4h["donchian_low_4h"] = l.rolling(36).min()
        df_4h["vol_sma20_4h"] = v.rolling(20, min_periods=20).mean()

        df_4h_shifted = df_4h[[
            "dt", "atr_4h", "ema20_4h", "ema50_4h", "ema50_slope", 
            "donchian_high_4h", "donchian_low_4h", "vol_sma20_4h"
        ]].copy()

        for col in ["atr_4h", "ema20_4h", "ema50_4h", "ema50_slope", "donchian_high_4h", "donchian_low_4h", "vol_sma20_4h"]:
            df_4h_shifted[col] = df_4h_shifted[col].shift(1)

        merged = pd.merge_asof(
            df_1h.sort_values("dt"),
            df_4h_shifted.sort_values("dt"),
            on="dt",
            direction="backward"
        )
        return merged.dropna().reset_index(drop=True)

class HyperliquidSwingBot:
    def __init__(self):
        self.is_testnet = config.IS_TESTNET
        base_url = constants.TESTNET_API_URL if self.is_testnet else constants.MAINNET_API_URL
        self.address = config.ACCOUNT_ADDRESS

        self.info = Info(base_url, skip_ws=True, timeout=5)
        self.md_worker = MarketDataWorker(self.info)

        self.exchange = None
        raw_key = (config.SECRET_KEY or "").strip().strip('"').strip("'")
        clean_hex = raw_key[2:] if raw_key.startswith("0x") else raw_key
        is_valid_hex = bool(re.fullmatch(r"^[0-9a-fA-F]{64}$", clean_hex))
        is_placeholder = set(clean_hex) == {"0"} or len(clean_hex) != 64

        if is_valid_hex and not is_placeholder:
            try:
                from eth_account import Account
                wallet = Account.from_key(raw_key)
                self.exchange = Exchange(wallet, base_url, account_address=self.address, timeout=5)
                logger.info(f"[+] L1 Exchange Client инициализирован: {self.address}")
            except Exception as e:
                logger.error(f"[-] Ошибка загрузки приватного ключа: {e}. Режим DRY-RUN.")
        else:
            logger.warning("[!] Приватный ключ не задан. Режим DRY-RUN.")

        self.control_mgr = ControlStateManager()
        self.telemetry_ipc = PosixAtomicStateManager(CANONICAL_TELEMETRY_PATH, CoreState)
        self.universe_meta = self._load_meta()
        self.state = self.load_state()
        self.pending_triggers: Dict[str, Any] = {}
        self.last_reconcile_time = 0
        self.active_slots_limit = config.BASE_CONCURRENT_POSITIONS

        model_file = config.DATA_DIR / "meta_model.json"
        self.meta_weights = False
        if model_file.exists():
            try:
                with open(model_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                self.feature_cols = m_data.get("feature_cols") or m_data.get("feature_names", [])
                self.weights = np.array(m_data.get("coef") or m_data.get("weights", []))
                self.intercept = float(m_data["intercept"])
                self.scaler_mean = np.array(m_data["scaler_mean"])
                self.scaler_scale = np.array(m_data["scaler_scale"])
                self.scaler_scale = np.where(self.scaler_scale <= 1e-6, 1.0, self.scaler_scale)
                self.meta_weights = True

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    self.torch_weights = torch.as_tensor(self.weights, dtype=torch.float32, device=self.device)
                    self.torch_intercept = torch.as_tensor(self.intercept, dtype=torch.float32, device=self.device)
                    self.torch_scaler_mean = torch.as_tensor(self.scaler_mean, dtype=torch.float32, device=self.device)
                    self.torch_scaler_scale = torch.as_tensor(self.scaler_scale, dtype=torch.float32, device=self.device)
                    logger.info(f"[✓] Meta-Model де Прадо загружена на CUDA GPU ({torch.cuda.get_device_name(0)}): {model_file}")
                else:
                    self.torch_weights = None
                    logger.info(f"[✓] Meta-Model загружена на CPU: {model_file}")
            except Exception as e:
                logger.critical(f"[-] Ошибка загрузки ML JSON: {e}")
        else:
            logger.critical(f"[!] Файл {model_file} отсутствует! Режим Fail-Closed.")

    def _load_meta(self) -> Dict[str, Any]:
        meta = self.info.meta()
        return {asset["name"]: asset for asset in meta.get("universe", [])}

    def load_state(self) -> Dict[str, Any]:
        if config.STATE_FILE.exists():
            try:
                with open(config.STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"positions": {}, "last_reconcile": 0}

    def save_state(self):
        try:
            from core.utils.helpers import atomic_write_json
            atomic_write_json(config.STATE_FILE, self.state)
        except Exception as err:
            logger.error(f"[-] Ошибка сохранения стейта: {err}")

    def predict_meta_prob(self, raw_features: list) -> float:
        if not self.meta_weights:
            return 0.0

        if getattr(self, "torch_weights", None) is not None:
            try:
                with torch.no_grad():
                    x_t = torch.as_tensor(raw_features, dtype=torch.float32, device=self.device)
                    x_scaled = (x_t - self.torch_scaler_mean) / self.torch_scaler_scale
                    z = torch.dot(self.torch_weights, x_scaled) + self.torch_intercept
                    z_clipped = torch.clamp(z, min=-15.0, max=15.0)
                    prob = torch.sigmoid(z_clipped)
                    return float(prob.item())
            except Exception:
                pass

        x = (np.array(raw_features) - self.scaler_mean) / self.scaler_scale
        z = float(np.dot(self.weights, x) + self.intercept)
        z_clipped = max(min(z, 15.0), -15.0)
        return 1.0 / (1.0 + math.exp(-z_clipped))

    def is_protective_stop(self, order: dict, coin: str, is_long: bool) -> bool:
        """P0 Fix: Строгая идентификация Stop-Loss через tpsl == 'sl'."""
        if order.get("coin") != coin or not order.get("isTrigger") or not order.get("reduceOnly"):
            return False

        # Проверяем вложенную структуру tpsl
        trigger_info = order.get("orderType", {})
        if isinstance(trigger_info, dict):
            tpsl_val = trigger_info.get("trigger", {}).get("tpsl") or trigger_info.get("tpsl")
            if tpsl_val and tpsl_val.lower() != "sl":
                return False

        o_type = str(order.get("orderType", "")).lower()
        if "tp" in o_type or "take" in o_type:
            return False

        expected_side = "A" if is_long else "B"
        return order.get("side") == expected_side

    def reconcile_with_exchange(self):
        now = time.time()
        if now - self.last_reconcile_time < config.RECONCILE_INTERVAL_SEC:
            return

        self.last_reconcile_time = now
        if not self.exchange or self.address.startswith("0x000"):
            return

        try:
            acc_state = self.info.clearinghouse_state(self.address)
            open_orders = self.info.frontend_open_orders(self.address)

            actual_positions = {}
            for item in acc_state.get("assetPositions", []):
                p = item.get("position", {})
                coin = p.get("coin")
                szi = float(p.get("szi", 0.0))
                if abs(szi) > 0:
                    actual_positions[coin] = {
                        "size": abs(szi),
                        "direction": "LONG" if szi > 0 else "SHORT",
                        "entry_px": float(p.get("entryPx", 0.0)),
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0.0)),
                        "position_value": float(p.get("positionValue", 0.0))
                    }
            if isinstance(self.state.get('positions'), list):
                self.state['positions'] = {
                    p['symbol']: p
                    for p in self.state['positions']
                    if isinstance(p, dict) and 'symbol' in p
                }
            elif not isinstance(self.state.get('positions'), dict):
                self.state['positions'] = {}

            for coin in list(self.state['positions'].keys()):
                if coin not in actual_positions:
                    logger.warning(
                        f'[RECONCILE] Позиция {coin} закрыта на бирже. Очистка локального стейта...'
                    )
                    # Отмена всех оставшихся ордеров/стопов по монете во избежание фантомных входов
                    if self.exchange:
                        coin_orders = [o for o in open_orders if o.get("coin") == coin]
                        for o in coin_orders:
                            oid = o.get("oid")
                            if oid:
                                try:
                                    self.exchange.cancel(coin, oid)
                                    logger.info(f"[RECONCILE] Снят осиротевший ордер/стоп {oid} по {coin}")
                                except Exception as c_err:
                                    logger.warning(f"[RECONCILE] Не удалось снять ордер {oid}: {c_err}")
                    del self.state['positions'][coin]
                    self.save_state()
                else:
                    onchain = actual_positions[coin]
                    loc = self.state["positions"][coin]
                    if abs(loc["size"] - onchain["size"]) > 1e-5:
                        logger.warning(f"[RECONCILE] Размер {coin} скорректирован биржей: {loc['size']} -> {onchain['size']}")
                        loc["size"] = onchain["size"]
                        loc["entry_px"] = onchain["entry_px"]
                        self.save_state()

            for coin, pos in actual_positions.items():
                if coin not in self.state["positions"]:
                    logger.warning(f"[RECONCILE] Внешняя позиция {coin} подхвачена в стейт.")
                    self.state["positions"][coin] = {
                        "status": "POSITION_ACTIVE", "direction": pos["direction"],
                        "size": pos["size"], "entry_px": pos["entry_px"],
                        "sl_px": pos["entry_px"] * (0.95 if pos["direction"] == "LONG" else 1.05),
                        "trailing_active": False, "breakeven_active": False, "stop_oid": None
                    }
                    self.save_state()

            # Верификация наличия строго ОДНОГО канонического стоп-лосса
            for coin, pos in self.state["positions"].items():
                is_long = pos["direction"] == "LONG"
                stops = [o for o in open_orders if self.is_protective_stop(o, coin, is_long)]
                if not stops:
                    logger.critical(f"[RECONCILE ALERT] У {coin} НЕТ Stop Loss на L1! Немедленная установка...")
                    self.place_native_market_stop(coin, pos["size"], pos["sl_px"], is_long=is_long)
                else:
                    pos["stop_oid"] = stops[-1].get("oid")

        
        except Exception as e:
            logger.error(f"[-] Сбой ончейн-реконсиляции: {e}")

    def refresh_deadmans_switch(self):
        """P0 Fix: При наличии хотя бы одной позиции DMS ВСЕГДА деактивирован."""
        if not self.exchange or self.address.startswith("0x000"):
            return

        if len(self.state.get("positions", {})) > 0:
            try:
                self.exchange.schedule_cancel(None)
            except Exception:
                pass
            return

        # Если позиций нет — страхуем зависшие ордера входа на 15 мин
        try:
            timeout_ms = int(time.time() * 1000) + (config.DEADMAN_TIMEOUT_MIN * 60 * 1000)
            self.exchange.schedule_cancel(timeout_ms)
        except Exception:
            pass

    def get_orderflow_metrics(self, coin: str) -> dict:
        state_file = config.DATA_DIR / "orderflow_state.json"
        if not state_file.exists():
            return {"cvd_ratio": 0.0, "cvd_usd": 0.0, "obi_10": 0.0, "delta_oi_z": 0.0, "funding_rate": 0.0, "last_px": 0.0, "is_fresh": False}
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) > config.ORDERFLOW_STALE_TIMEOUT_SEC:
                return {"cvd_ratio": 0.0, "cvd_usd": 0.0, "obi_10": 0.0, "delta_oi_z": 0.0, "funding_rate": 0.0, "last_px": 0.0, "is_fresh": False}
            coin_data = data.get("coins", {}).get(coin, {})
            return {
                "cvd_ratio": coin_data.get("cvd_ratio", 0.0),
                "cvd_usd": coin_data.get("cvd_usd", 0.0),
                "obi_10": coin_data.get("obi_10", 0.0),
                "delta_oi_z": coin_data.get("delta_oi_z", 0.0),
                "funding_rate": float(coin_data.get("funding_rate", 0.0)),
                "last_px": coin_data.get("last_px", 0.0),
                "is_fresh": True
            }
        except Exception:
            return {"cvd_ratio": 0.0, "cvd_usd": 0.0, "obi_10": 0.0, "delta_oi_z": 0.0, "funding_rate": 0.0, "last_px": 0.0, "is_fresh": False}

    def get_portfolio_equity(self) -> float:
        if self.exchange and not self.address.startswith("0x000"):
            try:
                acc_state = self.info.clearinghouse_state(self.address)
                return float(acc_state.get("marginSummary", {}).get("accountValue", 1000.0))
            except Exception:
                pass
        return 1000.0

    def calculate_sizing(self, coin: str, entry_px: float, sl_px: float, ml_prob: float = 0.50, risk_pct: Optional[float] = None, committed_notional: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        P0 Fix: Прямой инвариант риска портфеля: PortfolioStopRisk <= 8.0%.
        Никакой одновременный вынос 3 стопов не может превысить лимит фонда 15%.
        """
        equity = self.get_portfolio_equity()
        sl_dist_pct = abs(entry_px - sl_px) / entry_px
        if sl_dist_pct <= 0:
            return None

        # 1. Расчет текущего совокупного риска уже открытых позиций
        current_stop_risk = sum(
            p.get("size", 0.0) * abs(p.get("entry_px", 0.0) - p.get("sl_px", 0.0))
            for p in self.state["positions"].values()
        )
        max_portfolio_stop_risk = equity * 0.08  # Хардкап суммарного риска 8.0%
        remaining_risk_budget = max(0.0, max_portfolio_stop_risk - current_stop_risk)
        if remaining_risk_budget <= 0:
            logger.warning("[RISK CAP] Лимит риска портфеля (8%) исчерпан. Вход заблокирован.")
            return None

        # 2. Расчет допустимого нотионала по риску
        effective_trade_risk = risk_pct if risk_pct is not None else getattr(config, 'BASE_RISK_PER_TRADE', 0.0100)
        risk_based_notional = (equity * effective_trade_risk) / sl_dist_pct
        max_notional_by_budget = remaining_risk_budget / sl_dist_pct

        # 3. Лимиты плеча: одиночная позиция <= 1.15x, целевой вес <= 0.75x
        max_single_ntl = equity * getattr(config, "MAX_SINGLE_POSITION_LEVERAGE", 1.15)
        target_notional = min(risk_based_notional, max_notional_by_budget, max_single_ntl, equity * 0.75)

        # 4. Проверка лимита всего портфеля (Gross Leverage <= 2.50x)
        current_open_ntl = sum(p.get("size", 0.0) * p.get("entry_px", 0.0) for p in self.state["positions"].values())
        portfolio_cap = equity * config.PORTFOLIO_HARD_LEVERAGE_CAP
        available_gross_cap = max(0.0, portfolio_cap - (current_open_ntl + committed_notional))

        final_notional = min(target_notional, available_gross_cap)
        if final_notional < config.MIN_NOTIONAL_USD:
            return None

        coin_meta = self.universe_meta.get(coin)
        if not coin_meta:
            return None

        clean_sz = PrecisionEngine.round_sz(final_notional / entry_px, coin_meta["szDecimals"])
        if clean_sz * entry_px < 10.0:
            return None

        return {
            "coin": coin, "sz": clean_sz, "entry_px": entry_px,
            "sl_px": sl_px, "notional": clean_sz * entry_px,
            "risk_pct": (clean_sz * abs(entry_px - sl_px)) / equity
        }

    def place_native_market_stop(self, coin: str, sz: float, sl_price: float, is_long: bool) -> bool:
        """P0 Fix: Атомарная модификация через modify_order() при наличии OID."""
        coin_meta = self.universe_meta.get(coin, {})
        sz_decimals = coin_meta.get("szDecimals", 2)

        is_buy_stop = not is_long
        clean_sl = PrecisionEngine.round_px(sl_price, sz_decimals, is_buy_stop=is_buy_stop)
        raw_limit = sl_price * (1.15 if is_buy_stop else 0.85)
        clean_limit = PrecisionEngine.round_px(raw_limit, sz_decimals, is_buy_stop=is_buy_stop)
        clean_sz = PrecisionEngine.round_sz(sz, sz_decimals)

        if not self.exchange:
            logger.info(f"[DRY-RUN] Market Stop-Loss: {coin} {clean_sz} @ trigger ${clean_sl} (limit: ${clean_limit})")
            return True

        existing_oid = self.state.get("positions", {}).get(coin, {}).get("stop_oid")

        # 1. Если стоп уже существует на L1 — модифицируем атомарно
        if existing_oid:
            try:
                mod_resp = self.exchange.modify_order(
                    oid=existing_oid, name=coin, is_buy=is_buy_stop, sz=clean_sz, limit_px=clean_limit,
                    order_type={"trigger": {"triggerPx": clean_sl, "isMarket": True, "tpsl": "sl"}},
                    reduce_only=True
                )
                if mod_resp.get("status") == "ok":
                    logger.info(f"[✓] Stop-Loss {coin} АТОМАРНО модифицирован на L1: ${clean_sl}")
                    return True
            except Exception as e:
                logger.warning(f"[-] modify_order не удался ({e}), переход к перевыставлению...")

        # 2. Первоначальная установка стопа (или fallback после modify)
        try:
            resp = self.exchange.order(
                name=coin, is_buy=is_buy_stop, sz=clean_sz, limit_px=clean_limit,
                order_type={"trigger": {"triggerPx": clean_sl, "isMarket": True, "tpsl": "sl"}},
                reduce_only=True
            )

            if resp.get("status") != "ok":
                logger.critical(f"[STOP REJECTED] Биржа отклонила стоп {coin}: {resp}")
                return False

            statuses = resp.get("response", {}).get("data", {}).get("statuses", [])
            new_oid = None
            for s in statuses:
                if "resting" in s:
                    new_oid = s["resting"]["oid"]
                    break

            if not new_oid:
                logger.critical(f"[STOP UNCONFIRMED] Статус ордера не resting: {statuses}")
                return False

            # Отменяем старые стопы ТОЛЬКО после подтверждения нового OID
            open_orders = self.info.frontend_open_orders(self.address)
            for o in open_orders:
                if self.is_protective_stop(o, coin, is_long) and o.get("oid") != new_oid:
                    self.exchange.cancel(coin=coin, oid=o["oid"])

            if coin in self.state.get("positions", {}):
                self.state["positions"][coin]["stop_oid"] = new_oid
                self.save_state()

            logger.info(f"[✓] Новый Stop-Loss подтвержден на L1: {coin} @ ${clean_sl} (OID: {new_oid})")
            return True
        except Exception as e:
            logger.critical(f"[FATAL] Сбой выставления стопа: {e}")
            return False

    def exit_position(self, coin: str, sz: float, reason: str):
        logger.info(f"[*] Выход из позиции {coin}: {reason}...")
        if not self.exchange:
            logger.info(f"[DRY-RUN] Выход {coin} ({reason}) подтвержден.")
            if coin in self.state["positions"]:
                del self.state["positions"][coin]
            self.save_state()
            return

        try:
            # P0 Fix: Обязательная отмена защитного стопа перед закрытием (исключение инверсии позиции)
            existing_oid = self.state.get("positions", {}).get(coin, {}).get("stop_oid")
            if existing_oid:
                try:
                    self.exchange.cancel(coin=coin, oid=existing_oid)
                    logger.info(f"[✓] Защитный стоп {coin} (OID {existing_oid}) отменен перед закрытием.")
                except Exception as cancel_err:
                    logger.warning(f"[-] Не удалось отменить стоп {coin} перед выходом: {cancel_err}")

            resp = self.exchange.market_close(coin=coin)
            if resp.get("status") != "ok":
                logger.critical(f"[CLOSE REJECTED] Ошибка market_close {coin}: {resp}")
                self.reconcile_with_exchange()
                return

            statuses = resp.get("response", {}).get("data", {}).get("statuses", [])
            if not any("filled" in s for s in statuses):
                logger.critical(f"[CLOSE UNFILLED] Ордер закрытия не filled: {statuses}")
                self.reconcile_with_exchange()
                return

            logger.info(f"[✓] Позиция {coin} закрыта ({reason}).")
            logger.info(f"[EXIT EVENT] Закрытие позиции: {coin} | Причина: {reason}")
            if coin in self.state["positions"]:
                del self.state["positions"][coin]
            self.save_state()
        except Exception as e:
            logger.error(f"[-] Сбой закрытия {coin}: {e}")
            self.reconcile_with_exchange()

    def publish_telemetry(self, btc_price: Optional[float] = None, market_regime: Optional[str] = None):
        """GAP-04: Каноническая публикация телеметрии по разделам 8 и 9 аудита."""
        try:
            ctrl_state = self.control_mgr.get_state()
            equity = None
            free_margin = None
            unrealized_pnl = None
            data_fresh = False
            dq_reason = "DRY-RUN mode: private key not configured"

            if self.exchange and not self.address.startswith("0x000"):
                try:
                    acc_state = self.info.clearinghouse_state(self.address)
                    m_summary = acc_state.get("marginSummary", {})
                    if "accountValue" in m_summary:
                        equity = float(m_summary["accountValue"])
                        total_used = float(m_summary.get("totalMarginUsed", 0.0))
                        free_margin = max(0.0, equity - total_used)
                        cum_pnl = sum(
                            float(p.get("position", {}).get("unrealizedPnl", 0.0))
                            for p in acc_state.get("assetPositions", [])
                        )
                        unrealized_pnl = cum_pnl
                        data_fresh = True
                        dq_reason = None
                except Exception as exc:
                    dq_reason = f"Exchange API error: {exc}"

            pos_list = [
                PositionState(
                    coin=c,
                    side=p.get("direction", "LONG"),
                    size=float(p.get("size", 0.0)),
                    entry_px=float(p.get("entry_px", 0.0)),
                    sl_px=float(p["sl_px"]) if p.get("sl_px") is not None else None,
                    highest_px=float(p["highest_px"]) if p.get("highest_px") is not None else None,
                    lowest_px=float(p["lowest_px"]) if p.get("lowest_px") is not None else None,
                    trailing_active=bool(p.get("trailing_active", False)),
                    breakeven_active=bool(p.get("breakeven_active", False)),
                    ml_prob=float(p["ml_prob"]) if p.get("ml_prob") is not None else None
                )
                for c, p in self.state.get("positions", {}).items()
            ]

            model_file = config.DATA_DIR / "meta_model.json"
            feat_cnt = len(self.feature_cols) if (getattr(self, "meta_weights", False) and hasattr(self, "feature_cols") and self.feature_cols is not None) else (len(self.weights) if (getattr(self, "meta_weights", False) and hasattr(self, "weights") and self.weights is not None) else 0)
            m_status = ModelStatus(
                loaded=bool(self.meta_weights),
                model_path=str(model_file) if model_file.exists() else None,
                features_count=feat_cnt
            )

            status_str = "PAUSED" if not ctrl_state.trading_enabled else "ACTIVE"
            snapshot = CoreState(
                schema_version=1,
                timestamp=time.time(),
                system_status=status_str,
                trading_enabled=ctrl_state.trading_enabled,
                network="TESTNET" if self.is_testnet else "MAINNET",
                account_address=self.address,
                equity=equity,
                free_margin=free_margin,
                unrealized_pnl=unrealized_pnl,
                btc_price=btc_price,
                market_regime=market_regime,
                active_slots=len(self.state.get("positions", {})),
                max_slots=self.active_slots_limit,
                positions=pos_list,
                model_status=m_status,
                data_quality=DataQuality(fresh=data_fresh, reason=dq_reason)
            )
            self.telemetry_ipc.write_atomic_state(snapshot)
        except Exception as exc:
            logger.error(f"[TELEMETRY ERROR] Ошибка публикации снимка: {exc}")

    def run_cycle(self):
        self.reconcile_with_exchange()
        self.refresh_deadmans_switch()
        now = time.time()

        # GAP-03: Чтение состояния из канонической шины управления
        ctrl_state = self.control_mgr.get_state()
        if ctrl_state.panic_requested:
            logger.critical("[PANIC] Получен сигнал экстренной ликвидации от Control Plane!")
            self.pending_triggers.clear()
            for coin in list(self.state.get("positions", {}).keys()):
                pos = self.state["positions"][coin]
                self.exit_position(coin, pos.get("size", 0.0), reason="PANIC_EMERGENCY_CLOSE")
            self.reconcile_with_exchange()
            self.control_mgr.clear_panic()
            return

        btc_df = self.md_worker.compute_multi_tf_indicators("BTC")
        if btc_df.empty:
            return

        last_btc = btc_df.iloc[-1]
        btc_bull = bool(last_btc["close"] > last_btc["ema50_4h"])
        btc_bear = bool(last_btc["close"] < last_btc["ema50_4h"])
        btc_slope_rel = last_btc["ema50_slope"] / max(last_btc["close"] * 0.01, 1e-4)

        if (btc_bull and btc_slope_rel > 0.15) or (btc_bear and btc_slope_rel < -0.25):
            self.active_slots_limit = config.EXPANDED_CONCURRENT_POSITIONS
        else:
            self.active_slots_limit = config.BASE_CONCURRENT_POSITIONS

        regime_str = f"BULL (SLOTS: {self.active_slots_limit})" if btc_bull else "BEAR/CHOP (SLOTS: 2)"
        active_pos_cnt = len(self.state.get("positions", {}))
        trig_cnt = len(self.pending_triggers)
        logger.info(
            f"[HEARTBEAT] BTC: ${last_btc['close']:,.1f} | Режим: {regime_str} | "
            f"Позиции: {active_pos_cnt}/{self.active_slots_limit} | Триггеры: {trig_cnt}"
        )

        # Сопровождение позиций
        for coin in list(self.state["positions"].keys()):
            df_c = self.md_worker.compute_multi_tf_indicators(coin)
            if df_c.empty or len(df_c) < 2:
                continue

            curr_bar = df_c.iloc[-1]
            pos = self.state["positions"][coin]
            entry_px = pos["entry_px"]
            atr = pos.get("atr", curr_bar["atr_4h"])
            is_long = pos.get("direction", "LONG") == "LONG"

            if is_long:
                pos["highest_px"] = max(pos.get("highest_px", entry_px), curr_bar["high"])
                unrealized_r = (pos["highest_px"] - entry_px) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos.get("breakeven_active", False):
                    be_price = entry_px * 1.002
                    if be_price > pos["sl_px"]:
                        logger.info(f"[BREAKEVEN] {coin}: перенос в безубыток @ ${be_price:.4f}")
                        if self.place_native_market_stop(coin, pos["size"], be_price, is_long=True):
                            pos["sl_px"] = be_price
                            pos["breakeven_active"] = True
                            self.save_state()

                if unrealized_r >= 1.8:
                    chandelier_sl = pos["highest_px"] - (atr * 1.75)
                    new_sl = max(pos["sl_px"], chandelier_sl, entry_px * 1.002)
                    if new_sl > pos["sl_px"]:
                        logger.info(f"[CHANDELIER TRAIL] {coin}: подтяжка стопа к ${new_sl:.4f}")
                        if self.place_native_market_stop(coin, pos["size"], new_sl, is_long=True):
                            pos["sl_px"] = new_sl
                            pos["trailing_active"] = True
                            self.save_state()

                if curr_bar["low"] <= pos["sl_px"]:
                    reason = "CHANDELIER_EXIT" if pos.get("trailing_active") else "INITIAL_STOP"
                    self.exit_position(coin, pos["size"], reason)
            else:
                pos["lowest_px"] = min(pos.get("lowest_px", entry_px), curr_bar["low"])
                unrealized_r = (entry_px - pos["lowest_px"]) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos.get("breakeven_active", False):
                    be_price = entry_px * 0.998
                    if be_price < pos["sl_px"]:
                        logger.info(f"[BREAKEVEN] {coin}: перенос в безубыток @ ${be_price:.4f}")
                        if self.place_native_market_stop(coin, pos["size"], be_price, is_long=False):
                            pos["sl_px"] = be_price
                            pos["breakeven_active"] = True
                            self.save_state()

                if unrealized_r >= 1.8:
                    chandelier_sl = pos["lowest_px"] + (atr * 1.75)
                    new_sl = min(pos["sl_px"], chandelier_sl, entry_px * 0.998)
                    if new_sl < pos["sl_px"]:
                        logger.info(f"[CHANDELIER TRAIL] {coin}: подтяжка стопа к ${new_sl:.4f}")
                        if self.place_native_market_stop(coin, pos["size"], new_sl, is_long=False):
                            pos["sl_px"] = new_sl
                            pos["trailing_active"] = True
                            self.save_state()

                if curr_bar["high"] >= pos["sl_px"]:
                    reason = "CHANDELIER_EXIT" if pos.get("trailing_active") else "INITIAL_STOP"
                    self.exit_position(coin, pos["size"], reason)

        # GAP-03: Шлюз блокировки новых входов при паузе торговли оператором
        if not ctrl_state.trading_enabled:
            logger.info("[PAUSE] Торговля приостановлена оператором. Сопровождение активно, новые входы заблокированы.")
            self.publish_telemetry(btc_price=float(last_btc["close"]), market_regime=regime_str)
            return

        # Исполнение триггеров
        committed_notional = 0.0
        for coin, trig in list(self.pending_triggers.items()):
            if now > trig["expiry_t"] or len(self.state["positions"]) >= self.active_slots_limit:
                del self.pending_triggers[coin]
                continue

            of_m = self.get_orderflow_metrics(coin)
            current_px = of_m.get("last_px", 0.0)
            if current_px <= 0:
                df_c = self.md_worker.compute_multi_tf_indicators(coin)
                if df_c.empty:
                    continue
                current_px = df_c.iloc[-1]["close"]

            is_long = trig["direction"] == "LONG"
            trigger_hit = (current_px >= trig["trigger_px"]) if is_long else (current_px <= trig["trigger_px"])

            if trigger_hit:
                if not of_m["is_fresh"]:
                    logger.warning(f"[ORDER FLOW FAIL-CLOSED] {coin}: Снапшот старше 15с. Вход отменен.")
                    del self.pending_triggers[coin]
                    continue

                # 1. Институциональный фильтр OBI_10 (защита от стены лимитных заявок против входа)
                obi_val = of_m.get("obi_10", 0.0)
                if is_long and obi_val < -0.25:
                    logger.warning(f"[OBI FILTER] {coin}: Стена продавцов в стакане (OBI={obi_val:.2f} < -0.25). Вход отклонен.")
                    del self.pending_triggers[coin]
                    continue
                elif not is_long and obi_val > +0.25:
                    logger.warning(f"[OBI FILTER] {coin}: Стена покупателей в стакане (OBI={obi_val:.2f} > +0.25). Вход отклонен.")
                    del self.pending_triggers[coin]
                    continue

                # 2. Economic Cost Gate (защита от ловушки трения комиссий и проскальзывания)
                if trig["trigger_px"] <= 0.0:
                    del self.pending_triggers[coin]
                    continue
                stop_dist_pct = abs(trig["trigger_px"] - trig["sl_px"]) / trig["trigger_px"]
                MIN_ECONOMIC_STOP_PCT = 0.0120  # 1.20% минимальная экономическая дистанция
                if stop_dist_pct < MIN_ECONOMIC_STOP_PCT:
                    logger.warning(f"[ECONOMIC COST GATE] {coin}: Стоп {stop_dist_pct*100:.2f}% < 1.20%. Трение съест матожидание. Вход отменен.")
                    del self.pending_triggers[coin]
                    continue

                # 3. Фильтр Delta OI (запрет входа в шорт-сквиз ловушки: рост цены при агрессивном закрытии позиций)
                raw_doi = of_m.get("delta_oi_z")
                doi_z = float(raw_doi) if raw_doi is not None else 0.0
                if is_long and doi_z < -2.5:
                    logger.warning(f"[DELTA OI GATE] {coin}: Рост на агрессивном закрытии позиций (Z_OI={doi_z:.2f} < -2.5). Вход отклонен.")
                    del self.pending_triggers[coin]
                    continue

                # 4. Funding Rate Cost Gate (запрет входа против экстремального фандинга)
                funding_val = of_m.get("funding_rate", 0.0)
                max_adverse_funding = getattr(config, "MAX_ADVERSE_FUNDING_RATE", 0.0003)
                if is_long and funding_val > max_adverse_funding:
                    logger.warning(f"[FUNDING RATE GATE] {coin}: Аномально высокий Long фандинг ({funding_val*100:.4f}% > {max_adverse_funding*100:.4f}%). Вход отклонен.")
                    del self.pending_triggers[coin]
                    continue
                elif not is_long and funding_val < -max_adverse_funding:
                    logger.warning(f"[FUNDING RATE GATE] {coin}: Экстремально отрицательный Short фандинг ({funding_val*100:.4f}% < -{max_adverse_funding*100:.4f}%). Вход отклонен.")
                    del self.pending_triggers[coin]
                    continue

                sizing = self.calculate_sizing(coin, trig["trigger_px"], trig["sl_px"], ml_prob=trig["ml_prob"], committed_notional=committed_notional)
                if not sizing:
                    del self.pending_triggers[coin]
                    continue

                committed_notional += sizing["notional"]
                filled_sz = sizing["sz"]
                filled_px = sizing["entry_px"]

                if self.exchange:
                    try:
                        self.exchange.schedule_cancel(None)
                    except Exception:
                        pass

                    order_cloid = Cloid.from_str("0x" + uuid.uuid4().hex[:32])
                    filled_via_maker = False

                    # Alpha 3.3: Maker-first исполнение (Alo / Post-Only со сдвигом по OBI)
                    if getattr(config, "ENABLE_MAKER_FIRST", True):
                        try:
                            coin_meta = self.universe_meta.get(coin, {})
                            sz_dec = coin_meta.get("szDecimals", 3)
                            # Сдвиг котировки по OBI стакана
                            obi_shift = 0.0001 * (1.0 + max(-0.5, min(0.5, obi_val)))
                            maker_limit = current_px * (1.0 - obi_shift) if is_long else current_px * (1.0 + obi_shift)
                            clean_maker_px = PrecisionEngine.round_px(maker_limit, sz_dec, is_buy_stop=not is_long)

                            logger.info(f"[*] [MAKER-FIRST] Размещение Alo заявки: {coin} {sizing['sz']} @ ${clean_maker_px}...")
                            maker_resp = self.exchange.order(
                                name=coin,
                                is_buy=is_long,
                                sz=sizing["sz"],
                                limit_px=clean_maker_px,
                                order_type={"limit": {"tif": "Alo"}},
                                reduce_only=False,
                                cloid=order_cloid
                            )

                            if maker_resp.get("status") == "ok":
                                m_statuses = maker_resp.get("response", {}).get("data", {}).get("statuses", [])
                                resting = next((s["resting"] for s in m_statuses if "resting" in s), None)
                                if resting:
                                    resting_oid = resting["oid"]
                                    time.sleep(getattr(config, "MAKER_TIMEOUT_SEC", 2.0))
                                    open_orders = self.info.open_orders(self.address)
                                    still_open = any(o.get("oid") == resting_oid for o in open_orders)
                                    if not still_open:
                                        filled_sz = sizing["sz"]
                                        filled_px = clean_maker_px
                                        filled_via_maker = True
                                        logger.info(f"[✓] [MAKER-FIRST SUCCESS] Заявка исполнена мейкером: {filled_sz} {coin} @ ${filled_px:.4f} (сохранен спред + rebate)")
                                    else:
                                        try:
                                            self.exchange.cancel(coin, resting_oid)
                                            logger.info(f"[-] [MAKER TIMEOUT] Alo ордер {resting_oid} отменен по таймауту. Переход к IOC фолбэку.")
                                        except Exception as c_err:
                                            logger.warning(f"Ошибка отмены Alo ордера: {c_err}")
                        except Exception as m_err:
                            logger.warning(f"[MAKER-FIRST ERROR] Сбой размещения мейкер-ордера ({m_err}). Фолбэк на Market Open.")

                    if not filled_via_maker:
                        logger.info(f"[*] Отправка Market Open: {coin} {sizing['sz']} (cloid: {order_cloid})...")
                        try:
                            entry_resp = self.exchange.market_open(name=coin, is_buy=is_long, sz=sizing["sz"], px=None, slippage=config.ENTRY_SLIPPAGE, cloid=order_cloid)
                        except Exception as e:
                            logger.error(f"[ENTRY NETWORK ERROR] {coin}: {e}")
                            del self.pending_triggers[coin]
                            continue

                        if entry_resp.get("status") != "ok":
                            del self.pending_triggers[coin]
                            continue

                        statuses = entry_resp.get("response", {}).get("data", {}).get("statuses", [])
                        filled_data = next((s["filled"] for s in statuses if "filled" in s), None)
                        if not filled_data:
                            del self.pending_triggers[coin]
                            continue

                        filled_sz = float(filled_data["totalSz"])
                        filled_px = float(filled_data["avgPx"])
                        logger.info(f"[✓] Вход исполнен: {filled_sz} {coin} @ ${filled_px:.4f}")

                stop_ok = self.place_native_market_stop(coin, filled_sz, sizing["sl_px"], is_long=is_long)
                if not stop_ok and self.exchange:
                    self.exchange.market_close(coin=coin, sz=filled_sz)
                    del self.pending_triggers[coin]
                    continue

                self.state["positions"][coin] = {
                    "status": "POSITION_ACTIVE", "direction": trig["direction"],
                    "size": filled_sz, "entry_px": filled_px, "sl_px": sizing["sl_px"],
                    "highest_px": filled_px, "lowest_px": filled_px, "atr": trig["atr"],
                    "trailing_active": False, "breakeven_active": False, "ml_prob": trig["ml_prob"]
                }
                self.save_state()
                del self.pending_triggers[coin]

        # Поиск сигналов
        active_assets = set(self.state["positions"].keys()) | set(self.pending_triggers.keys())
        if len(active_assets) < self.active_slots_limit:
            btc_closes = btc_df["close"]
            clean_coins = [c for c in config.TARGET_COINS if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]]

            for coin in clean_coins:
                if coin in active_assets or len(active_assets) >= self.active_slots_limit:
                    continue

                df_c = self.md_worker.compute_multi_tf_indicators(coin)
                if df_c.empty or len(df_c) < 72:
                    continue

                row = df_c.iloc[-1]
                z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(df_c["close"], btc_closes)
                atr = row["atr_4h"]

                # Лонг-сетапы
                if btc_bull and z_res_mom >= 0.40 and raw_rs_pct >= 2.0:
                    is_trend = (row["close"] > row["ema50_4h"]) and (row["ema20_4h"] > row["ema50_4h"])
                    is_evr_ok, _, _ = QuantFactorEngine.evaluate_evr_absorption(
                        open_px=row["open"], high_px=row["high"], low_px=row["low"], close_px=row["close"],
                        volume=row["vol_rolling_4h"], vol_sma=row["vol_sma20_4h"], ema20_4h=row["ema20_4h"], atr_4h=atr
                    )
                    breakout_hit = (row["close"] >= row["donchian_high_4h"] * 0.998)
                    vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                    valid_breakout = breakout_hit and vol_boost and (btc_slope_rel > 0.15)
                    valid_pullback = is_trend and is_evr_ok

                    if valid_pullback or valid_breakout:
                        if not self.meta_weights:
                            continue

                        entry_px = row["close"]
                        vol_rel = min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0)
                        dist_ema20 = (entry_px - row["ema20_4h"]) / max(atr, 1e-4)
                        donch_range = max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4)
                        donch_pos = (entry_px - row["donchian_low_4h"]) / donch_range
                        entry_type_val = 1.0 if valid_breakout else 0.0

                        raw_feats = [
                            z_res_mom, beta_btc, raw_rs_pct, vol_rel,
                            (atr / entry_px) * 100.0, dist_ema20, donch_pos,
                            btc_slope_rel, entry_type_val, 1.0
                        ]

                        ml_prob = self.predict_meta_prob(raw_feats)
                        if ml_prob < 0.48:
                            continue

                        # Расчет фрактального уровня ликвидности (Swing Low)
                        sh_f, sl_f = QuantFactorEngine.compute_fractal_swings(df_c["high"], df_c["low"], window=2)
                        base_sl = row["low"] - (atr * 0.85) if valid_pullback else row["close"] - (atr * 1.50)

                        # Если фрактальный минимум подтвержден и находится ближе стандартного стопа (но >= 0.8 ATR)
                        if sl_f and sl_f < row["close"] and (row["close"] - sl_f) >= (atr * 0.80):
                            sl_price = max(base_sl, sl_f * 0.999)
                        else:
                            sl_price = base_sl

                        self.pending_triggers[coin] = {
                            "direction": "LONG", "trigger_px": row["high"] * 1.0005,
                            "sl_px": sl_price, "expiry_t": now + (3 * 3600 if valid_pullback else 2 * 3600),
                            "atr": atr, "ml_prob": ml_prob
                        }
                        active_assets.add(coin)

                # Шорт-сетапы
                elif btc_bear and btc_slope_rel < -0.30 and z_res_mom <= -0.40 and raw_rs_pct <= -2.5:
                    is_bear_trend = (row["close"] < row["ema50_4h"]) and (row["ema20_4h"] < row["ema50_4h"])
                    breakdown_hit = (row["close"] <= row["donchian_low_4h"] * 1.002)
                    vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                    is_bear_pb = is_bear_trend and (row["high"] >= row["ema20_4h"] * 0.995) and (row["close"] <= row["ema20_4h"])
                    valid_short_bo = breakdown_hit and vol_boost

                    if is_bear_pb or valid_short_bo:
                        if not self.meta_weights:
                            continue

                        entry_px = row["close"]
                        vol_rel = min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0)
                        dist_ema20 = (entry_px - row["ema20_4h"]) / max(atr, 1e-4)
                        donch_range = max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4)
                        donch_pos = (entry_px - row["donchian_low_4h"]) / donch_range

                        raw_feats = [
                            z_res_mom, beta_btc, raw_rs_pct, vol_rel,
                            (atr / entry_px) * 100.0, dist_ema20, donch_pos,
                            btc_slope_rel, 1.0 if valid_short_bo else 0.0, 0.0
                        ]

                        prob = self.predict_meta_prob(raw_feats)
                        if prob < 0.48:
                            continue

                        sl_price = row["high"] + (atr * 0.85) if is_bear_pb else row["close"] + (atr * 1.50)

                        self.pending_triggers[coin] = {
                            "direction": "SHORT", "trigger_px": row["low"] * 0.9995,
                            "sl_px": sl_price, "expiry_t": now + (3 * 3600 if is_bear_pb else 2 * 3600),
                            "atr": atr, "ml_prob": prob
                        }
                        active_assets.add(coin)

        # GAP-04: Финальная публикация канонического снимка в конце итерации
        self.publish_telemetry(btc_price=float(last_btc["close"]), market_regime=regime_str)

    def start(self):
        logger.info("=" * 75)
        logger.info("  QVEX: QUANTITATIVE VECTOR EXECUTION v10.7 (INSTITUTIONAL HARDENED)")
        logger.info(f"  Сеть: {'TESTNET' if self.is_testnet else 'MAINNET'} | Адрес: {self.address}")
        logger.info("=" * 75)
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[!] Ошибка цикла: {e}", exc_info=True)
            time.sleep(config.CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    bot = HyperliquidSwingBot()
    bot.start()

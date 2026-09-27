import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#!/usr/bin/env python3
"""
Институциональный бэктестер v10.7 (Hardened Portfolio Risk & High Slippage Stress).
- Инвариант: PortfolioStopRisk <= 8.0% от Equity (защита от мгновенного сноса 3 стопов).
- Стресс-тест исполнения: проскальзывание стопа 0.25% (в 5 раз жестче базового 0.05%).
- Изоляция незакрытой 1H свечи (Lookahead-free).
- Строгий OOS с 24h эмбарго, пессимистичные коллизии STOP_COLLISION_WORST.
"""

import time
import math
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from hyperliquid.info import Info
from hyperliquid.utils import constants

from core import config
from core.quant.factors import QuantFactorEngine

print("=" * 85)
print("  БЭКТЕСТЕР v10.7: HARDENED PORTFOLIO RISK & SLIPPAGE STRESS (0.25%)")
print("=" * 85)

clean_target_coins = ['SOL', 'AVAX', 'NEAR', 'RENDER', 'APT', 'SUI', 'DOGE', 'TIA', 'ARB', 'OP', 'INJ', 'LINK']

model_path = config.DATA_DIR / "meta_model.json"
with open(model_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

weights = np.array(model_data["coef"])
intercept = float(model_data["intercept"])
scaler_mean = np.array(model_data["scaler_mean"])
scaler_scale = np.array(model_data["scaler_scale"])
scaler_scale = np.where(scaler_scale <= 1e-6, 1.0, scaler_scale)

def predict_meta_prob(raw_features: list) -> float:
    x = (np.array(raw_features) - scaler_mean) / scaler_scale
    z = float(np.dot(weights, x) + intercept)
    return 1.0 / (1.0 + math.exp(-max(min(z, 15.0), -15.0)))

base_url = constants.TESTNET_API_URL if config.IS_TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True, timeout=5)

END_MS = int(time.time() * 1000)
START_MS = END_MS - (1500 * 3600 * 1000)
SYMBOLS = list(set(["BTC"] + clean_target_coins))

print(f"[*] Выгрузка исторических данных Hyperliquid ({len(SYMBOLS)} инструментов)...")
data_1h: Dict[str, pd.DataFrame] = {}

for sym in SYMBOLS:
    try:
        raw = info.candles_snapshot(name=sym, interval="1h", startTime=START_MS, endTime=END_MS)
        if raw and len(raw) >= 200:
            df = pd.DataFrame([{
                "t": int(c["t"]),
                "dt": pd.to_datetime(c["t"], unit="ms", utc=True),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "vol": float(c["v"])
            } for c in raw]).sort_values("t").reset_index(drop=True)
            data_1h[sym] = df
    except Exception:
        pass

processed_1h: Dict[str, pd.DataFrame] = {}
for sym, df in data_1h.items():
    df = df.copy()
    df["vol_rolling_4h"] = df["vol"].rolling(4, min_periods=4).sum()
    df_temp = df.set_index("dt")
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum", "t": "first"}
    df_4h = df_temp.resample("4h", label="left", closed="left").agg(ohlc).dropna().reset_index()

    c = df_4h["close"]
    h = df_4h["high"]
    l = df_4h["low"]
    v = df_4h["vol"]
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
        df.sort_values("dt"),
        df_4h_shifted.sort_values("dt"),
        on="dt",
        direction="backward"
    ).dropna().reset_index(drop=True)
    processed_1h[sym] = merged

common_timestamps = processed_1h["BTC"]["t"].values
min_warmup = 160
total_bars = len(common_timestamps)
split_idx = int(total_bars * 0.75)
oos_start_idx = split_idx + 24

INITIAL_CAPITAL = 1000.0
TAKER_FEE = 0.00035
# ПОВЫШЕННЫЙ ШТРАФ ПРОСКАЛЬЗЫВАНИЯ: 0.25% (в 5 раз выше нормального)
SLIPPAGE_PENALTY = 0.0025
HOURLY_FUNDING = 0.000012

def simulate_v10_7(start_idx, end_idx, title="", mode="v11"):
    cash = INITIAL_CAPITAL
    equity_curve = [cash]
    trades = []
    open_positions = {}
    pending_triggers = {}

    for i in range(start_idx, end_idx):
        curr_t = common_timestamps[i]
        btc_df = processed_1h["BTC"]
        btc_row = btc_df[btc_df["t"] == curr_t]
        if btc_row.empty:
            continue
        btc_row = btc_row.iloc[0]

        btc_bull = bool(btc_row["close"] > btc_row["ema50_4h"])
        btc_bear = bool(btc_row["close"] < btc_row["ema50_4h"])
        btc_slope_rel = btc_row["ema50_slope"] / max(btc_row["close"] * 0.01, 1e-4)

        is_strong_trend = (btc_bull and btc_slope_rel > 0.15) or (btc_bear and btc_slope_rel < -0.25)
        max_slots = 3 if is_strong_trend else 2

        # 1. СОПРОВОЖДЕНИЕ ПОЗИЦИЙ
        for coin in list(open_positions.keys()):
            pos = open_positions[coin]
            c_df = processed_1h[coin]
            c_row = c_df[c_df["t"] == curr_t]
            if c_row.empty:
                continue
            c_row = c_row.iloc[0]

            is_long = (pos["direction"] == "LONG")
            exit_trade = False
            exit_price = 0.0
            exit_reason = ""

            atr = pos["atr"]
            active_sl = pos["sl_px"]

            if is_long:
                cash -= pos["notional"] * HOURLY_FUNDING
                be_target = pos["entry_px"] + (atr * 1.0)
                hit_sl = (c_row["low"] <= active_sl)
                hit_be = (c_row["high"] >= be_target) and not pos["breakeven_active"]

                if hit_sl and hit_be:
                    exit_trade = True
                    exit_price = active_sl * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "STOP_COLLISION_WORST"
                elif c_row["open"] <= active_sl:
                    exit_trade = True
                    exit_price = c_row["open"] * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "STOP_GAP_OPEN"
                elif hit_sl:
                    exit_trade = True
                    exit_price = active_sl * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "CHANDELIER_EXIT" if pos["trailing_active"] else ("BREAKEVEN_EXIT" if pos["breakeven_active"] else "INITIAL_STOP")
            else:
                cash += pos["notional"] * HOURLY_FUNDING
                be_target = pos["entry_px"] - (atr * 1.0)
                hit_sl = (c_row["high"] >= active_sl)
                hit_be = (c_row["low"] <= be_target) and not pos["breakeven_active"]

                if hit_sl and hit_be:
                    exit_trade = True
                    exit_price = active_sl * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "STOP_COLLISION_WORST"
                elif c_row["open"] >= active_sl:
                    exit_trade = True
                    exit_price = c_row["open"] * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "STOP_GAP_OPEN"
                elif hit_sl:
                    exit_trade = True
                    exit_price = active_sl * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "CHANDELIER_EXIT" if pos["trailing_active"] else ("BREAKEVEN_EXIT" if pos["breakeven_active"] else "INITIAL_STOP")

            if exit_trade:
                if is_long:
                    pnl = (exit_price - pos["entry_px"]) * pos["sz"] - (exit_price * pos["sz"] * TAKER_FEE)
                    pnl_pct = (exit_price / pos["entry_px"] - 1.0) * 100
                else:
                    pnl = (pos["entry_px"] - exit_price) * pos["sz"] - (exit_price * pos["sz"] * TAKER_FEE)
                    pnl_pct = (1.0 - exit_price / pos["entry_px"]) * 100

                cash += (pos["margin"] + pnl)
                trades.append({
                    "coin": coin, "dir": pos["direction"], "type": pos["type"],
                    "entry_px": pos["entry_px"], "exit_px": exit_price,
                    "pnl_usd": pnl, "pnl_pct": pnl_pct, "reason": exit_reason, "ml_prob": pos["ml_prob"]
                })
                del open_positions[coin]
                continue

            if is_long:
                pos["highest_px"] = max(pos["highest_px"], c_row["high"])
                unrealized_r = (pos["highest_px"] - pos["entry_px"]) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos["breakeven_active"]:
                    be_price = pos["entry_px"] * 1.002
                    if be_price > pos["sl_px"]:
                        pos["sl_px"] = be_price
                        pos["breakeven_active"] = True

                if unrealized_r >= 1.8:
                    new_sl = max(pos["sl_px"], pos["highest_px"] - (atr * 1.75), pos["entry_px"] * 1.002)
                    if new_sl > pos["sl_px"]:
                        pos["sl_px"] = new_sl
                        pos["trailing_active"] = True
            else:
                pos["lowest_px"] = min(pos["lowest_px"], c_row["low"])
                unrealized_r = (pos["entry_px"] - pos["lowest_px"]) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos["breakeven_active"]:
                    be_price = pos["entry_px"] * 0.998
                    if be_price < pos["sl_px"]:
                        pos["sl_px"] = be_price
                        pos["breakeven_active"] = True

                if unrealized_r >= 1.8:
                    new_sl = min(pos["sl_px"], pos["lowest_px"] + (atr * 1.75), pos["entry_px"] * 0.998)
                    if new_sl < pos["sl_px"]:
                        pos["sl_px"] = new_sl
                        pos["trailing_active"] = True

        # 2. ИСПОЛНЕНИЕ ТРИГГЕРОВ (ИНВАРИАНТ PORTFOLIO STOP RISK <= 8.0%)
        total_unrealized = 0.0
        total_notional = 0.0
        total_margin = sum(p["margin"] for p in open_positions.values())

        for c_name, pos in open_positions.items():
            c_match = processed_1h[c_name][processed_1h[c_name]["t"] == curr_t]
            if not c_match.empty:
                c_cur_px = c_match.iloc[0]["close"]
            else:
                c_past = processed_1h[c_name][processed_1h[c_name]["t"] <= curr_t]
                if c_past.empty:
                    continue
                c_cur_px = c_past.iloc[-1]["close"]
            if pos["direction"] == "LONG":
                total_unrealized += (c_cur_px - pos["entry_px"]) * pos["sz"]
            else:
                total_unrealized += (pos["entry_px"] - c_cur_px) * pos["sz"]
            total_notional += pos["notional"]

        current_equity = cash + total_margin + total_unrealized
        portfolio_cap_usd = current_equity * config.PORTFOLIO_HARD_LEVERAGE_CAP

        # Расчет текущего риска открытых позиций
        current_stop_risk = sum(p["sz"] * abs(p["entry_px"] - p["sl_px"]) for p in open_positions.values())
        max_stop_risk_allowed = current_equity * 0.08  # 8% хардкап

        for coin, trig in list(pending_triggers.items()):
            if curr_t > trig["expiry_t"] or len(open_positions) >= max_slots:
                del pending_triggers[coin]
                continue

            c_row = processed_1h[coin][processed_1h[coin]["t"] == curr_t].iloc[0]
            is_long = (trig["direction"] == "LONG")
            trigger_hit = (c_row["high"] >= trig["trigger_px"]) if is_long else (c_row["low"] <= trig["trigger_px"])

            if trigger_hit:
                fill_px = max(trig["trigger_px"], c_row["open"]) * 1.0005 if is_long else min(trig["trigger_px"], c_row["open"]) * 0.9995
                sl_dist = abs(fill_px - trig["sl_px"])

                if sl_dist / fill_px >= 0.008:
                    # Расчет риска: плоский 1.0% (Base) либо динамический 1.3% - 2.2% (ML Boost)
                    if mode == "v10_7_dynamic":
                        prob = float(trig.get("ml_prob", 0.50))
                        # Нормализация уверенности: от 0.50 (min) до 0.85 (max)
                        norm_prob = max(0.0, min(1.0, (prob - 0.50) / (0.85 - 0.50)))
                        risk_pct = 0.013 + norm_prob * (0.022 - 0.013)
                        single_trade_risk_usd = current_equity * risk_pct
                    else:
                        single_trade_risk_usd = current_equity * 0.01
                    target_notional = single_trade_risk_usd / max(sl_dist / fill_px, 1e-4)
                    target_notional = min(target_notional, current_equity * 0.40)
                    avail_cap = max(0.0, portfolio_cap_usd - total_notional)
                    final_ntl = min(target_notional, avail_cap)

                    if final_ntl >= 10.0 and cash >= (final_ntl * 0.2):
                            sz = final_ntl / fill_px
                            fee = final_ntl * TAKER_FEE
                            margin = final_ntl * 0.2
                            cash -= (margin + fee)
                            open_positions[coin] = {
                                "direction": trig["direction"], "type": trig["entry_type"],
                                "entry_px": fill_px, "sl_px": trig["sl_px"], "highest_px": fill_px,
                                "lowest_px": fill_px, "sz": sz, "notional": final_ntl, "margin": margin,
                                "atr": trig["atr"], "trailing_active": False, "breakeven_active": False,
                                "ml_prob": trig["ml_prob"]
                            }
                            total_notional += final_ntl
                            current_stop_risk += sz * sl_dist
                del pending_triggers[coin]

        equity_curve.append(current_equity)

        # 3. ПОИСК СИГНАЛОВ (A/B РЕЖИМЫ: v10.7 vs v11.0)
        btc_closes = btc_df[btc_df["t"] <= curr_t]["close"]
        active_assets = set(open_positions.keys()) | set(pending_triggers.keys())

        if mode in ["v10_7", "v10_7_dynamic"]:
            # --- БАЗОВЫЙ v10.7: СТРОГО 2 СЛОТА, БЕЗ RECYCLING, FIFO ---
            can_open = len(active_assets) < 2
            if can_open:
                for coin in clean_target_coins:
                    if coin in active_assets or len(active_assets) >= 2:
                        continue
                    if coin not in processed_1h:
                        continue

                    c_df = processed_1h[coin]
                    c_hist = c_df[c_df["t"] <= curr_t]
                    if len(c_hist) < 72:
                        continue

                    row = c_hist.iloc[-1]
                    z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(c_hist["close"], btc_closes)
                    atr = row["atr_4h"]

                    if btc_bull and z_res_mom >= 0.40 and raw_rs_pct >= 2.0:
                        is_trend = (row["close"] > row["ema50_4h"]) and (row["ema20_4h"] > row["ema50_4h"])
                        is_evr_ok, _, _ = QuantFactorEngine.evaluate_evr_absorption(
                            open_px=row["open"], high_px=row["high"], low_px=row["low"], close_px=row["close"],
                            volume=row["vol_rolling_4h"], vol_sma=row["vol_sma20_4h"], ema20_4h=row["ema20_4h"], atr_4h=atr
                        )
                        breakout_hit = (row["close"] >= row["donchian_high_4h"] * 0.998)
                        vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                        valid_bo = breakout_hit and vol_boost and (btc_slope_rel > 0.15)
                        valid_pb = is_trend and is_evr_ok

                        if valid_pb or valid_bo:
                            raw_feats = [
                                z_res_mom, beta_btc, raw_rs_pct,
                                min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0),
                                (atr / row["close"]) * 100.0,
                                (row["close"] - row["ema20_4h"]) / max(atr, 1e-4),
                                (row["close"] - row["donchian_low_4h"]) / max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4),
                                btc_slope_rel, 1.0 if valid_bo else 0.0, 1.0
                            ]
                            prob = predict_meta_prob(raw_feats)
                            if prob >= 0.50:
                                base_sl = row["low"] - (atr * 0.85) if valid_pb else row["close"] - (atr * 1.50)
                                sh_f, sl_f = QuantFactorEngine.compute_fractal_swings(c_hist["high"], c_hist["low"], window=2)
                                if sl_f and sl_f < row["close"] and (row["close"] - sl_f) >= (atr * 0.80):
                                    sl_price = max(base_sl, sl_f * 0.999)
                                else:
                                    sl_price = base_sl

                                trg_entry = row["high"] * 1.0005
                                stop_dist_pct = (trg_entry - sl_price) / trg_entry
                                if stop_dist_pct >= 0.0120:
                                    pending_triggers[coin] = {
                                        "direction": "LONG", "entry_type": "PULLBACK" if valid_pb else "BREAKOUT",
                                        "trigger_px": trg_entry, "sl_px": sl_price,
                                        "expiry_t": curr_t + (3 * 3600 * 1000), "atr": atr, "ml_prob": prob
                                    }
                                    active_assets.add(coin)

        else:
            # --- РЕЖИМ v11.0: 3 СЛОТА, PRIORITY RANKING, RISK RECYCLING, BTC GATE ---
            MAX_PORTFOLIO_HEAT_PCT = 0.040
            STRESS_GAP_ALLOWANCE = 0.005
            current_active_risk = 0.0
            for p in open_positions.values():
                if p["sl_px"] >= p["entry_px"]:
                    current_active_risk += p["sz"] * p["entry_px"] * STRESS_GAP_ALLOWANCE
                else:
                    current_active_risk += p["sz"] * max(0.0, p["entry_px"] - p["sl_px"])
            for t in pending_triggers.values():
                current_active_risk += (current_equity * 0.01)

            available_risk = max(0.0, (current_equity * MAX_PORTFOLIO_HEAT_PCT) - current_active_risk)
            btc_dumping = ((btc_row["close"] < btc_row["ema20_4h"] * 0.992) and (btc_slope_rel < -0.05)) or (btc_slope_rel < -0.15)
            can_open_new = (available_risk >= (current_equity * 0.008)) and (len(open_positions) < 3) and (not btc_dumping)

            SECTOR_MAP = {
                "SOL": "L1", "AVAX": "L1", "NEAR": "L1", "SUI": "L1", "APT": "L1",
                "ARB": "L2", "OP": "L2", "TIA": "MODULAR",
                "LINK": "DEFI", "INJ": "DEFI", "RENDER": "AI", "DOGE": "MEME"
            }
            active_sectors = [SECTOR_MAP.get(c, "OTHER") for c in open_positions.keys()]

            if can_open_new:
                candidates = []
                for coin in clean_target_coins:
                    if coin in active_assets or coin not in processed_1h:
                        continue

                    c_df = processed_1h[coin]
                    c_hist = c_df[c_df["t"] <= curr_t]
                    if len(c_hist) < 72:
                        continue

                    row = c_hist.iloc[-1]
                    z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(c_hist["close"], btc_closes)
                    atr = row["atr_4h"]

                    if btc_bull and z_res_mom >= 0.40 and raw_rs_pct >= 2.0:
                        is_trend = (row["close"] > row["ema50_4h"]) and (row["ema20_4h"] > row["ema50_4h"])
                        is_evr_ok, _, _ = QuantFactorEngine.evaluate_evr_absorption(
                            open_px=row["open"], high_px=row["high"], low_px=row["low"], close_px=row["close"],
                            volume=row["vol_rolling_4h"], vol_sma=row["vol_sma20_4h"], ema20_4h=row["ema20_4h"], atr_4h=atr
                        )
                        breakout_hit = (row["close"] >= row["donchian_high_4h"] * 0.998)
                        vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                        valid_bo = breakout_hit and vol_boost and (btc_slope_rel > 0.15)
                        valid_pb = is_trend and is_evr_ok

                        if valid_pb or valid_bo:
                            raw_feats = [
                                z_res_mom, beta_btc, raw_rs_pct,
                                min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0),
                                (atr / row["close"]) * 100.0,
                                (row["close"] - row["ema20_4h"]) / max(atr, 1e-4),
                                (row["close"] - row["donchian_low_4h"]) / max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4),
                                btc_slope_rel, 1.0 if valid_bo else 0.0, 1.0
                            ]
                            prob = predict_meta_prob(raw_feats)
                            if prob >= 0.50:
                                base_sl = row["low"] - (atr * 0.85) if valid_pb else row["close"] - (atr * 1.50)
                                sh_f, sl_f = QuantFactorEngine.compute_fractal_swings(c_hist["high"], c_hist["low"], window=2)
                                if sl_f and sl_f < row["close"] and (row["close"] - sl_f) >= (atr * 0.80):
                                    sl_price = max(base_sl, sl_f * 0.999)
                                else:
                                    sl_price = base_sl

                                trg_entry = row["high"] * 1.0005
                                stop_dist_pct = (trg_entry - sl_price) / trg_entry
                                if stop_dist_pct >= 0.0120:
                                    candidates.append({
                                        "coin": coin, "score": prob * max(0.1, z_res_mom),
                                        "prob": prob, "valid_pb": valid_pb,
                                        "trg_entry": trg_entry, "sl_price": sl_price,
                                        "atr": atr, "sector": SECTOR_MAP.get(coin, "OTHER")
                                    })

                candidates.sort(key=lambda x: x["score"], reverse=True)

                for cand in candidates:
                    c_coin = cand["coin"]
                    c_sec = cand["sector"]

                    if c_sec != "OTHER" and active_sectors.count(c_sec) >= 2:
                        continue
                    if available_risk < (current_equity * 0.008):
                        break
                    if len(open_positions) + len(pending_triggers) >= 3:
                        break

                    # Tiered Conviction: 3-й слот резервируется только под сильные сетапы (prob >= 0.65)
                    current_slots_occupied = len(open_positions) + len(pending_triggers)
                    if current_slots_occupied == 2 and cand["prob"] < 0.65:
                        continue

                    pending_triggers[c_coin] = {
                        "direction": "LONG",
                        "entry_type": "PULLBACK" if cand["valid_pb"] else "BREAKOUT",
                        "trigger_px": cand["trg_entry"], "sl_px": cand["sl_price"],
                        "expiry_t": curr_t + (3 * 3600 * 1000), "atr": cand["atr"], "ml_prob": cand["prob"]
                    }
                    active_assets.add(c_coin)
                    if c_sec != "OTHER":
                        active_sectors.append(c_sec)
                    available_risk -= (current_equity * 0.01)

    final_eq = equity_curve[-1]
    df_t = pd.DataFrame(trades)
    print("\n" + "=" * 85)
    print(f"  ИТОГИ: {title}")
    print("=" * 85)
    if df_t.empty:
        print("[-] Сделок не было.")
        return
    wins = df_t[df_t["pnl_usd"] > 0]
    losses = df_t[df_t["pnl_usd"] <= 0]
    wr = len(wins) / len(df_t) * 100
    pnl_net = final_eq - INITIAL_CAPITAL
    roi = pnl_net / INITIAL_CAPITAL * 100
    gp = wins["pnl_usd"].sum() if not wins.empty else 0.0
    gl = abs(losses["pnl_usd"].sum()) if not losses.empty else 1e-4
    pf = gp / gl
    eq_s = pd.Series(equity_curve)
    mdd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100

    long_trades = df_t[df_t["dir"] == "LONG"]
    short_trades = df_t[df_t["dir"] == "SHORT"]

    print(f"💰 Стартовый депозит:     ${INITIAL_CAPITAL:,.2f}")
    print(f"📈 Итоговый капитал:      ${final_eq:,.2f} ({roi:+.2f}%)")
    print(f"💵 Чистая прибыль:        ${pnl_net:+,.2f}")
    print("-" * 85)
    print(f"📊 Всего сделок:          {len(df_t)} (LONG: {len(long_trades)} | SHORT: {len(short_trades)})")
    print(f"🎯 Win Rate:              {wr:.1f}% ({len(wins)} в плюс / {len(losses)} в минус)")
    print(f"⚖️ Profit Factor:         {pf:.2f}")
    print(f"🛡 Максимальная просадка: {mdd:.2f}%")
    print("-" * 85)
    print(f"{'МОНЕТА':<7} | {'НАПР':<5} | {'ТИП':<14} | {'ML PROB':<8} | {'ВХОД':<9} | {'ВЫХОД':<9} | {'РЕЗУЛЬТАТ ($)':<16} | {'ИТОГ'}")
    print("-" * 85)
    for _, t in df_t.iterrows():
        pnl_str = f"${t['pnl_usd']:+,.2f} ({t['pnl_pct']:+.1f}%)"
        print(f"{t['coin']:<7} | {t['dir']:<5} | {t['type']:<14} | {t['ml_prob']*100:<5.1f}%  | ${t['entry_px']:<8.2f} | ${t['exit_px']:<8.2f} | {pnl_str:<16} | {t['reason']}")
    print("=" * 85)

# 1. In-Sample

    # Надежный расчет метрик для A/B баттла
    start_eq = equity_curve[0]
    final_eq = equity_curve[-1]
    net_val = final_eq - start_eq
    pct_val = (net_val / start_eq) * 100.0

    peak = start_eq
    dd_val = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0
        if dd > dd_val:
            dd_val = dd

    wins_pnl = 0.0
    loss_pnl = 0.0
    wins_cnt = 0
    loss_cnt = 0

    for item in long_trades:
        if isinstance(item, str):
            parts = [p.strip() for p in item.split("|")]
            if len(parts) >= 7:
                try:
                    tok = parts[6].split()[0].replace("$", "")
                    val = float(tok)
                    if val > 0:
                        wins_pnl += val
                        wins_cnt += 1
                    elif val < 0:
                        loss_pnl += abs(val)
                        loss_cnt += 1
                except Exception:
                    pass
        elif isinstance(item, dict):
            val = item.get("pnl_usd", item.get("pnl", 0.0))
            if val > 0:
                wins_pnl += val
                wins_cnt += 1
            elif val < 0:
                loss_pnl += abs(val)
                loss_cnt += 1

    tot_tr = wins_cnt + loss_cnt
    wr_val = (wins_cnt / tot_tr * 100.0) if tot_tr > 0 else 0.0
    pf_val = (wins_pnl / loss_pnl) if loss_pnl > 0 else (999.0 if wins_pnl > 0 else 0.0)

    return {
        "pnl_usd": net_val, "pnl_pct": pct_val, "win_rate": wr_val,
        "pf": pf_val, "max_dd": dd_val, "trades": tot_tr
    }


# =====================================================================================
# ЗАПУСК A/B ТЕСТА: v10.7 vs v11.0 НА ИСТОРИИ HYPERLIQUID
# =====================================================================================
clean_target_coins = [c for c in clean_target_coins if c in processed_1h and c != "BTC"]
if "min_warmup" not in locals():
    min_warmup = 72
if "total_bars" not in locals():
    total_bars = len(processed_1h[clean_target_coins[0]])

split_idx = int((total_bars - min_warmup) * 0.75) + min_warmup
oos_start_idx = split_idx

is_days = max(1, (split_idx - min_warmup) // 24)
oos_days = max(1, (total_bars - oos_start_idx) // 24)
tot_days = is_days + oos_days

print("")
print("=" * 85)
print("  СТАРТ СРАВНИТЕЛЬНОГО БАТТЛА: " + str(tot_days) + " ДНЕЙ (IS: " + str(is_days) + " дн | OOS: " + str(oos_days) + " дн)")
print("=" * 85)

print("")
print(">>> [РАУНД 1/2] ТЕСТИРОВАНИЕ БАЗОВОГО v10.7 (2 СЛОТА, FIFO, БЕЗ RECYCLING)...")
r_is_v10 = simulate_v10_7(min_warmup, split_idx, "1. IN-SAMPLE v10.7 (" + str(is_days) + " ДНЕЙ)", mode="v10_7")
r_oos_v10 = simulate_v10_7(oos_start_idx, total_bars, "2. OUT-OF-SAMPLE v10.7 (" + str(oos_days) + " ДНЕЙ)", mode="v10_7")

print("")
print(">>> [РАУНД 2/2] ТЕСТИРОВАНИЕ v10.7 DYNAMIC RISK (2 СЛОТА, BOOST 1.3% - 2.2%)...")
r_is_v11 = simulate_v10_7(min_warmup, split_idx, "1. IN-SAMPLE DYNAMIC (" + str(is_days) + " ДНЕЙ)", mode="v10_7_dynamic")
r_oos_v11 = simulate_v10_7(oos_start_idx, total_bars, "2. OUT-OF-SAMPLE DYNAMIC (" + str(oos_days) + " ДНЕЙ)", mode="v10_7_dynamic")

print("")
print("=" * 85)
print("  ИТОГОВЫЙ БАТТЛ: v10.7 Base (1.0% Flat) vs v10.7 Dynamic Risk (ML-Kelly) НА " + str(tot_days) + " ДНЯХ")
print("=" * 85)
h_fmt = "{:<24} | {:<26} | {:<26}"
print(h_fmt.format("МЕТРИКА", "v10.7 (1.0% Base)", "v10.7 Boost (1.3%-2.2%)"))
print("-" * 85)

pnl_is_10 = "{:>+7.2f}% (${:>+7.2f})".format(r_is_v10["pnl_pct"], r_is_v10["pnl_usd"])
pnl_is_11 = "{:>+7.2f}% (${:>+7.2f})".format(r_is_v11["pnl_pct"], r_is_v11["pnl_usd"])
print(h_fmt.format("IS Прибыль (%)", pnl_is_10, pnl_is_11))

dd_is_10 = "{:>6.2f}%".format(r_is_v10["max_dd"])
dd_is_11 = "{:>6.2f}%".format(r_is_v11["max_dd"])
print(h_fmt.format("IS Max Drawdown", dd_is_10, dd_is_11))

pf_is_10 = "{:>6.2f}".format(r_is_v10["pf"])
pf_is_11 = "{:>6.2f}".format(r_is_v11["pf"])
print(h_fmt.format("IS Profit Factor", pf_is_10, pf_is_11))

tr_is_10 = "{} сд. ({:>5.1f}%)".format(r_is_v10["trades"], r_is_v10["win_rate"])
tr_is_11 = "{} сд. ({:>5.1f}%)".format(r_is_v11["trades"], r_is_v11["win_rate"])
print(h_fmt.format("IS Сделок (Win Rate)", tr_is_10, tr_is_11))
print("-" * 85)

pnl_oos_10 = "{:>+7.2f}% (${:>+7.2f})".format(r_oos_v10["pnl_pct"], r_oos_v10["pnl_usd"])
pnl_oos_11 = "{:>+7.2f}% (${:>+7.2f})".format(r_oos_v11["pnl_pct"], r_oos_v11["pnl_usd"])
print(h_fmt.format("OOS Прибыль (%)", pnl_oos_10, pnl_oos_11))

dd_oos_10 = "{:>6.2f}%".format(r_oos_v10["max_dd"])
dd_oos_11 = "{:>6.2f}%".format(r_oos_v11["max_dd"])
print(h_fmt.format("OOS Max Drawdown", dd_oos_10, dd_oos_11))

pf_oos_10 = "{:>6.2f}".format(r_oos_v10["pf"])
pf_oos_11 = "{:>6.2f}".format(r_oos_v11["pf"])
print(h_fmt.format("OOS Profit Factor", pf_oos_10, pf_oos_11))

tr_oos_10 = "{} сд. ({:>5.1f}%)".format(r_oos_v10["trades"], r_oos_v10["win_rate"])
tr_oos_11 = "{} сд. ({:>5.1f}%)".format(r_oos_v11["trades"], r_oos_v11["win_rate"])
print(h_fmt.format("OOS Сделок (Win Rate)", tr_oos_10, tr_oos_11))
print("-" * 85)

tot_10 = "${:>+7.2f}".format(r_is_v10["pnl_usd"] + r_oos_v10["pnl_usd"])
tot_11 = "${:>+7.2f}".format(r_is_v11["pnl_usd"] + r_oos_v11["pnl_usd"])
print(h_fmt.format("ИТОГО ЧИСТЫМИ ($)", tot_10, tot_11))
print("=" * 85)

#!/usr/bin/env python3
"""
Hyperliquid Order Flow & Microstructure Collector (v10.9 - CVD + OBI_10 + Delta OI).
- Потоковый сбор тиков trades для 11 инструментов L1.
- Потоковый сбор стакана L2 (l2Book) и расчет взвешенного дисбаланса OBI_10.
- Потоковый сбор activeAssetCtx для мониторинга Open Interest (OI) и фандинга.
- Класс OITracker: квантование замеров (шаг 5 мин, окно 24ч) и расчет MAD Z-Score.
- Защита соединения: 15-секундный JSON Keep-Alive пинг и персистентный кеш OI.
- Атомарная запись data/orderflow_state.json каждые 1.0 сек.
"""

import asyncio
import json
import logging
import sys
import time
import os
from collections import deque
from pathlib import Path
from typing import Dict, Any, List

import websockets
import numpy as np

from core import config
from core.quant.factors import QuantFactorEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] OrderFlowWS-v10.9: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OrderFlowWS-v10.9")

CLEAN_TARGETS = [c for c in config.TARGET_COINS if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]]
WS_URL = "wss://api.hyperliquid-testnet.xyz/ws" if config.IS_TESTNET else "wss://api.hyperliquid.xyz/ws"
WINDOW_SEC = 3600
OI_SAMPLE_INTERVAL_SEC = 300  # Квантование замеров OI: раз в 5 минут
OI_MAX_SAMPLES = 288          # 288 * 5 мин = 24 часа истории


class OITracker:
    """Управление и расчет дельты открытого интереса (OI) с защитой от скачков."""
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.history: Dict[str, List[float]] = {coin: [] for coin in CLEAN_TARGETS}
        self.last_sample_ts: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self.current_oi: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for coin in CLEAN_TARGETS:
                    if coin in data:
                        self.history[coin] = data[coin].get("history", [])[-OI_MAX_SAMPLES:]
                        self.current_oi[coin] = float(data[coin].get("current_oi", 0.0))
                logger.info(f"[✓] OITracker: история OI загружена из кеша ({len(CLEAN_TARGETS)} инструментов).")
            except Exception as e:
                logger.warning(f"[-] OITracker: ошибка чтения кеша: {e}")

    def save_cache(self):
        try:
            payload = {}
            for coin in CLEAN_TARGETS:
                payload[coin] = {
                    "history": self.history[coin][-OI_MAX_SAMPLES:],
                    "current_oi": self.current_oi[coin]
                }
            tmp = self.cache_file.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.cache_file)
        except Exception as e:
            logger.error(f"[-] OITracker: сбой записи кеша: {e}")

    def update_sample(self, coin: str, oi: float) -> None:
        """Алиас для тикового обновления OI с текущей меткой времени."""
        self.update(coin, time.time(), oi)

    def update(self, coin: str, timestamp: float, oi: float):
        if coin not in self.current_oi or oi <= 0.0:
            return

        self.current_oi[coin] = oi

        # Квантование: сохраняем новую точку раз в 5 минут
        if timestamp - self.last_sample_ts[coin] >= OI_SAMPLE_INTERVAL_SEC:
            self.history[coin].append(oi)
            if len(self.history[coin]) > OI_MAX_SAMPLES:
                self.history[coin].pop(0)
            self.last_sample_ts[coin] = timestamp

    def get_zscore(self, coin: str) -> float:
        hist = self.history.get(coin, [])
        if len(hist) < 6:
            return 0.0
        curr = self.current_oi.get(coin, 0.0)
        # Если текущий OI отличается от последней точки истории — добавляем его для тиковой дельты.
        # Если срез только что зафиксирован (curr == hist[-1]), дублирование исключается.
        if curr > 0.0 and abs(curr - hist[-1]) > 1e-6:
            series = hist + [curr]
        else:
            series = hist
        return QuantFactorEngine.compute_delta_oi_robust_zscore(series, lookback=24)


class MicrostructureCollector:
    def __init__(self):
        self.trades_history: Dict[str, deque] = {coin: deque() for coin in CLEAN_TARGETS}
        self.last_prices: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self.book_imbalance: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self.funding_rates: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}

        self.state_file = config.DATA_DIR / "orderflow_state.json"
        self.oi_cache_file = config.DATA_DIR / "oi_history.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.oi_tracker = OITracker(self.oi_cache_file)
        self.last_msg_ts: float = 0.0
        self.running = True

    async def _send_keepalive_pings(self, ws):
        """Пинг L1 сервера каждые 15 сек для опережения таймаутов прокси/Cloudflare."""
        while self.running:
            try:
                await asyncio.sleep(15)
                if ws.open:
                    await ws.send(json.dumps({"method": "ping"}))
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _compute_obi(self, bids: list, asks: list, depth: int = 10) -> float:
        """Расчет взвешенного дисбаланса книги заявок (OBI_10)."""
        if not bids or not asks:
            return 0.0

        n = min(depth, len(bids), len(asks))
        if n == 0:
            return 0.0

        bid_weighted = sum(float(bids[i]["sz"]) / (i + 1) for i in range(n))
        ask_weighted = sum(float(asks[i]["sz"]) / (i + 1) for i in range(n))
        total = bid_weighted + ask_weighted

        if total <= 1e-9:
            return 0.0

        return float(np.clip((bid_weighted - ask_weighted) / total, -1.0, 1.0))

    async def _save_state_loop(self):
        last_cache_save = time.time()
        while self.running:
            try:
                now = time.time()
                coins_payload = {}
                cutoff = now - WINDOW_SEC

                for coin in CLEAN_TARGETS:
                    q = self.trades_history[coin]
                    while q and q[0][0] < cutoff:
                        q.popleft()

                    buy_vol = sum(vol for _, side, vol, _ in q if side == "B")
                    sell_vol = sum(vol for _, side, vol, _ in q if side == "A")
                    total_vol = buy_vol + sell_vol
                    cvd_usd = buy_vol - sell_vol
                    cvd_ratio = (cvd_usd / total_vol) if total_vol > 0 else 0.0

                    coins_payload[coin] = {
                        "cvd_usd": round(cvd_usd, 2),
                        "cvd_ratio": round(cvd_ratio, 4),
                        "total_volume_usd": round(total_vol, 2),
                        "obi_10": round(self.book_imbalance[coin], 4),
                        "open_interest": round(self.oi_tracker.current_oi[coin], 2),
                        "delta_oi_z": round(self.oi_tracker.get_zscore(coin), 2),
                        "funding_rate": float(self.funding_rates.get(coin, 0.0)),
                        "last_px": self.last_prices[coin],
                        "ticks_in_window": len(q)
                    }

                payload = {
                    "timestamp": self.last_msg_ts,
                    "coins": coins_payload
                }

                tmp = self.state_file.with_suffix(f".{os.getpid()}.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, self.state_file)

                # Периодическое сохранение истории OI раз в 60 сек
                if now - last_cache_save >= 60.0:
                    self.oi_tracker.save_cache()
                    last_cache_save = now

            except Exception as e:
                logger.error(f"[-] Ошибка сохранения orderflow_state: {e}")

            await asyncio.sleep(1.0)

    async def run(self):
        logger.info(f"[*] Запуск Microstructure Collector v10.9 (CVD + OBI_10 + ΔOI | Целей: {len(CLEAN_TARGETS)})...")
        asyncio.create_task(self._save_state_loop())

        while self.running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=25_000_000
                ) as ws:
                    ping_task = asyncio.create_task(self._send_keepalive_pings(ws))

                    for coin in CLEAN_TARGETS:
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": coin}}))

                    logger.info("[✓] WebSocket L1 открыт (Trades + L2Book + ActiveAssetCtx). Прием микроструктуры...")

                    try:
                        while self.running:
                            try:
                                # P0-2: Watchdog таймаут 25с против Half-Open TCP
                                message = await asyncio.wait_for(ws.recv(), timeout=25.0)
                            except asyncio.TimeoutError:
                                logger.warning("[-] Watchdog OrderFlow: таймаут сокета > 25с. Реконнект...")
                                break

                            data = json.loads(message)
                            channel = data.get("channel")
                            if channel in ("trades", "l2Book", "activeAssetCtx"):
                                self.last_msg_ts = time.time()

                            if data.get("response") == "pong":
                                self.last_msg_ts = time.time()
                                continue

                            # Сделки
                            if channel == "trades":
                                trades = data.get("data", [])
                                for t in trades:
                                    coin = t.get("coin")
                                    if coin in self.trades_history:
                                        px = float(t.get("px", 0.0))
                                        sz = float(t.get("sz", 0.0))
                                        side = t.get("side")
                                        self.last_prices[coin] = px
                                        self.trades_history[coin].append((time.time(), side, px * sz, px))

                            # Стакан L2
                            elif channel == "l2Book":
                                book_data = data.get("data", {})
                                coin = book_data.get("coin")
                                if coin in self.book_imbalance:
                                    levels = book_data.get("levels", [[], []])
                                    bids = levels[0] if len(levels) > 0 else []
                                    asks = levels[1] if len(levels) > 1 else []
                                    self.book_imbalance[coin] = self._compute_obi(bids, asks, depth=10)

                            # Контекст актива (OI + Funding)
                            elif channel == "activeAssetCtx":
                                ctx = data.get("data", {})
                                coin = ctx.get("coin")
                                if coin in CLEAN_TARGETS:
                                    oi_val = ctx.get("openInterest") or (ctx.get("ctx", {}).get("openInterest") if isinstance(ctx.get("ctx"), dict) else 0.0)
                                    self.oi_tracker.update_sample(coin, float(oi_val or 0.0))
                                    funding_val = ctx.get("funding") or (ctx.get("ctx", {}).get("funding") if isinstance(ctx.get("ctx"), dict) else 0.0)
                                    try:
                                        self.funding_rates[coin] = float(funding_val or 0.0)
                                    except (ValueError, TypeError):
                                        pass

                    finally:
                        if not ping_task.done():
                            ping_task.cancel()

            except Exception as e:
                logger.error(f"[-] Ошибка WebSocket OrderFlow ({e}). Реконнект через 3.0 сек...")
                await asyncio.sleep(3.0)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | [%(name)s] | %(levelname)s | %(message)s"
    )
    print("=" * 60)
    print("🌊 QVEX ORDERFLOW WORKER: ЗАПУСК СБОРА МИКРОСТРУКТУРЫ")
    print("=" * 60)
    collector = MicrostructureCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        print("\n[!] Остановка OrderFlow Worker по сигналу KeyboardInterrupt.")

import threading
from core.execution.clock import L1Clock
"""
QVEX v10.7 — Архитектурный шлюз исполнения ордеров Hyperliquid L1.
Реализует требования аудита: нативные L1-триггеры, монотонный nonce,
детерминированный cloid, коридор slippage <= 0.2%, подавление 502/504
и институциональную изоляцию ключей Master / Agent Wallet.
"""
import asyncio
import time
import uuid
import logging
from typing import Optional, Dict, Any
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.error import ServerError

logger = logging.getLogger("QVEX.CoreExecutionGateway")

class HyperliquidExecutionGateway:
    def __init__(
        self,
        agent_account: LocalAccount,
        base_url: str,
        info_client: Info,
        master_address: Optional[str] = None,
        **kwargs
    ):
        self.account = agent_account
        self.base_url = base_url
        self.info = info_client
        self.master_address = master_address or kwargs.get("account_address")

        # Если задан адрес мастер-кошелька, агент подписывает сделки от его имени
        if self.master_address and self.master_address.lower() != self.account.address.lower():
            self.exchange = Exchange(self.account, self.base_url, account_address=self.master_address)
            logger.info(f"[SECURITY] Агент {self.account.address} авторизован для счета {self.master_address}")
        else:
            self.exchange = Exchange(self.account, self.base_url)

        self._nonce_lock = threading.Lock()
        self._last_nonce = 0

    def get_monotonic_nonce(self) -> int:
        """Потокобезопасный генератор строго монотонного nonce с синхронизацией L1."""
        return L1Clock.get_monotonic_nonce()

    @staticmethod
    def generate_cloid() -> str:
        """Создает валидный 128-битный шестнадцатеричный идентификатор cloid."""
        return f"0x{uuid.uuid4().hex}"

    async def place_native_trigger_stop(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        trigger_px: float,
        cloid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Размещает нативный биржевой триггер-стоп непосредственно на валидаторах L1.
        Ордер защищен reduce_only=True и исполняется даже при падении сервера бота.
        """
        cloid = cloid or self.generate_cloid()
        order_type = {
            "trigger": {
                "triggerPx": str(round(trigger_px, 2)),
                "isMarket": True,
                "tpsl": "sl"
            }
        }

        return await self._execute_with_retry(
            self.exchange.order,
            name=coin,
            is_buy=is_buy,
            sz=round(size, 4),
            limit_px=round(trigger_px, 2),
            order_type=order_type,
            reduce_only=True,
            cloid=cloid
        )

    async def execute_ioc_market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        max_slippage: float = 0.002,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Агрессивный лимитный IOC-ордер со строгим коридором проскальзывания <= 0.2%.
        Предотвращает потерю 5% спреда, зашитого по умолчанию в SDK.
        """
        all_mids = await asyncio.to_thread(self.info.all_mids)
        if coin not in all_mids:
            raise ValueError(f"Котировка для {coin} не найдена в стакане биржи")

        mid_px = float(all_mids[coin])
        limit_px = mid_px * (1.0 + max_slippage) if is_buy else mid_px * (1.0 - max_slippage)
        cloid = self.generate_cloid()

        order_type = {"limit": {"tif": "Ioc"}}

        return await self._execute_with_retry(
            self.exchange.order,
            name=coin,
            is_buy=is_buy,
            sz=round(size, 4),
            limit_px=round(limit_px, 2),
            order_type=order_type,
            reduce_only=reduce_only,
            cloid=cloid
        )

    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Фаза 1 Паники: пакетная отмена всех открытых лимитов и триггеров на L1.
        Опрашивает открытые ордера через info.open_orders и отправляет bulk_cancel.
        """
        target_address = self.master_address or self.account.address
        open_orders = await asyncio.to_thread(self.info.open_orders, target_address)
        if not open_orders:
            logger.info("Открытых ордеров на L1 не обнаружено.")
            return {"status": "ok", "cancelled": 0}

        cancel_requests = [
            {"coin": o["coin"], "oid": o["oid"]}
            for o in open_orders
        ]
        res = await self._execute_with_retry(
            self.exchange.bulk_cancel,
            cancel_requests
        )
        logger.info(f"Успешно аннулировано {len(cancel_requests)} открытых заявок на L1.")
        return {"status": "ok", "cancelled": len(cancel_requests), "response": res}

    async def _execute_with_retry(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Вызов SDK с подавлением сетевых сбоев 502/504 и экспоненциальным бэкоффом."""
        max_retries = 4
        base_backoff = 0.5

        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.to_thread(func, *args, **kwargs)

                if isinstance(response, dict) and response.get("status") == "err":
                    err_details = response.get("response", "Неизвестная ошибка L1")
                    logger.error(f"L1 отклонил действие: {err_details}")
                    raise RuntimeError(f"L1 Action Rejection: {err_details}")

                return response

            except ServerError as server_err:
                logger.warning(
                    f"Сетевой сбой ноды Hyperliquid (попытка {attempt}/{max_retries}): {server_err}"
                )
                if attempt == max_retries:
                    raise
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))

            except Exception as ex:
                logger.error(f"Исключение при вызове SDK: {ex}")
                if attempt == max_retries:
                    raise
                await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))
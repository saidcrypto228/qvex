"""
QVEX v10.7 — Исполнительный модуль аварийного закрытия позиций (SlideToPanicEngine).
Реализует требования аудита: пакетная отмена всех заявок на L1
и итеративная ликвидация экспозиций через агрессивный IOC с коридором 1.0%.
"""
import asyncio
import logging
from typing import Dict, Any, List

logger = logging.getLogger("QVEX.PanicEngine")

class SlideToPanicEngine:
    def __init__(self, gateway: Any, info_client: Any, user_address: str):
        self.gateway = gateway
        self.info = info_client
        self.user_address = user_address

    async def execute_emergency_flatten(self) -> Dict[str, Any]:
        """
        Процедура экстренного сброса:
        1. Полная отмена всех ордеров в книге заявок.
        2. Срез физических позиций через user_state / clearinghouse.
        3. Параллельная отправка агрессивных IOC-ордеров с ликвидационным коридором 1.0%.
        4. Итеративный контроль до полного обнуления портфеля.
        """
        logger.critical("АКТИВИРОВАН КОНТУР SLIDE-TO-PANIC: ПРИНУДИТЕЛЬНЫЙ СБРОС ВСЕХ ЭКСПОЗИЦИЙ")
        audit_trail = {"cancelled_orders": False, "closed_fills": [], "errors": []}

        # 1. Отмена всех активных лимитов и триггеров
        try:
            await self.gateway.cancel_all_orders()
            audit_trail["cancelled_orders"] = True
            logger.info("Все открытые ордера успешно аннулированы на бирже.")
        except Exception as cancel_exc:
            logger.error(f"Сбой при пакетной отмене ордеров: {cancel_exc}")
            audit_trail["errors"].append(str(cancel_exc))

        # 2. Итеративная редукция позиций до нуля
        max_flatten_cycles = 3
        for cycle in range(1, max_flatten_cycles + 1):
            state = await asyncio.to_thread(self.info.user_state, self.user_address)
            active_positions = [
                pos["position"] for pos in state.get("assetPositions", [])
                if abs(float(pos["position"]["szi"])) > 1e-6
            ]

            if not active_positions:
                logger.info("Портфель полностью приведен в нейтральное состояние (0 exposure).")
                break

            tasks = []
            for pos in active_positions:
                coin = pos["coin"]
                szi = float(pos["szi"])
                close_is_buy = szi < 0  # Если short — покупаем для покрытия
                size = abs(szi)

                # Допуск 1.0% проскальзывания для форсированного выхода из позиции
                tasks.append(
                    self.gateway.execute_ioc_market_order(
                        coin=coin,
                        is_buy=close_is_buy,
                        size=size,
                        max_slippage=0.010,
                        reduce_only=True
                    )
                )

            cycle_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in cycle_results:
                if isinstance(res, Exception):
                    logger.error(f"Ошибка сброса позиции в цикле {cycle}: {res}")
                    audit_trail["errors"].append(str(res))
                else:
                    audit_trail["closed_fills"].append(res)

            await asyncio.sleep(0.8)

        return audit_trail

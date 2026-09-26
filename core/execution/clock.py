import threading
"""
QVEX v10.7 — Проверка рассинхронизации системного времени (HyperBFT Drift Check).
Проверяет расхождение локальных часов относительно биржевых серверов Hyperliquid.
Лимит консенсуса: строго < 1000 мс.
"""
import time
import requests

def verify_hyperliquid_drift():
    print("=" * 60)
    print("⏱️ ПРОВЕРКА ДРЕЙФА СИСТЕМНОГО ВРЕМЕНИ ДЛЯ HYPERLIQUID L1")
    print("=" * 60)

    t_start = time.time()
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "meta"},
            timeout=5
        )
        t_recv = time.time()

        # Оценка сетевой задержки (Round-Trip Time)
        rtt_ms = (t_recv - t_start) * 1000

        # Серверный заголовок даты
        date_header = response.headers.get("Date")
        if not date_header:
            print("⚠️ Заголовок Date отсутствует в ответе ноды.")
            return

        from email.utils import parsedate_to_datetime
        server_dt = parsedate_to_datetime(date_header)
        server_ts = server_dt.timestamp()
        local_ts = t_recv

        drift_ms = abs(local_ts - server_ts) * 1000
        print(f"• Сетевая задержка (RTT): {rtt_ms:.1f} мс")
        print(f"• Расхождение времени:    {drift_ms:.1f} мс")

        if drift_ms < 1000:
            print("✔ [СТАТУС: В НОРМЕ] Дрейф часов соответствует консенсусу HyperBFT (< 1000 мс).")
        else:
            print("❌ [СТАТУС: РИСК] Расхождение превышает 1000 мс! Необходима синхронизация chrony.")

    except Exception as e:
        print(f"Ошибка проверки времени: {e}")
    print("=" * 60)

if __name__ == "__main__":
    verify_hyperliquid_drift()


class L1Clock:
    """
    Централизованный сервис монотонного времени L1 Hyperliquid.
    Компенсирует дрейф системных часов Windows и гарантирует строгую монотонность nonce.
    """
    _offset_ms: float = 0.0
    _last_nonce: int = 0
    _lock = threading.Lock()

    @classmethod
    def set_offset_ms(cls, offset_ms: float) -> None:
        with cls._lock:
            cls._offset_ms = offset_ms

    @classmethod
    def get_offset_ms(cls) -> float:
        with cls._lock:
            return cls._offset_ms

    @classmethod
    def get_time_ms(cls) -> int:
        """Возвращает текущее время в миллисекундах с поправкой на дрейф L1."""
        with cls._lock:
            return int((time.time() * 1000.0) + cls._offset_ms)

    @classmethod
    def get_monotonic_nonce(cls) -> int:
        """Потокобезопасный генератор строго монотонно возрастающего nonce."""
        with cls._lock:
            current_ms = int((time.time() * 1000.0) + cls._offset_ms)
            if current_ms <= cls._last_nonce:
                cls._last_nonce += 1
            else:
                cls._last_nonce = current_ms
            return cls._last_nonce

    @classmethod
    def sync_with_l1(cls, base_url: str = "https://api.hyperliquid.xyz") -> float:
        """
        Калибровка смещения системных часов относительно валидаторов Hyperliquid L1.
        """
        import requests
        from email.utils import parsedate_to_datetime

        t_start = time.time()
        try:
            resp = requests.post(f"{base_url}/info", json={"type": "meta"}, timeout=5)
            t_recv = time.time()
            date_hdr = resp.headers.get("Date")
            if date_hdr:
                server_dt = parsedate_to_datetime(date_hdr)
                server_ts = server_dt.timestamp()
                # Середина интервала RTT как оценка момента получения ответа сервером
                local_mid_ts = (t_start + t_recv) / 2.0
                offset_sec = server_ts - local_mid_ts
                cls.set_offset_ms(offset_sec * 1000.0)
                return cls.get_offset_ms()
        except Exception:
            pass
        return cls.get_offset_ms()

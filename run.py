"""
QVEX Institutional Trading Core — Главная точка входа.
Запуск: python run.py
"""
import sys
from pathlib import Path

# Гарантируем корректный PYTHONPATH для корня проекта
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    from core.engine import HyperliquidSwingBot
    bot = HyperliquidSwingBot()
    bot.start()

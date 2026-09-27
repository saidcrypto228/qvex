import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from core.quant.factors import QuantFactorEngine
from core.utils.helpers import atomic_write_json

logger = logging.getLogger("QVEX-Retrain")


class AdaptiveRetrainEngine:
    """
    Институциональный контур адаптивного обучения QVEX v10.7:
    - Triple-Barrier Purging + Embargo интервалы (де Прадо)
    - Защита от микровыборок (Min Sample Barrier)
    - Валидационный шлюз качества OOS AUC (Quality Gate)
    - Байесовское сглаживание весов (Bayesian Shrinkage)
    """

    MIN_SAMPLES = 80          # Минимальный размер выборки для переобучения
    MIN_AUC_THRESHOLD = 0.53  # Порог OOS AUC для допуска весов в бой
    SHRINKAGE_FACTOR = 0.35   # Вес новой модели при сглаживании с базовыми весами

    def __init__(self, target_r_multiple: float = 1.5, holding_horizon_bars: int = 6):
        self.r_mult = target_r_multiple
        self.horizon = holding_horizon_bars  # Удержание барьера (24ч = 6 баров 4H)
        self.embargo_bars = 3                # Защитный зазор между фолдами (12ч)
        self.model = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", random_state=42)
        self.factor_engine = QuantFactorEngine()

    def apply_purging_and_embargo(self, total_len: int, train_end_idx: int, test_start_idx: int):
        """Удаление пересекающихся баров (Purging) и зазор безопасности (Embargo)."""
        safe_train_end = max(0, train_end_idx - self.horizon)
        train_indices = list(range(0, safe_train_end))

        safe_test_start = test_start_idx + self.embargo_bars
        test_indices = list(range(safe_test_start, total_len))

        return train_indices, test_indices

    def validate_market_regime(self, btc_df: pd.DataFrame = None, evr_ratio: float = None) -> bool:
        """
        Проверка пригодности рыночного режима для переобучения.
        Блокирует обучение в моменты системного шока (EVR > 0.85).
        """
        if evr_ratio is not None and evr_ratio >= 0.85:
            logger.warning(f"⚠️ [REGIME-GATE] Рыночный шок: EVR {evr_ratio:.2f} >= 0.85. Переобучение заблокировано.")
            return False

        if btc_df is not None and len(btc_df) >= 50:
            btc_close = btc_df["close"].iloc[-1]
            btc_ema50 = btc_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
            if btc_close < btc_ema50 * 0.96:
                logger.warning("⚠️ [REGIME-GATE] Глубокий дамп BTC (>4% под EMA50). Свинг-модель заморожена.")
                return False

        return True

    def train_walk_forward(self, X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8,
                           prior_weights: list = None, btc_df: pd.DataFrame = None,
                           evr_ratio: float = None) -> dict:
        """Обучение с полной валидацией выборки, OOS и контролем рыночного режима."""
        n = len(X)

        # 1. Проверка минимального объема выборки
        if n < self.MIN_SAMPLES:
            logger.warning(f"⚠️ [SAMPLE-GATE] Недостаточно данных: {n} < {self.MIN_SAMPLES}. Переобучение пропущено.")
            return {"status": "REJECTED_INSUFFICIENT_SAMPLES", "auc": 0.5, "updated": False}

        # 2. Проверка рыночного режима (Regime Shift)
        if not self.validate_market_regime(btc_df=btc_df, evr_ratio=evr_ratio):
            return {"status": "REJECTED_UNSTABLE_REGIME", "auc": 0.5, "updated": False}

        # 3. Проверка баланса классов
        pos_ratio = y.mean()
        if pos_ratio < 0.15 or pos_ratio > 0.85:
            logger.warning(f"⚠️ [CLASS-GATE] Дисбаланс классов ({pos_ratio:.1%}). Переобучение небезопасно.")
            return {"status": "REJECTED_CLASS_IMBALANCE", "auc": 0.5, "updated": False}

        # 4. Разделение выборки с Purging и Embargo
        train_end = int(n * split_ratio)
        train_idx, test_idx = self.apply_purging_and_embargo(n, train_end, train_end)

        if len(train_idx) < 30 or len(test_idx) < 10:
            logger.warning("⚠️ [SPLIT-GATE] Слишком короткие фолды после Purging/Embargo.")
            return {"status": "REJECTED_TINY_FOLDS", "auc": 0.5, "updated": False}

        X_train_raw, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test_raw, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # 5. Изолированная нормализация (без Lookahead Bias)
        X_train_scaled = self.factor_engine.fit_transform(X_train_raw)
        X_test_scaled = self.factor_engine.transform(X_test_raw)

        # 6. Обучение
        self.model.fit(X_train_scaled, y_train)

        # 7. Контроль качества на Out-of-Sample
        preds = self.model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5

        if auc < self.MIN_AUC_THRESHOLD:
            logger.warning(f"❌ [QUALITY-GATE] Новая модель слабая: OOS AUC {auc:.3f} < {self.MIN_AUC_THRESHOLD}. Отклонено.")
            return {"status": "REJECTED_LOW_AUC", "auc": float(auc), "updated": False}

        new_weights = self.model.coef_[0].tolist()

        # 8. Байесовское сглаживание (Shrinkage), если переданы базовые веса
        if prior_weights and len(prior_weights) == len(new_weights):
            blended = [
                float(self.SHRINKAGE_FACTOR * w_new + (1.0 - self.SHRINKAGE_FACTOR) * w_old)
                for w_new, w_old in zip(new_weights, prior_weights)
            ]
            final_weights = blended
            logger.info("✔ [SHRINKAGE] Веса плавно сглажены с базовой априорной моделью.")
        else:
            final_weights = new_weights

        logger.info(f"✔ [ML-SUCCESS] Модель успешно обновлена! OOS AUC: {auc:.3f} (N={n})")

        import time
        scaler_mean = self.factor_engine.scaler.mean_.tolist() if hasattr(self.factor_engine.scaler, "mean_") else []
        scaler_scale = self.factor_engine.scaler.scale_.tolist() if hasattr(self.factor_engine.scaler, "scale_") else []

        return {
            "status": "ACCEPTED",
            "auc": float(auc),
            "auc_test": float(auc),
            "coef": final_weights,
            "weights": final_weights,
            "intercept": float(self.model.intercept_[0]),
            "feature_cols": list(X.columns),
            "feature_names": list(X.columns),
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "trained_at": time.time(),
            "updated": True
        }

    def save_model_weights(self, meta: dict, path: Path | str = "data/meta_model.json") -> bool:
        """Атомарный сброс весов только при успешном прохождении валидации."""
        if not meta.get("updated", False):
            logger.info("ℹ️ Пропуск записи на диск: статус модели не требует обновления.")
            return False

        atomic_write_json(path, meta)
        logger.info(f"✔ Защищенная модель атомарно сохранена: {path}")
        return True

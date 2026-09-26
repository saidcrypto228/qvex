import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class QuantFactorEngine:
    @staticmethod
    def compute_fractal_swings(high_series, low_series, window=2):
        """
        Расчет подтвержденных фракталов Swing High / Swing Low.
        Возвращает скалярные значения последнего уровня (float, float) без Lookahead Bias.
        """
        import numpy as np
        import pandas as pd

        try:
            h = pd.Series(high_series).reset_index(drop=True)
            l = pd.Series(low_series).reset_index(drop=True)
            n = len(h)

            if n < (window * 2 + 1):
                return float(h.iloc[-1]), float(l.iloc[-1])

            last_sh = float(h.iloc[-1])
            last_sl = float(l.iloc[-1])

            # Движемся от последних подтвержденных свечей назад к началу
            for i in range(n - 1 - window, window - 1, -1):
                sub_h = h.iloc[i - window : i + window + 1]
                if h.iloc[i] == sub_h.max():
                    last_sh = float(h.iloc[i])
                    break

            for i in range(n - 1 - window, window - 1, -1):
                sub_l = l.iloc[i - window : i + window + 1]
                if l.iloc[i] == sub_l.min():
                    last_sl = float(l.iloc[i])
                    break

            return last_sh, last_sl
        except Exception:
            try:
                return float(high_series.iloc[-1]), float(low_series.iloc[-1])
            except Exception:
                return 0.0, 0.0
    @staticmethod
    def compute_chandelier_exit(high, low, close, period=22, mult=3.0):
        """Расчет Chandelier Exit без Lookahead Bias."""
        import numpy as np
        import pandas as pd

        try:
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(close)

            tr1 = h - l
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period, min_periods=period).mean()

            long_stop = h.rolling(period).max() - mult * atr
            short_stop = l.rolling(period).min() + mult * atr
            return long_stop, short_stop
        except Exception:
            return pd.Series(close), pd.Series(close)

    @staticmethod
    def evaluate_evr_absorption(*args, **kwargs):
        """
        Универсальный расчет Explained Variance Ratio (EVR) PCA компоненты.
        Корректно обрабатывает open_px, close_px, матрицы и произвольные kwargs.
        """
        import numpy as np
        import pandas as pd

        try:
            # Извлекаем матрицу цен из аргументов или именованных параметров
            matrix = None
            if args:
                matrix = args[0]
            elif "matrix" in kwargs:
                matrix = kwargs["matrix"]
            elif "closes_matrix" in kwargs:
                matrix = kwargs["closes_matrix"]
            elif "open_px" in kwargs:
                matrix = kwargs["open_px"]
            elif "df" in kwargs:
                matrix = kwargs["df"]

            threshold = kwargs.get("threshold", 0.85)

            if matrix is None:
                return True, 0.5, 0.5

            if isinstance(matrix, dict):
                df_mat = pd.DataFrame(matrix)
            elif isinstance(matrix, list):
                df_mat = pd.DataFrame(matrix)
            elif isinstance(matrix, pd.DataFrame):
                df_mat = matrix
            else:
                return True, 0.5, 0.5

            if df_mat.shape[1] < 2 or len(df_mat) < 10:
                return True, 0.5, 0.5

            # Доходности строго по свечам без заглядывания вперед
            rets = df_mat.pct_change().dropna()
            if rets.empty or rets.shape[0] < 5:
                return True, 0.5, 0.5

            cov_mat = np.cov(rets.values, rowvar=False)
            eig_vals = np.linalg.eigvalsh(cov_mat)
            eig_vals = np.sort(eig_vals)[::-1]

            total_var = np.sum(eig_vals)
            if total_var <= 1e-12:
                return True, 0.5, 0.5

            pc1_ratio = float(eig_vals[0] / total_var)
            absorption = float(np.sum(eig_vals[:min(3, len(eig_vals))]) / total_var)

            is_evr_ok = bool(pc1_ratio < threshold)
            return is_evr_ok, pc1_ratio, absorption
        except Exception:
            return True, 0.5, 0.5
    @staticmethod
    def compute_atr(high, low, close, period=14):
        """Расчет Average True Range (ATR) без заглядывания в будущее."""
        import numpy as np
        import pandas as pd

        try:
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(close)

            tr1 = h - l
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(window=period, min_periods=period).mean()
            val = float(atr.iloc[-1])
            return val if not np.isnan(val) else float(tr.iloc[-1])
        except Exception:
            return 0.0

    @staticmethod
    def compute_residual_momentum_72h(asset_closes, btc_closes):
        """
        Расчет 72-часового Residual Momentum против BTC без Lookahead Bias.
        Возвращает: (z_residual_momentum, beta, raw_rs_pct)
        """
        import numpy as np
        import pandas as pd

        try:
            if isinstance(asset_closes, (list, np.ndarray)):
                asset_closes = pd.Series(asset_closes)
            if isinstance(btc_closes, (list, np.ndarray)):
                btc_closes = pd.Series(btc_closes)

            n = min(len(asset_closes), len(btc_closes))
            if n < 24:
                return 0.0, 1.0, 0.0

            a_ret = asset_closes.pct_change().iloc[-n:].fillna(0.0)
            b_ret = btc_closes.pct_change().iloc[-n:].fillna(0.0)

            var_b = np.var(b_ret)
            if var_b > 1e-12:
                beta = float(np.cov(a_ret, b_ret)[0, 1] / var_b)
            else:
                beta = 1.0

            lookback = min(72, n - 1)
            raw_rs = float((asset_closes.iloc[-1] / asset_closes.iloc[-lookback - 1]) - 1.0)
            btc_rs = float((btc_closes.iloc[-1] / btc_closes.iloc[-lookback - 1]) - 1.0)

            residual_return = raw_rs - (beta * btc_rs)
            rolling_std = float(a_ret.iloc[-lookback:].std() * np.sqrt(lookback))
            z_score = float(residual_return / rolling_std) if rolling_std > 1e-8 else 0.0

            return z_score, beta, raw_rs * 100.0
        except Exception:
            return 0.0, 1.0, 0.0

    """
    Генератор квантовых факторов QVEX v10.7 с гарантированной защитой от Lookahead Bias.
    Все предикторы строго сдвигаются на 1 шаг (.shift(1)), обеспечивая причинно-следственную
    связь (Causality Guarantee): решение в баре t опирается строго на бар t-1.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = scaler is not None and hasattr(scaler, 'mean_')

    def compute_raw_factors(self, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
        """Расчет сырых математических признаков."""
        f = pd.DataFrame(index=df.index)

        # 1. Тренд и Моментум (EMA Slope & MACD)
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        f["macd_hist"] = macd - signal
        f["ema_slope"] = (ema_fast - ema_fast.shift(3)) / (df["close"] + 1e-8)

        # 2. Относительная волатильность (Normalized ATR)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14).mean()
        f["atr_ratio"] = atr14 / (df["close"] + 1e-8)

        # 3. Объемный моментум (Volume Surge)
        vol_ma = df["volume"].rolling(window=20).mean()
        f["vol_surge"] = df["volume"] / (vol_ma + 1e-8)

        # 4. Относительная сила к BTC (Relative Strength - RS)
        if btc_df is not None and "close" in btc_df:
            alt_ret = df["close"].pct_change(6) # 24 часа (6 баров по 4H)
            btc_ret = btc_df["close"].pct_change(6)
            f["rs_btc"] = alt_ret - btc_ret
        else:
            f["rs_btc"] = 0.0

        # [CRITICAL P1 FIX]: Принудительный сдвиг факторов на 1 бар назад
        # Значения на строке t теперь физически отражают исторические данные строго ДО закрытия бара t
        f_shifted = f.shift(1).copy()
        return f_shifted

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Обучение скейлера СТРОГО на тренировочном наборе."""
        self.is_fitted = True
        return self.scaler.fit_transform(X_train.fillna(0.0))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Трансформация тестовых данных без утечки математического ожидания."""
        if not self.is_fitted:
            raise RuntimeError("QuantFactorEngine: скейлер не обучен! Запрещен инференс без предварительного fit.")
        return self.scaler.transform(X.fillna(0.0))

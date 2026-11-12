import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest


class TimeSeriesOutlierCleaner
    def __init__(
        self,
        method str = stl,          # stl, iqr, zscore, iforest
        period int  None = None,    # para datos con estacionalidad (mensual=12, semanal=52, etc.)
        window int = 12,             # para IQR  Z-score rolling
        iqr_mult float = 1.5,        # qué tan agresivo el IQR
        z_thresh float = 3.0,        # z  z_thresh = outlier
        contamination float = 0.01,  # para IsolationForest
        random_state int = 42,
    )
        self.method = method
        self.period = period
        self.window = window
        self.iqr_mult = iqr_mult
        self.z_thresh = z_thresh
        self.contamination = contamination
        self.random_state = random_state

        # atributos que se llenan en fit()
        self.outlier_idx_ = None
        self.resid_ = None
        self.stl_result_ = None
        self.iforest_model_ = None
        self.y_index_ = None

    # ------------ MÉTODOS INTERNOS DE DETECCIÓN ------------ #

    def _detect_stl(self, y pd.Series)
        if self.period is None
            raise ValueError(For 'stl' method you must provide 'period' (e.g., 12 for monthly data).)

        stl = STL(y, period=self.period, robust=True)
        result = stl.fit()
        resid = result.resid

        # regla 3σ
        threshold = 3  resid.std()
        outliers = resid[resid.abs()  threshold].index

        self.resid_ = resid
        self.stl_result_ = result
        return outliers

    def _detect_iqr(self, y pd.Series)
        outlier_idx = []

        for i in range(self.window, len(y))
            window_data = y.iloc[i-self.windowi]
            q1 = window_data.quantile(0.25)
            q3 = window_data.quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.iqr_mult  iqr
            high = q3 + self.iqr_mult  iqr

            if (y.iloc[i]  low) or (y.iloc[i]  high)
                outlier_idx.append(y.index[i])

        return pd.Index(outlier_idx)

    def _detect_zscore(self, y pd.Series)
        roll_mean = y.rolling(self.window, min_periods=self.window2).mean()
        roll_std = y.rolling(self.window, min_periods=self.window2).std()

        z = (y - roll_mean)  roll_std
        outliers = y[(z.abs()  self.z_thresh)].index
        return outliers

    def _detect_iforest(self, y pd.Series)
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        # reshape para sklearn
        y_2d = y.values.reshape(-1, 1)
        labels = model.fit_predict(y_2d)  # -1 = outlier

        self.iforest_model_ = model
        outliers = y.index[labels == -1]
        return outliers

    # ------------ API PRINCIPAL FIT  TRANSFORM ------------ #

    def fit(self, y pd.Series)
        
        Detecta outliers y guarda los índices internamente.
        y debe ser pd.Series (idealmente con DateTimeIndex).
        
        if not isinstance(y, pd.Series)
            raise TypeError(y must be a pandas Series.)

        self.y_index_ = y.index

        method = self.method.lower()
        if method == stl
            outliers = self._detect_stl(y)
        elif method == iqr
            outliers = self._detect_iqr(y)
        elif method == zscore
            outliers = self._detect_zscore(y)
        elif method == iforest
            outliers = self._detect_iforest(y)
        else
            raise ValueError(fUnknown method '{self.method}'. Use 'stl', 'iqr', 'zscore', or 'iforest'.)

        self.outlier_idx_ = pd.Index(outliers)
        return self

    def _impute_median_neighbors(self, y pd.Series, outliers pd.Index)
        y_clean = y.copy()

        for idx in outliers
            pos = y_clean.index.get_loc(idx)

            # vecinos anterior y siguiente (si existen)
            values = []
            if pos - 1 = 0
                values.append(y_clean.iloc[pos - 1])
            if pos + 1  len(y_clean)
                values.append(y_clean.iloc[pos + 1])

            if len(values) == 0
                # si no hay vecinos, no imputamos
                continue

            y_clean.loc[idx] = np.median(values)

        return y_clean

    def _impute_stl_reconstruct(self, y pd.Series, outliers pd.Index)
        
        Usa la descomposición STL para reemplazar outliers por
        trend + seasonal (sin residual).
        
        if self.stl_result_ is None
            raise RuntimeError(STL result is not available. Use method='stl' and call fit() first.)

        stl_fit = self.stl_result_
        recon = stl_fit.trend + stl_fit.seasonal  # sin residuo

        y_clean = y.copy()
        y_clean.loc[outliers] = recon.loc[outliers]
        return y_clean

    def transform(self, y pd.Series, strategy str = median_neighbors)
        
        Aplica la imputación sobre los outliers detectados.
        strategy
            - 'median_neighbors' mediana de vecinos (default)
            - 'stl_reconstruct'  usa trend+seasonal de STL
        
        if self.outlier_idx_ is None
            raise RuntimeError(You must call fit() before transform().)

        if not isinstance(y, pd.Series)
            raise TypeError(y must be a pandas Series.)

        if not y.index.equals(self.y_index_)
            # puedes relajar esto si quieres, pero es más seguro así
            raise ValueError(y index is not the same as the one used in fit().)

        if strategy == median_neighbors
            y_clean = self._impute_median_neighbors(y, self.outlier_idx_)
        elif strategy == stl_reconstruct
            y_clean = self._impute_stl_reconstruct(y, self.outlier_idx_)
        else
            raise ValueError(Unknown strategy. Use 'median_neighbors' or 'stl_reconstruct'.)

        return y_clean

    def fit_transform(self, y pd.Series, strategy str = median_neighbors)
        
        Atajo para fit() + transform().
        
        self.fit(y)
        return self.transform(y, strategy=strategy)

    # ------------ VISUALIZACIÓN ------------ #

    def plot(self, y pd.Series, y_clean pd.Series  None = None, figsize=(10, 5))
        
        Grafica la serie original y, opcionalmente, la limpia.
        También marca los outliers detectados.
        
        if self.outlier_idx_ is None
            raise RuntimeError(You must call fit() before plot().)

        plt.figure(figsize=figsize)
        plt.plot(y.index, y.values, label=Original, alpha=0.7)

        if y_clean is not None
            plt.plot(y_clean.index, y_clean.values, label=Clean, linewidth=2, alpha=0.9)

        # marcar outliers
        plt.scatter(
            self.outlier_idx_,
            y.loc[self.outlier_idx_],
            marker=o,
            edgecolor=red,
            facecolor=none,
            s=80,
            label=Outliers
        )

        plt.title(fTime Series with Outliers (method={self.method}))
        plt.xlabel(Time)
        plt.ylabel(Value)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

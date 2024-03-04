from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL, DecomposeResult
from tqdm import tqdm


def trend_determination(trend_comp: pd.Series, resid_comp: pd.Series) -> float:
    return max(0, 1 - np.var(resid_comp) / max(np.finfo(np.float32).eps, np.var(trend_comp + resid_comp)))


def trend_slope(trend_comp: pd.Series) -> float:
    slope = LinearRegression(fit_intercept=True).fit(np.arange(len(trend_comp)).reshape(-1, 1), trend_comp).coef_
    return slope[0] / np.clip(np.abs(np.mean(trend_comp)), a_min=1e-6, a_max=None)


def trend_linearity(trend_comp: pd.Series) -> float:
    model = LinearRegression(fit_intercept=True).fit(np.arange(len(trend_comp)).reshape(-1, 1), trend_comp)
    predictions = model.predict(np.arange(len(trend_comp)).reshape(-1, 1))
    residuals = trend_comp - predictions
    return max(0, 1 - np.var(residuals) / max(np.finfo(np.float32).eps, np.var(trend_comp)))


def seasonal_determination(seasonal_comp: pd.Series, resid_comp: pd.Series) -> float:
    return max(0, 1 - np.var(resid_comp) / max(np.finfo(np.float32).eps, np.var(seasonal_comp + resid_comp)))


def extract_trend(ts: pd.DataFrame, filter_size: int = 7) -> DecomposeResult:
    def moving_average(x, w):
        """Source: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
        """
        return np.convolve(x, np.ones(w), mode="valid") / w
    
    trend = moving_average(ts.values, filter_size)
    residual = ts[filter_size // 2:-(filter_size // 2)] - trend

    new_idx = ts.index[filter_size // 2:-(filter_size // 2)]  # remove points without a value from the moving average
    trend = pd.Series(index=new_idx, data=trend)
    seasonal = pd.Series(index=new_idx, data=np.zeros_like(trend))
    residual = pd.Series(index=new_idx, data=residual)
    decomp = DecomposeResult(ts, seasonal, trend, residual)
    
    return decomp


def decomps_and_features(data: List[pd.DataFrame], sp: int,
                         dataset_size: int = None) -> Tuple[List[DecomposeResult], np.ndarray]:
    if dataset_size is not None:
        data = data[:dataset_size]

    decomps = []
    features = np.empty((len(data), 4))
    for i, df in tqdm(enumerate(data)):
        if sp > 1:
            decomp = STL(df, period=sp).fit()
        else:
            decomp = extract_trend(df)

        decomps.append(decomp)

        features[i, 0] = trend_determination(decomp.trend, decomp.resid)
        features[i, 1] = trend_slope(decomp.trend)
        features[i, 2] = trend_linearity(decomp.trend)
        features[i, 3] = seasonal_determination(decomp.seasonal, decomp.resid) if sp > 1 else 0

    return decomps, features

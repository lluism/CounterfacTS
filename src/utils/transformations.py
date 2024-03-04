import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def manipulate_trend_component(trend_comp: pd.Series, f: float, g: float, h: float, m: float) -> pd.Series:
    """Manipulates a trend component as suggested by "Generating what-if scenarios for time series data".
    """
    model = LinearRegression(fit_intercept=True).fit(np.arange(len(trend_comp)).reshape(-1, 1), trend_comp)
    predictions = model.predict(np.arange(len(trend_comp)).reshape(-1, 1))
    residuals = trend_comp - predictions

    new_trend = model.intercept_ + f * (g * model.coef_ * np.arange(len(trend_comp)) + (1/h * residuals))
    # The additional trend is just a percentage increase from the intercept per time step
    additional_trend = m * model.intercept_ * np.arange(len(trend_comp))
    return new_trend + additional_trend


def manipulate_seasonal_determination(seasonal_comp: pd.Series, k: float) -> pd.Series:
    """Manipulates the seasonal component as suggested by "Generating what-if scenarios for time series data".
    """
    return k * seasonal_comp

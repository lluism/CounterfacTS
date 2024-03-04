from typing import List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def score_batch(y_test: np.ndarray,
                y_pred: np.ndarray,
                y_train: np.ndarray,
                sp: int) -> List[np.ndarray]:
    mape = horizon_mape(y_test, y_pred)
    smape = horizon_smape(y_test, y_pred)
    mase = horizon_mase(y_test, y_pred, y_train, sp=1)
    seasonal_mase = horizon_mase(y_test, y_pred, y_train, sp=sp)
    mse = horizon_mse(y_test, y_pred)
    mae = horizon_mae(y_test, y_pred)

    return mape, smape, mase, seasonal_mase, mse, mae


def create_score_df(scores: dict) -> DataFrame:
    df = pd.DataFrame()
    for k, v in scores.items():
        df[k] = v

    return df


def score_model(y_test: np.ndarray,
                y_pred: np.ndarray,
                y_train: np.ndarray,
                sp: int = 1,
                metrics: List[str] = ["MASE", "sMAPE", "MAPE", "seasonal_MASE", "MSE"]) -> dict:
    string_to_funcs = {
        "MASE": horizon_mase,
        "sMAPE": horizon_smape,
        "MAPE": horizon_mape,
        "seasonal_MASE": horizon_mase,
        "MSE": horizon_mse,
        "MAE": horizon_mae,
    }

    scores = {}
    for metric in metrics:
        horizon_func = string_to_funcs[metric]

        if metric == "MASE":
            horizon_loss = horizon_func(y_test, y_pred, y_train, 1)
        elif metric == "seasonal_MASE":
            horizon_loss = horizon_func(y_test, y_pred, y_train, sp)
        else:
            horizon_loss = horizon_func(y_test, y_pred)

        scores[metric] = horizon_loss

    return scores


def mean_mase(y_test: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, sp: int = 1) -> np.float32:
    """Calculates the average MASE score, ignoring nan values.
    """
    return np.nanmean(horizon_mase(y_test, y_pred, y_train, sp=sp))


def horizon_mase(y_test: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, sp: int = 1) -> np.ndarray:
    """MASE score at each individual horizon.
    """
    y_pred_naive = y_train[:, :-sp]
    mae_naive = np.mean(np.abs(y_train[:, sp:] - y_pred_naive), axis=1).reshape(-1, 1)

    # return the mean MASE and ignore points (time-series) where mae_naive is 0
    mae_naive_tiled = np.tile(mae_naive, [1, y_pred.shape[-1]])  # shape [batch_size, prediction_length]
    mask = mae_naive_tiled > 0

    denominator = mae_naive_tiled
    denominator[~mask] = 1  # avoid division by 0
    numerator = np.abs(y_test - y_pred)
    mase = numerator / denominator

    mase[~mask] = np.nan  # replace positions where denominator was 0 with undefined values
    return mase.reshape(y_test.shape)


def mean_mape(y_test: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """Calculates the average MAPE score, ignoring nan values.
    """
    return np.nanmean(horizon_mape(y_test, y_pred))


def horizon_mape(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE score at each individual horizon.
    """
    mask = np.abs(y_test) > 0

    denominator = np.abs(y_test)
    denominator[~mask] = 1  # avoid division by 0
    numerator = np.abs(y_test - y_pred)
    mape = numerator / denominator

    mape[~mask] = np.nan  # replace positions where denominator was 0 with undefined values
    return mape.reshape(y_test.shape)


def mean_smape(y_test: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """Calculates the average sMAPE score.
    """
    return np.mean(horizon_smape(y_test, y_pred))


def horizon_smape(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """sMAPE score at each individual horizon.
    """
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return 2.0 * nominator / denominator


def mean_mse(y_test: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """Calculates the average MSE score.
    """
    return np.mean(horizon_mse(y_test, y_pred))


def horizon_mse(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the MSE for each individual horizon.
    """
    return np.square(y_test - y_pred)


def mean_mae(y_test: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """Calculates the average MAE score.
    """
    return np.mean(horizon_mae(y_test, y_pred))


def horizon_mae(y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates the MAE for each individual horizon.
    """
    return np.abs(y_test - y_pred)

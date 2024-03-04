from typing import Union

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, PreText, Range1d
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from .subplot import Figure
from ..utils.evaluation import horizon_mase, horizon_mape, horizon_smape, horizon_mse


class MetricPlot(Figure):

    def __init__(self, horizon_scores: np.ndarray, sp: int, metric: str) -> None:
        super().__init__()
        self.horizon_scores = horizon_scores
        self.sp = sp
        self.metric = metric

        self.orig_ts = None
        self.orig_forecast = None
        self.orig_scores = np.full(self.horizon_scores.shape[1], np.nan)
        self.modified_ts = None
        self.modified_forecast = None
        self.modified_scores = np.full(self.horizon_scores.shape[1], np.nan)

        self.mean = np.around(np.nanmean(self.horizon_scores), 3)
        self.median = np.nanmedian(self.horizon_scores, axis=0)
        self.lower = np.nanquantile(self.horizon_scores, 0.25, axis=0)
        self.upper = np.nanquantile(self.horizon_scores, 0.75, axis=0)

        self.source = ColumnDataSource(data={
            "x": np.arange(1, self.horizon_scores.shape[1] + 1),
            "test": self.median,
            "test_lower": self.lower,
            "test_upper": self.upper,
            "original": self.orig_scores,
            "modified": self.modified_scores})

        self.df = self.source.to_df()

        self.fig = figure(title=self.metric.replace("_", " "), x_axis_label="horizon",
                          y_axis_label=self.metric.replace("_", " "), tools="pan, box_zoom, wheel_zoom, reset",
                          height=400, width=800)
        self.fig.y_range = Range1d(0, self.upper.max() + self.upper.max() * 0.1)

        self.fig.line("x", "test", source=self.source,
                      legend_label=f"median test score", color=Category10_10[1])
        self.fig.varea("x", "test_upper", "test_lower", legend_label="50% confidence interval", source=self.source,
                       color=Category10_10[1], fill_alpha=0.5)
        self.fig.line("x", "original", source=self.source, legend_label="original", color=Category10_10[2])
        self.fig.line("x", "modified", source=self.source,
                      legend_label="modified", color=Category10_10[3])

        self.fig.legend.background_fill_alpha = 0
        self.fig.legend.border_line_alpha = 0

        self.text = PreText(text=f"Scores:\nMean test: {self.mean}\nOriginal ts: {self.orig_scores.mean()}\n"
                                 f"Modified ts: {(self.modified_scores.mean())}")

    def _calc_metric_score(self, target: np.ndarray, forecast: np.ndarray, ts: np.ndarray) -> np.ndarray:
        target = target.reshape([1, -1])
        forecast = forecast.reshape([1, -1])
        ts = ts.reshape([1, -1, 1])

        if self.metric == "seasonal_MASE":
            score = horizon_mase(target, forecast, ts, self.sp)
        elif self.metric == "MASE":
            score = horizon_mase(target, forecast, ts, 1)
        elif self.metric == "MAPE":
            score = horizon_mape(target, forecast)
        elif self.metric == "sMAPE":
            score = horizon_smape(target, forecast)
        elif self.metric == "MSE":
            score = horizon_mse(target, forecast)

        return score.flatten()

    def set_active(self, index: int) -> None:
        super().set_active(index)
        self.orig_ts = None
        self.orig_forecast = None
        self.orig_scores = np.full(self.horizon_scores.shape[1], np.nan)
        self.modified_ts = None
        self.modified_forecast = None
        self.modified_scores = np.full(self.horizon_scores.shape[1], np.nan)

    def update_horizon_scores(self, metric: str, horizon_scores: np.ndarray) -> None:
        self.metric = metric
        self.horizon_scores = horizon_scores

        self.mean = np.around(np.nanmean(self.horizon_scores), 3)
        self.median = np.nanmedian(self.horizon_scores, axis=0)
        self.lower = np.nanquantile(self.horizon_scores, 0.25, axis=0)
        self.upper = np.nanquantile(self.horizon_scores, 0.75, axis=0)
        self.orig_scores = np.full(self.horizon_scores.shape[1], np.nan)

        self.fig.title.text = self.metric.replace("_", " ")
        self.fig.yaxis.axis_label = self.metric.replace("_", " ")
        self.fig.y_range.end = self.upper.max() + self.upper.max() * 0.1

    def update_source(self) -> None:
        if self.orig_ts is not None and np.isnan(self.orig_scores).sum() == len(self.orig_scores):
            target = self.orig_ts.values[-len(self.orig_forecast):]
            self.orig_scores = self._calc_metric_score(target, self.orig_forecast,
                                                       self.orig_ts.values[:-len(self.orig_forecast)])

        if self.modified_ts is not None:
            target = self.modified_ts.values[-len(self.modified_forecast):]
            self.modified_scores = self._calc_metric_score(target, self.modified_forecast,
                                                           self.modified_ts.values[:-len(self.modified_forecast)])

        self.source.data = dict(
            x=np.arange(1, self.horizon_scores.shape[1] + 1),
            test=self.median,
            test_lower=self.lower,
            test_upper=self.upper,
            original=self.orig_scores,
            modified=self.modified_scores
        )
        self.text.text = f"Mean scores:\nTest: {self.mean}\nOriginal ts: {np.around(self.orig_scores.mean(), 3)}\n"\
                         f"Modified ts: {np.around(self.modified_scores.mean(), 3)}"

    def update_mod_ts(self, modified_ts: pd.Series) -> None:
        self.modified_ts = modified_ts

    def update_forecast(self, forecast: Union[np.ndarray, None]) -> None:
        if forecast is not None and self.orig_forecast is None:
            forecast = forecast[:, 0]
            self.orig_forecast = forecast
        elif forecast is not None and self.orig_forecast is not None:
            forecast = forecast[:, 0]
            self.modified_forecast = forecast

    def reset(self) -> None:
        if self.active_index is None:
            return

        self.modified_ts = None
        self.modified_forecast = None
        self.modified_scores = np.full(self.horizon_scores.shape[1], np.nan)

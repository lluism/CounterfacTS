from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from bokeh.models import Legend
from bokeh.models.sources import ColumnDataSource
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from .subplot import Figure
from .utils import get_ts, get_decomp
from ..utils.transformations import (
    manipulate_trend_component,
    manipulate_seasonal_determination
)


class TSPlot(Figure):

    def __init__(self, test_data: List[pd.Series], len_train_data: int, config: dict) -> None:
        super().__init__()
        self.test_data = test_data
        self.len_train_data = len_train_data
        self.config = config

        self.active_index = None
        self.active_ts = None
        self.active_decomp = None
        self.modified_ts = None
        self.modified_decomp = None
        self.perturbations = None
        self.perturbed_points = None
        self.selected_points = None
        self.orig_forecast = None
        self.multiplicative_global_const = None
        self.additive_global_const = None
        self.additive_local_const = None

        nan_array = np.full(100, np.nan)
        self.source = ColumnDataSource(data={"x": nan_array, "orig": nan_array, "mod": nan_array})

        self.fig = figure(title="Select a time series", x_axis_label='time',
                          y_axis_label='observation', x_axis_type="datetime", height=400, width=1600,
                          tools="pan, box_zoom, wheel_zoom, reset, xbox_select")

        orig_line = self.fig.line("x", "orig", source=self.source, line_width=1, color=Category10_10[2],
                                  selection_color="orange", nonselection_alpha=1)
        mod_line = self.fig.line("x", "mod", source=self.source, line_width=1, color=Category10_10[3],
                                 nonselection_alpha=1)

        # selection does not work for lines, so we plot as circles with size=0
        self.fig.circle("x", "orig", source=self.source, size=0)

        self.forecast_source = ColumnDataSource(data={"x": nan_array, "orig_forecast": nan_array,
                                                      "orig_lower": nan_array, "orig_upper": nan_array,
                                                      "mod_forecast": nan_array, "mod_lower": nan_array,
                                                      "mod_upper": nan_array})

        orig_forecast_line = self.fig.line("x", "orig_forecast", source=self.forecast_source, color=Category10_10[6])
        orig_forecast_interval = self.fig.varea("x", "orig_upper", "orig_lower", source=self.forecast_source,
                                                color=Category10_10[6], fill_alpha=0.5)
        mod_forecast_line = self.fig.line("x", "mod_forecast", source=self.forecast_source, color=Category10_10[9])
        mod_forecast_interval = self.fig.varea("x", "mod_upper", "mod_lower", source=self.forecast_source,
                                               color=Category10_10[9], fill_alpha=0.5)

        self.perturbation_source = ColumnDataSource(data={"x": nan_array, "y": nan_array})
        perturbation_circles = self.fig.circle("x", "y", source=self.perturbation_source, color=Category10_10[3])

        self.legend = Legend(
            items=[
                ("original", [orig_line]),
                ("modified", [mod_line]),
                ("original forecast", [orig_forecast_line]),
                ("original 90% prediction interval", [orig_forecast_interval]),
                ("modified forecast", [mod_forecast_line]),
                ("modified 90% prediciton interval", [mod_forecast_interval]),
                ("perturbed points", [perturbation_circles])
            ], location="top_left", click_policy="hide", background_fill_alpha=0, border_line_alpha=0
        )
        self.fig.add_layout(self.legend, "center")

    def set_active(self, index: int) -> None:
        super().set_active(index)
        self.orig_forecast = None
        self.active_ts = get_ts(self.active_index, self.test_data, self.len_train_data, self.config)
        self.active_decomp = get_decomp(self.active_ts, self.config["sp"])

    def update_source(self) -> None:
        if self.active_index is None:
            self.fig.title.text = "Select a time series"
            self.source.data = dict(
                x=np.full(100, np.nan),
                orig=np.full(100, np.nan),
                mod=np.full(100, np.nan)
            )
            return

        ts = self.active_ts
        x = ts.index
        orig = ts.values
        mod = np.full(len(ts), np.nan)

        if self.modified_ts is not None:
            mod = self.modified_ts.values

        self.source.data = dict(x=x, orig=orig, mod=mod)
        self.update_perturbation_source()

        if self.active_index >= self.len_train_data:
            self.fig.title.text = f"Test time series {self.active_index - self.len_train_data}"
        else:
            self.fig.title.text = f"Train time series {self.active_index}"

    def update_perturbation_source(self) -> None:
        if self.modified_ts is None or self.perturbed_points is None:
            nan_array = np.full(100, np.nan)
            self.perturbation_source.data = dict(x=nan_array, y=nan_array)
            return

        perturbation_index = self.modified_ts.index[self.perturbed_points]
        perturbation_values = self.modified_ts.iloc[self.perturbed_points]
        self.perturbation_source.data = dict(x=perturbation_index, y=perturbation_values)

    def _build_modified_ts(self) -> pd.Series:
        if self.modified_decomp is None or self.modified_ts is None:
            decomp = self.active_decomp
            self.modified_decomp = decomp.trend + decomp.seasonal + decomp.resid

        self.modified_ts = deepcopy(self.modified_decomp)

        if self.perturbations is not None:
            self.modified_ts.iloc[self.perturbed_points] += self.perturbations

        if self.additive_global_const is not None:
            self.modified_ts = self.modified_ts + self.additive_global_const

        if self.multiplicative_global_const is not None:
            self.modified_ts *= self.multiplicative_global_const

        if self.additive_local_const is not None:
            mean = self.modified_ts.iloc[self.selected_points].median()
            values = self.modified_ts.iloc[self.selected_points] + mean * self.additive_local_const
            self.modified_ts.iloc[self.selected_points] = values

        return self.modified_ts

    def modify_global_decomp(self, f: int, g: int, h: int, m: int, k: int) -> pd.Series:
        decomp = self.active_decomp

        new_trend = manipulate_trend_component(decomp.trend, f, g, h, m)
        new_season = manipulate_seasonal_determination(decomp.seasonal, k)

        self.modified_decomp = new_trend + new_season + decomp.resid
        self.modified_ts = deepcopy(self.modified_decomp)

        return self._build_modified_ts()

    def modify_local_decomp(self, k: float) -> pd.Series:
        decomp = self.active_decomp

        if self.modified_decomp is None:
            self.modified_decomp = decomp.trend + decomp.seasonal + decomp.resid

        selected_season = manipulate_seasonal_determination(decomp.seasonal, k).iloc[self.selected_points]
        selected_trend = decomp.trend.iloc[self.selected_points]
        selected_resid = decomp.resid.iloc[self.selected_points]

        self.modified_decomp.iloc[self.selected_points] = selected_trend + selected_season + selected_resid
        self.modified_ts = deepcopy(self.modified_decomp)

        return self._build_modified_ts()

    def multiply_global_const(self, const: float) -> pd.Series:
        self.multiplicative_global_const = const
        return self._build_modified_ts()

    def add_global_const(self, const: float) -> pd.Series:
        self.additive_global_const = const
        return self._build_modified_ts()

    def add_local_const(self, const: float) -> Union[pd.Series, None]:
        if self.selected_points is None:
            return

        self.additive_local_const = const
        self.perturbed_points = self.selected_points
        return self._build_modified_ts()

    def _modify_perturbations(self, ts: pd.Series, legal_indexes: List[int], percentage: int,
                              strength: int) -> pd.Series:
        ts_vals = ts.values.flatten()

        num_points = int(len(legal_indexes) * (percentage / 100))
        start = legal_indexes[0]
        end = legal_indexes[-1]
        self.perturbed_points = np.random.randint(low=start, high=end, size=num_points)

        var = (np.abs(ts_vals[self.perturbed_points]) + 1)
        perturbations = np.random.normal(loc=0, scale=var, size=len(self.perturbed_points))
        clip_val = np.abs(ts_vals[self.perturbed_points]) * (strength / 100)

        self.perturbations = np.clip(perturbations, -clip_val, clip_val)
        return self._build_modified_ts()

    def modify_global_perturbations(self, percentage: int, strength: int) -> pd.Series:
        ts = get_ts(self.active_index, self.test_data, self.len_train_data, self.config)
        legal_indexes = [i for i in range(len(ts))]
        return self._modify_perturbations(ts, legal_indexes, percentage, strength)

    def modify_local_perturbations(self, percentage: int, strength: int) -> Union[pd.Series, None]:
        if self.selected_points is None:
            return

        ts = get_ts(self.active_index, self.test_data, self.len_train_data, self.config)
        legal_indexes = self.selected_points
        return self._modify_perturbations(ts, legal_indexes, percentage, strength)

    def reset_perturbations(self) -> pd.Series:
        self.perturbed_points = None
        self.perturbations = None
        self.modified_ts = self.modified_decomp
        return self.modified_ts

    def update_forecast(self, forecast: Union[np.ndarray, None]) -> None:
        def update_forecast_source(orig_forecast: np.ndarray, mod_forecast: np.ndarray, index: np.ndarray) -> None:
            if orig_forecast.shape[1] == 3:
                self.forecast_source.data = dict(x=index, orig_forecast=orig_forecast[:, 0],
                                                 orig_lower=orig_forecast[:, 1], orig_upper=orig_forecast[:, 2],
                                                 mod_forecast=mod_forecast[:, 0], mod_lower=mod_forecast[:, 1],
                                                 mod_upper=mod_forecast[:, 2])
            else:
                nan_array = np.full_like(orig_forecast[:, 0], np.nan)
                self.forecast_source.data = dict(x=index, orig_forecast=orig_forecast[:, 0], orig_lower=nan_array,
                                                 orig_upper=nan_array, mod_forecast=mod_forecast[:, 0],
                                                 mod_lower=nan_array, mod_upper=nan_array)

        if forecast is not None:
            index = get_ts(self.active_index, self.test_data, self.len_train_data, self.config).index[-len(forecast):]
            if self.orig_forecast is None or np.array_equal(self.orig_forecast, forecast):
                self.orig_forecast = forecast
                update_forecast_source(self.orig_forecast, np.full_like(self.orig_forecast, np.nan), index)
            else:
                update_forecast_source(self.orig_forecast, forecast, index)
        else:
            nan_array = np.full((100, 1), np.nan)
            update_forecast_source(nan_array, nan_array, nan_array)

    def reset(self) -> None:
        if self.active_index is None:
            return

        self.modified_ts = None
        self.modified_decomp = None
        self.perturbations = None
        self.perturbed_points = None
        self.selected_points = None
        self.orig_forecast = None
        self.multiplicative_global_const = None
        self.additive_global_const = None
        self.additive_local_const = None
        self.source.selected.indices = []
        self.update_perturbation_source()

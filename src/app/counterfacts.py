import os
from typing import List, Callable, Union

import numpy as np
import pandas as pd
import torch
from bokeh.events import ButtonClick, MenuItemClick
from bokeh.io import curdoc
from bokeh.models import Button, Slider, Div, Panel, Tabs, TextInput, Dropdown
from bokeh.layouts import column, row
from gluonts.dataset.common import ListDataset

from .hist_fig import HistPlot
from .metric_fig import MetricPlot
from .pca_fig import PCAPlot
from .ts_fig import TSPlot
from .utils import get_ts, get_prediction_dataloader, get_train_data_id
from ..utils.features import decomps_and_features
from ..models.utils import get_model
from ..utils.data_loading import load_features, load_score, load_test_data, load_metadata


class CounterfacTS:

    def __init__(self, config: dict) -> None:
        self.config = config
        self.active_index = None
        self.metrics = ["MASE", "MAPE", "sMAPE", "seasonal_MASE", "MSE"]

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model(config["model_name"])(**self.config["model_args"], device=device,
                               path=config["path"]).to(device)
        self.model.load_state_dict(torch.load(os.path.join(config["path"], "model.pth"), map_location=torch.device('cpu')))

        # Setup data and features
        self.freq, self.cardinality = load_metadata(self.config["dataset"])
        self.len_train_data = int(len(os.listdir(os.path.join(self.config["datadir"], "training_data"))) *
                                  config["trainer_args"]["batch_size"])
        self.test_data = load_test_data(self.config["dataset"],
                                        ts_length=config["context_length"] + config["prediction_length"])
        self.train_features = load_features(self.config["datadir"], train=True)
        self.test_features = load_features(self.config["datadir"], train=False)
        self.horizon_score = load_score(self.config["path"], "MASE")

        self._initialize_figure()

    def _initialize_figure(self) -> None:
        # Create sub app
        self.pca_plot = PCAPlot(self.train_features, self.test_features)
        self.hist_plot = HistPlot(self.train_features, self.test_features)
        self.ts_plot = TSPlot(self.test_data, self.len_train_data, self.config)
        self.metric_plot = MetricPlot(self.horizon_score, self.config["sp"], "MASE")

        # Add callbacks
        self.pca_plot.source.selected.on_change("indices", self.tapselect)
        self.ts_plot.source.selected.on_change("indices", self.recselect)

        # Create general selection widgets
        self.test_ts_selection_input = TextInput(value="", title="Select test time series by index")
        self.test_ts_selection_input.on_change("value", self.input_index_select)

        self.metric_selection = Dropdown(label="Metric selection", menu=self.metrics)
        self.metric_selection.on_click(self.change_metric)

        self.general_selections = column(self.test_ts_selection_input, self.metric_selection)

        # Global general transformations
        self.global_additive_const_slider = Slider(start=-10, end=10, value=0, step=0.1, title="Additive constant")
        self._throttle_sliders([self.global_additive_const_slider], self.add_global_const)

        self.global_additive_const_input = TextInput(value="", title="Input additive constant")
        self.global_additive_const_input.on_change("value", self.add_global_const_input)

        self.global_multiplicative_const_slider = Slider(start=-10, end=10, value=0, step=0.1,
                                                         title="Multiplicative constant")
        self._throttle_sliders([self.global_multiplicative_const_slider], self.multiply_global_const)

        self.global_multiplicative_const_input = TextInput(value="", title="Input multiplicative constant")
        self.global_multiplicative_const_input.on_change("value", self.multiply_global_const_input)

        self.general_modifiers = column(self.global_additive_const_slider, self.global_additive_const_input,
                                        self.global_multiplicative_const_slider, self.global_multiplicative_const_input)

        # Feature transformation sliders
        self.global_f_slider = Slider(start=0.01, end=10, value=1, step=0.01, title="Trend strength")
        self.global_h_slider = Slider(start=0.01, end=10, value=1, step=0.01, title="Trend linearity")
        self.global_m_slider = Slider(start=-1, end=1, value=0, step=0.01, title="Slope slider")
        self.global_k_slider = Slider(start=0.01, end=10, value=1, step=0.01, title="Seasonal strength")
        self._throttle_sliders([self.global_f_slider, self.global_h_slider, self.global_m_slider, self.global_k_slider],
                               self.mod_global_features)

        self.global_pert_percent_slider = Slider(start=0, end=100, value=0, step=1, title="% points perturbed")
        self.global_pert_str_slider = Slider(start=0, end=100, value=0, step=1, title="Perturbation strength")
        self._throttle_sliders([self.global_pert_percent_slider, self.global_pert_str_slider],
                               self.mod_global_perturbations)

        self.feature_modifiers = column(self.global_f_slider, self.global_h_slider, self.global_m_slider,
                                        self.global_k_slider, self.global_pert_percent_slider,
                                        self.global_pert_str_slider)

        # Local transformation sliders
        self.local_const_slider = Slider(start=-10, end=10, value=0, step=0.1, title="Add percentage of mean")
        self._throttle_sliders([self.local_const_slider], self.add_local_consts)

        self.local_k_slider = Slider(start=0.01, end=10, value=1, step=0.01, title="Seasonal strength")
        self._throttle_sliders([self.local_k_slider], self.mod_local_features)

        self.local_pert_percent_slider = Slider(start=0, end=100, value=0, step=1, title="% points perturbed")
        self.local_pert_str_slider = Slider(start=0, end=100, value=0, step=1, title="Perturbation strength")
        self._throttle_sliders([self.local_pert_percent_slider, self.local_pert_str_slider],
                               self.mod_local_perturbations)

        self.local_modifiers = column(self.local_const_slider, self.local_k_slider, self.local_pert_percent_slider,
                                      self.local_pert_str_slider)

        # Manipulation tabs
        self.options_panel = Panel(child=self.general_selections, title="General selections")
        self.general_panel = Panel(child=self.general_modifiers, title="General transformations")
        self.feature_panel = Panel(child=self.feature_modifiers, title="Feature transformations")
        self.local_panel = Panel(child=self.local_modifiers, title="Local transformations")
        self.manipulation_tabs = Tabs(tabs=[self.options_panel, self.general_panel, self.feature_panel,
                                            self.local_panel])

        # Reset button
        button = Button(label="Reset")
        button.on_click(self.reset)
        self.manipulation_column = column(self.manipulation_tabs, button)

        # Set up various tabs
        pca_hist_tab = Tabs(tabs=[Panel(child=self.pca_plot.fig, title="PCA"),
                                  Panel(child=self.hist_plot.fig, title="Histogram")])
        metrics_tab = Tabs(tabs=[Panel(child=self.metric_plot.fig, title="Horizon score")])
        info_tab = Tabs(tabs=[Panel(child=self.metric_plot.text, title="Metric scores"),
                              Panel(child=self.hist_plot.text, title="Feature values")])

        # ORGANIZE FIGURE
        title = row(Div(text=f"Dataset: {self.config['dataset']}, model: {self.config['model_name']}",
                        style={"font-size": "150%"}))
        plot_row1 = row(pca_hist_tab, metrics_tab, info_tab)
        plot_row2 = row(self.ts_plot.fig, self.manipulation_column)
        fig = column(title, plot_row1, plot_row2, sizing_mode="scale_both")

        curdoc().add_root(fig)

    @staticmethod
    def _throttle_sliders(sliders: List[Slider], function: Callable) -> None:
        for slider in sliders:
            slider.on_change("value_throttled", function)

    def set_active(self) -> None:
        self.metric_plot.set_active(self.active_index)
        self.hist_plot.set_active(self.active_index)
        self.pca_plot.set_active(self.active_index)
        self.ts_plot.set_active(self.active_index)

        self.metric_plot.orig_ts = get_ts(self.active_index, self.test_data, self.len_train_data, self.config)

    def update_forecast(self, modified_ts: pd.Series) -> None:
        if self.active_index is None:
            self.metric_plot.update_forecast(None)
            self.ts_plot.update_forecast(None)
            return

        if modified_ts is None:
            ts = get_ts(self.active_index, self.test_data, self.len_train_data, self.config)
        else:
            ts = modified_ts

        if self.active_index >= self.len_train_data:
            ts_id = self.active_index - self.len_train_data
            # if there are multiple windows per time series in the dataset
            # we have to loop until the id is less than the number of unique time series in the data set
            while ts_id >= self.cardinality:
                ts_id = ts_id - self.cardinality
        else:
            ts_id = get_train_data_id(self.active_index, self.config)

        data = ListDataset([{"start": ts.index[0], "target": ts.values, "feat_static_cat": np.array([ts_id])}],
                           freq=self.freq)
        dataloader = get_prediction_dataloader(data, self.config["context_length"], self.config["prediction_length"])
        batch = next(iter(dataloader))
        forecast = self.model.predict(batch)

        # models usually return [batch_size, prediction_length, 3], so we remove the batch dimension
        if len(forecast.shape) > 2:
            forecast = forecast.reshape([self.config["prediction_length"], -1])

        self.metric_plot.update_forecast(forecast)
        self.ts_plot.update_forecast(forecast)

    def update_subplots(self, modified_ts: pd.Series = None) -> None:
        self.update_forecast(modified_ts)
        self.metric_plot.update_source()
        self.hist_plot.update_source()
        self.pca_plot.update_source()
        self.ts_plot.update_source()

    def recselect(self, attr: str, old: List[int], new: List[int]) -> None:
        if len(new) == 0:
            self.ts_plot.selected_points = None
        else:
            self.ts_plot.selected_points = new

    def tapselect(self, attr: str, old: List[int], new: List[int]) -> None:
        if len(new) == 0:
            self.active_index = None
            self.reset(None, update_subplots=False)
            self.set_active()
            self.update_subplots()
            return
        else:
            self.pca_plot.source.selected.indices = [new[0]]  # multiple points might be selected so we choose the first
            if new[0] == self.active_index:
                return
            self.active_index = new[0]

        self.reset(None, update_subplots=False)
        self.set_active()
        self.update_subplots()

    def change_metric(self, event: MenuItemClick) -> None:
        metric = event.item
        if metric == self.metric_plot.metric:
            return

        self.horizon_score = load_score(self.config["path"], metric)
        self.metric_plot.update_horizon_scores(metric, self.horizon_score)
        self.metric_plot.update_source()

    def input_index_select(self, attr: str, old: str, new: str) -> None:
        try:
            index = int(new)
        except ValueError:
            index = None

        if index is not None and index >= len(self.test_data):
            index = None

        if index is None:
            return

        # Both train and test series are present when we select point from the pca plot, so offset by length of train
        # series to be consistent with that selection
        self.active_index = index + len(self.train_features)

        self.pca_plot.source.selected.indices = [self.active_index]
        self.reset(None, update_subplots=False)
        self.set_active()
        self.update_subplots()

    def _transform_and_replot(self, func: Callable, args: dict) -> None:
        modified_ts = func(**args)
        _, features = decomps_and_features([modified_ts], self.config["sp"])

        self.pca_plot.update_features(features)
        self.hist_plot.update_features(features)
        self.metric_plot.update_mod_ts(modified_ts)
        self.update_subplots(modified_ts=modified_ts)

    def mod_global_features(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        f = self.global_f_slider.value
        g = 1  # The m attribute is used instead
        h = self.global_h_slider.value
        m = self.global_m_slider.value / self.config["context_length"]  # set the m value to be a % change per season
        k = self.global_k_slider.value
        self._transform_and_replot(self.ts_plot.modify_global_decomp, dict(f=f, g=g, h=h, m=m, k=k))

    def mod_local_features(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        k = self.local_k_slider.value
        self._transform_and_replot(self.ts_plot.modify_local_decomp, dict(k=k))

    def mod_global_perturbations(self, attr: Union[str, None], old: Union[float, None],
                                 new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        percentage = self.global_pert_percent_slider.value
        strength = self.global_pert_str_slider.value
        self._transform_and_replot(self.ts_plot.modify_global_perturbations,
                                   dict(percentage=percentage, strength=strength))

    def mod_local_perturbations(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        percentage = self.local_pert_percent_slider.value
        strength = self.local_pert_str_slider.value
        self._transform_and_replot(self.ts_plot.modify_local_perturbations,
                                   dict(percentage=percentage, strength=strength))

    def multiply_global_const(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        const = self.global_multiplicative_const_slider.value
        self._transform_and_replot(self.ts_plot.multiply_global_const, dict(const=const))

    def multiply_global_const_input(self, attr: Union[str, None], old: Union[float, None],
                                    new: Union[float, None]) -> None:
        try:
            const = float(new)
        except ValueError:
            return

        self.global_multiplicative_const_slider.value = const
        self.multiply_global_const(None, None, None)

    def add_global_const(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        const = self.global_additive_const_slider.value
        self._transform_and_replot(self.ts_plot.add_global_const, dict(const=const))

    def add_global_const_input(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        try:
            const = float(new)
        except ValueError:
            return

        self.global_additive_const_slider.value = const
        self.add_global_const(None, None, None)

    def add_local_consts(self, attr: Union[str, None], old: Union[float, None], new: Union[float, None]) -> None:
        if self.active_index is None:
            return

        const = self.local_const_slider.value
        self._transform_and_replot(self.ts_plot.add_local_const, dict(const=const))

    def reset(self, event: Union[ButtonClick, None], update_subplots: bool = True) -> None:
        for slider in [self.global_f_slider, self.global_h_slider, self.global_k_slider, self.local_k_slider,
                       self.global_multiplicative_const_slider]:
            slider.value = 1

        for slider in [self.global_m_slider, self.global_pert_percent_slider, self.global_pert_str_slider,
                       self.local_const_slider, self.local_pert_percent_slider, self.local_pert_str_slider,
                       self.global_additive_const_slider]:
            slider.value = 0

        self.global_additive_const_input.value = ""
        self.global_multiplicative_const_input.value = ""

        self.hist_plot.reset()
        self.metric_plot.reset()
        self.pca_plot.reset()
        self.ts_plot.reset()

        if update_subplots:
            self.update_subplots(None)

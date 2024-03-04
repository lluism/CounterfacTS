from typing import List

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, PreText
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from .subplot import Figure


class HistPlot(Figure):
    FEATURE_NAMES = ["trend str", "trend slope", "trend linearity", "seasonal str"]

    def __init__(self, train_features: np.ndarray, test_features: np.ndarray) -> None:
        super().__init__()
        self.train_means = np.around(train_features.mean(axis=0), 3)
        self.train_stds = np.around(train_features.std(axis=0), 3)
        self.test_means = np.around(test_features.mean(axis=0), 3)
        self.test_stds = np.around(test_features.std(axis=0), 3)

        self.features = np.vstack([train_features, test_features])
        self.mod_features = None

        self.histogram_sources = []
        self.orig_pos_source = []
        self.plots = []
        for i, feature in enumerate(self.FEATURE_NAMES):
            hist, edges = np.histogram(train_features[:, i], density=True, bins="auto")
            source = ColumnDataSource(dict(top=hist, bottom=[0 for _ in range(len(hist))], left=edges[:-1],
                                           right=edges[1:], mid=(edges[:-1] + edges[1:]) / 2,
                                           circle_top=hist + hist.max() * 0.1))
            orig_pos_source = ColumnDataSource(dict(top=hist, bottom=[0 for _ in range(len(hist))], left=edges[:-1],
                                                    right=edges[1:], mid=(edges[:-1] + edges[1:]) / 2,
                                                    circle_top=hist + hist.max() * 0.1))

            plot = figure(title=feature, tools="reset, xbox_zoom")
            plot.quad(top="top", bottom="bottom", left="left", right="right", source=source, alpha=0.75,
                      color=Category10_10[0], selection_color=Category10_10[3], selection_alpha=1,
                      nonselection_alpha=0.75)
            # Plot histogram where only the selected index is visible. We use this to display the original position
            # in the histogram
            plot.quad(top="top", bottom="bottom", left="left", right="right", source=orig_pos_source, alpha=0,
                      color=Category10_10[2], selection_alpha=1, nonselection_alpha=0)

            # Plot circles above selected indices to easily display positions in the histogram with low bars
            plot.circle(x="mid", y="circle_top", source=source, size=5, alpha=0, selection_color=Category10_10[3],
                        selection_alpha=1, nonselection_alpha=0)
            plot.circle(x="mid", y="circle_top", source=orig_pos_source, size=5, alpha=0, color=Category10_10[2],
                        selection_alpha=1, nonselection_alpha=0)

            self.histogram_sources.append(source)
            self.orig_pos_source.append(orig_pos_source)
            self.plots.append(plot)

        self.fig = gridplot(self.plots, ncols=2, plot_height=200, plot_width=400, toolbar_location="right")

        self.feature_text = self.format_feature_summary()
        self.text = PreText(text=self.feature_text)

    def set_active(self, index: int) -> None:
        if index != self.active_index:
            self.mod_features = None

        self.active_index = index

    def format_feature_summary(self) -> str:

        def format_feature_stats(text, means, stds):
            for name, mean, std in zip(self.FEATURE_NAMES, means, stds):
                text += f"{name}: {mean} \u00B1 {std}\n"  # \u00B1 is the +- sign
            return text

        def format_single_ts_features(text, features):
            for name, value in zip(self.FEATURE_NAMES, features):
                text += f"{name}: {np.around(value, 3)}\n"
            return text

        text = "TRAIN FEATURES\n"
        text = format_feature_stats(text, self.train_means, self.train_stds)

        text += "\nTEST FEATURES\n"
        text = format_feature_stats(text, self.test_means, self.test_stds)

        text += "\nORIGNAL TS\n"
        features = np.full(4, np.nan)
        if self.active_index is not None:
            features = self.features[self.active_index]
        text = format_single_ts_features(text, features)

        text += "\nMODIFIED TS\n"
        features = np.full(4, np.nan)
        if self.mod_features is not None:
            features = self.mod_features
        text = format_single_ts_features(text, features)

        return text

    @staticmethod
    def _set_source_selected(features: np.ndarray, sources: List[ColumnDataSource]) -> None:
        for i, source in enumerate(sources):
            left = source.data["left"]
            idx = np.where(features[i] >= left)[0]

            if len(idx) == 0:
                idx = [0]

            source.selected.indices = [idx[-1]]

    @staticmethod
    def _reset_source_selected(sources: List[ColumnDataSource]) -> None:
        for source in sources:
            source.selected.indices = []

    def update_source(self) -> None:
        if self.active_index is None:
            self._reset_source_selected(self.histogram_sources)
            self._reset_source_selected(self.orig_pos_source)
            return

        self._set_source_selected(self.features[self.active_index], self.orig_pos_source)

        if self.mod_features is not None:
            self._set_source_selected(self.mod_features, self.histogram_sources)

        self.text.text = self.format_feature_summary()

    def update_features(self, features: np.ndarray) -> None:
        self.mod_features = features.flatten()

    def reset(self) -> None:
        self.mod_features = None
        self._reset_source_selected(self.histogram_sources)

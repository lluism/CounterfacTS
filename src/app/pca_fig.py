from copy import deepcopy
from typing import Union

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10_10
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .subplot import Figure


class PCAPlot(Figure):

    def __init__(self, train_features: np.ndarray, test_features: np.ndarray) -> None:
        super().__init__()
        self.train_features = train_features
        self.test_features = test_features

        self.active_index = None
        self.modified_pca = None

        self.scaler = StandardScaler()
        self.train_features = self.scaler.fit_transform(self.train_features)
        self.pca = PCA(n_components=2).fit(self.train_features)
        self.train_pca_data = self.pca.transform(self.train_features)

        self.test_features = self.scaler.transform(self.test_features)
        self.test_pca_data = self.pca.transform(self.test_features)

        source_array = np.vstack([self.train_pca_data, self.test_pca_data])
        index_array = np.concatenate([np.arange(len(self.train_pca_data)), np.arange(len(self.test_pca_data))])

        self.labels = ["train data"] * len(self.train_pca_data) + \
            ["test data"] * len(self.test_pca_data)

        self.source = ColumnDataSource(data={"comp1": source_array[:, 0], "comp2": source_array[:, 1],
                                             "ts_index": index_array, "label": self.labels})
        self.orig_point_source = ColumnDataSource(data={"comp1": np.full(1, np.nan), "comp2": np.full(1, np.nan)})

        tooltips = [
            ("index", "@label at index @ts_index"),
            ("x val", "@comp1"),
            ("y val", "@comp2"),
        ]

        self.fig = figure(x_axis_label="component 0", y_axis_label="component 1",
                          tools="pan, box_zoom, wheel_zoom, reset, tap", tooltips=tooltips, height=400, width=800)

        self.fig.circle("comp1", "comp2", source=self.source, selection_color=Category10_10[3], nonselection_alpha=0.5,
                        color=factor_cmap("label", Category10_10, ["train data", "test data"]), legend_field="label",)

        # invisible circle to set legend for the selected point
        self.fig.circle("comp1", "comp2", source=self.orig_point_source, color=Category10_10[2],
                        legend_label="original position")
        self.fig.circle("comp1", "comp2", source=self.orig_point_source, color=Category10_10[3], size=0,
                        legend_label="modified position")

        self.fig.legend.background_fill_alpha = 0
        self.fig.legend.border_line_alpha = 0

    def _get_original_pca(self) -> np.ndarray:
        pca_data = np.vstack([self.train_pca_data, self.test_pca_data])
        return pca_data[self.active_index]

    def update_source(self) -> None:
        source_array = deepcopy(np.vstack([self.train_pca_data, self.test_pca_data]))
        index_array = np.concatenate([np.arange(len(self.train_features)), np.arange(len(self.test_features))])
        orig_pos = [np.nan, np.nan]

        if self.active_index is not None and self.modified_pca is not None:
            orig_pos = deepcopy(source_array[self.active_index])
            source_array[self.active_index] = self.modified_pca

        self.source.data = dict(comp1=source_array[:, 0], comp2=source_array[:, 1],
                                ts_index=index_array, label=self.labels)
        self.orig_point_source.data = dict(comp1=[orig_pos[0]], comp2=[orig_pos[1]])

    def update_features(self, features: Union[np.ndarray, None]) -> None:
        if features is None:
            return

        scaled_features = self.scaler.transform(features)
        pca_data = self.pca.transform(scaled_features)

        self.modified_pca = pca_data

    def reset(self) -> None:
        if self.active_index is None:
            return

        self.modified_pca = None

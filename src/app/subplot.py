from abc import ABC

import pandas as pd


class Figure(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.active_index = None

    def set_active(self, index: int) -> None:
        self.active_index = index

    def update_source(self) -> None:
        raise NotImplementedError

    def update_mod_ts(self, modified_ts: pd.DataFrame) -> None:
        raise NotImplementedError

    def update_forecast(self, forecast: pd.DataFrame) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

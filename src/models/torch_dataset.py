import os

import numpy as np
from torch.utils.data import Dataset


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data"))


class LocalDataset(Dataset):
    def __init__(self, dataset: str, prefix: str) -> None:
        self.past_target = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_past_target.npy"))
        self.future_target = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_future_target.npy"))

        self.past_time_feat = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_past_time_feat.npy"))
        self.future_time_feat = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_future_time_feat.npy"))

        self.past_dynamic_age = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_past_dynamic_age.npy"))
        self.future_dynamic_age = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_future_dynamic_age.npy"))

        self.feat_static_cat = np.load(os.path.join(DATA_PATH, dataset, f"f{prefix}_feat_static_cat.npy"))

        self.len = self.past_target.shape[0]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> dict:
        return {
            "past_target": self.past_target[index],
            "future_target": self.future_target[index],
            "past_time_feat": self.past_time_feat[index],
            "future_time_feat": self.future_time_feat[index],
            "past_feat_dynamic_age": self.past_dynamic_age[index],
            "future_feat_dynamic_age": self.future_dynamic_age[index],
            "feat_static_cat": self.feat_static_cat[index]
        }


class TrainDataset(LocalDataset):
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset, "train")


class ValidationDataset(LocalDataset):
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset, "validation")


class TestDataset(LocalDataset):
    def __init__(self, dataset: str) -> None:
        super().__init__(dataset, "test")

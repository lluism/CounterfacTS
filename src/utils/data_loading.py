import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset
from tqdm import tqdm


def load_features(datadir: str, train: bool) -> np.ndarray:
    file_name = "train_features.npy" if train else "test_features.npy"
    return np.load(os.path.join(datadir, file_name))


def load_decomps(datadir: str, train: bool) -> List[pd.DataFrame]:
    file_name = "train_decomps.csv" if train else "test_decomps.csv"
    df = pd.read_csv(os.path.join(datadir, file_name), index_col=0)

    dfs = []
    print(f"Loading {'train' if train else 'test'} decomps")
    for i in tqdm(range(1, len(df.columns) + 1, 3)):
        decomp = df[df.columns[i:i + 3]].set_index(df[f"index{i // 3}"])
        decomp.index = pd.DatetimeIndex(decomp.index)
        decomp = decomp.drop([f"index{i // 3}"])
        dfs.append(decomp)

    return dfs


def load_metadata(dataset: str) -> Tuple[str, int]:
    metadata = get_dataset(dataset).metadata
    freq = metadata.freq
    cardinality = metadata.feat_static_cat[0].cardinality
    return freq, int(cardinality)


def load_train_data(datadir: str, batch_size: int) -> List[pd.Series]:
    print("Loading training data")
    training_path = os.path.join(datadir, "training_data")
    dfs = [0 for i in range(len(os.listdir(training_path)) * batch_size)]
    
    for f in tqdm(os.listdir(training_path)):
        batch_num = int(f.split(".")[0][5:])
        idx = batch_num * batch_size  # we need to keep track of the index that each column belongs to
        df = pd.read_csv(os.path.join(training_path, f), index_col=0, parse_dates=True)
        for i in range(0, len(df.columns), 3):
            ts = df[df.columns[i:i + 2]].set_index(df[f"index{i // 3}"])
            ts.index = pd.DatetimeIndex(ts.index, freq="infer")
            ts = ts.drop([f"index{i // 3}"], axis=1)

            dfs[idx] = ts.T.iloc[0]  # convert to series
            idx += 1

    return dfs


def load_test_data(dataset: str, ts_length: int) -> List[pd.Series]:
    dataset = get_dataset(dataset).test
    data = []
    print("Loading test data")
    for ts in tqdm(dataset):
        values = ts["target"][-ts_length:]
        index = pd.date_range(ts["start"], periods=len(values), freq=ts["start"].freq)
        ts = pd.Series(data=values, index=index)
        data.append(ts)  # convert to series

    return data


def load_score(datadir: str, metric: str) -> np.ndarray:
    return np.load(os.path.join(datadir, f"{metric.lower()}.npy"))

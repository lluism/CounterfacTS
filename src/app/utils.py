import os
from typing import List, Union

import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import ValidationDataLoader
from gluonts.time_feature import (
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    MonthOfYear
)
from gluonts.torch.batchify import batchify
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler
)
from statsmodels.tsa.seasonal import STL, seasonal_decompose


def get_ts(index: int, test_data: List[pd.Series], len_train_data: int, config: dict) -> Union[pd.Series, None]:
    if index is None:
        return None

    if index >= len_train_data:
        index = index - len_train_data
        ts = test_data[index]
    else:
        # find the epoch, batch and column number of the selected time series in the train data
        batch_size = config["trainer_args"]["batch_size"]
        batch = index // batch_size
        col_num = index % batch_size

        ts = pd.read_csv(os.path.join(config['datadir'], "training_data", f"batch{batch}.csv"))
        ts = ts[[f"index{col_num}", f"observation{col_num}"]]
        ts.index = pd.DatetimeIndex(ts[f"index{col_num}"])
        ts = ts.drop([f"index{col_num}"], axis=1)
        ts = pd.Series(data=ts.values.flatten(), index=ts.index)

    return ts


def get_train_data_id(index: int, config: dict) -> int:
    # find the epoch, batch and column number of the selected time series in the train data
    batch_size = config["trainer_args"]["batch_size"]
    batch = index // batch_size
    col_num = index % batch_size

    ts = pd.read_csv(os.path.join(config['datadir'], "training_data", f"batch{batch}.csv"))
    return ts[[f"id{col_num}"]].values[0][0]


def get_decomp(ts: Union[pd.Series, None], sp: int) -> Union[pd.Series, None]:
    if ts is None:
        return None

    if sp > 1:
        decomp = STL(ts, period=sp).fit()
    else:
        decomp = seasonal_decompose(ts, period=1)
    
    return decomp


def get_prediction_dataloader(dataset: ListDataset, context_length: int, prediction_length: int,
                               batch_size: int = 64) -> ValidationDataLoader:
    transformation = Chain([
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ),
        AddTimeFeatures(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,
            pred_length=prediction_length,
            time_features=[HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
        ),
        InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ValidationSplitSampler(min_future=prediction_length),
            past_length=context_length,
            future_length=prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
        )
    ])
    dataloader = ValidationDataLoader(
        dataset,
        batch_size=batch_size,
        stack_fn=batchify,
        transform=transformation
    )
    return dataloader

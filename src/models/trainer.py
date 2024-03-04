import logging
import os
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gluonts.env import env
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.repository.datasets import get_dataset
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
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from .torch_dataset import TrainDataset, ValidationDataset

env.max_idle_transforms = 1000
logger = logging.getLogger("trainer")


class Trainer:

    def __init__(self,
                 dataset: str,
                 epochs: int,
                 batch_size: int,
                 num_batches_per_epoch: int,
                 context_length: int,
                 prediction_length: int,
                 use_val_data: bool = True,
                 num_validation_windows: int = 1,
                 eval_every: int = 5,
                 patience: int = 10,
                 sp: int = 1,
                 allow_padded_sampling: bool = False,
                 num_batches_to_write: int = 1000,
                 data_source: str = "gluon") -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.use_val_data = use_val_data
        self.num_validation_windows = num_validation_windows
        self.eval_every = eval_every
        self.patience = patience
        self.sp = sp
        self.min_past = 0 if allow_padded_sampling else self.context_length
        self.batches_written = 0
        self.num_batches_to_write = num_batches_to_write
        self.data_source = data_source

        assert self.data_source == "local" or self.data_source == "gluon"
        if self.data_source == "local":
            self.train_dataloader, self.validation_dataloader = self._get_local_dataloaders(dataset)
        else:
            self.train_dataloader, self.validation_dataloader = self._get_gluon_dataloaders(dataset)

    def _get_local_dataloaders(self, dataset: str) -> Tuple[DataLoader, DataLoader]:
        train_dataset = TrainDataset(dataset)
        validation_dataset = ValidationDataset(dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      sampler=RandomSampler(train_dataset))
        validation_datalaoder = DataLoader(validation_dataset, batch_size=self.batch_size,
                                           sampler=RandomSampler(validation_dataset))

        return train_dataloader, validation_datalaoder

    def _get_gluon_dataloaders(self, dataset: str) -> Tuple[TrainDataLoader, ValidationDataLoader]:
        logger.info("Creating train, validation and test splits")
        dataset = get_dataset(dataset)
        train_data = ListDataset(list(iter(dataset.train)), freq=dataset.metadata.freq)

        if self.use_val_data:
            # Use the last n windows from each time series as validation data
            validation_data = []
            for i in range(self.num_validation_windows):
                for ts in train_data.list_data:
                    # only add time series long enough that we can remove one horizon and still have context_length +
                    # prediction_length values left
                    if len(ts["target"]) <= self.context_length + self.prediction_length * (i + 2):
                        continue

                    val_ts = deepcopy(ts)
                    val_ts["target"] = val_ts["target"][:-self.prediction_length * i if i > 0 else None]
                    validation_data.append(val_ts)

                    # slice off the validation data from the training data
                    if i == self.num_validation_windows - 1:
                        ts["target"] = ts["target"][:-self.prediction_length * (i + 1)]
            validation_data = ListDataset(validation_data, freq=dataset.metadata.freq)
            return self._get_train_dataloader(train_data), self._get_validation_dataloader(validation_data)
        else:
            return self._get_train_dataloader(train_data), None

    def _get_train_dataloader(self, dataset: ListDataset) -> TrainDataLoader:
        transformation = Chain([
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                pred_length=self.prediction_length,
                time_features=[HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(num_instances=1, min_past=self.min_past,
                                                            min_future=self.prediction_length),
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
            )
        ])
        dataloader = TrainDataLoader(
            dataset,
            batch_size=self.batch_size,
            stack_fn=batchify,
            transform=transformation,
            num_batches_per_epoch=self.num_batches_per_epoch
        )
        return dataloader

    def _get_validation_dataloader(self, dataset: ListDataset) -> ValidationDataLoader:
        transformation = Chain([
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                pred_length=self.prediction_length,
                time_features=[HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ValidationSplitSampler(min_future=self.prediction_length),
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
            )
        ])
        dataloader = ValidationDataLoader(
            dataset,
            batch_size=self.batch_size,
            stack_fn=batchify,
            transform=transformation,
        )
        return dataloader

    def save_batch(self, batch: dict, datadir: str) -> None:
        df = pd.DataFrame()
        for i, (forecast_start, context, target, id) in enumerate(zip(batch["forecast_start"],
                                                                      batch["past_target"],
                                                                      batch["future_target"],
                                                                      batch["feat_static_cat"])):
            ts_length = len(context) + len(target)
            start_time = pd.date_range(start=forecast_start, freq=-forecast_start.freq, periods=len(context) + 1)[-1]
            index = pd.date_range(start_time, freq=forecast_start.freq, periods=ts_length)
            values = torch.cat([context, target], dim=-1).numpy()
            df[f"index{i}"] = index
            df[f"observation{i}"] = values
            df[f"id{i}"] = np.repeat(id.numpy(), len(index))

        df.to_csv(os.path.join(datadir, f"batch{self.batches_written}.csv"))
        self.batches_written += 1

    def train(self, model: nn.Module, learning_rate: float, datadir: str, early_stopping: bool = False):
        logger.info(f"Starting training of {type(model).__name__}")

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience,
                                                               factor=0.5, min_lr=5e-5, verbose=True)

        best_mase = None
        model.train()
        for epoch_no in range(1, self.epochs + 1):
            epoch_start = time.time()
            sum_epoch_loss = 0
            for batch_no, batch in enumerate(self.train_dataloader, start=1):
                # calculate loss
                loss = model.calculate_loss(batch)
                if torch.isnan(torch.sum(loss)):
                    logger.critical(f"NaN loss value, epoch {epoch_no} batch {batch_no}")
                    raise ValueError(f"NaN loss value, epoch {epoch_no} batch {batch_no}")

                # step
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                sum_epoch_loss += loss.detach().cpu().numpy().item()
                if self.data_source == "gluon" and self.batches_written < self.num_batches_to_write:
                    self.save_batch(batch, datadir)

            if self.use_val_data and epoch_no % self.eval_every == 0:
                mases = []
                smapes = []
                mses = []
                model.eval()
                for batch in self.validation_dataloader:
                    mase, smape, mse = model.validate(batch, sp=self.sp)

                    mases.append(mase)
                    smapes.append(smape)
                    mses.append(mse)

                model.train()
                val_mase = np.mean(mases)
                val_smape = np.mean(smapes)
                val_mse = np.mean(mses)

                logger.info(f"Epoch {epoch_no}, time spent: {round(time.time() - epoch_start, 1)}, "
                            f"average training loss: {sum_epoch_loss / self.num_batches_per_epoch}, validation scores: "
                            f"[MASE: {val_mase}, MSE: {val_mse}, sMAPE: {val_smape}]")

                scheduler.step(val_mase)
                if early_stopping and (best_mase is None or val_mase < best_mase):
                    best_mase = val_mase
                    torch.save(model.state_dict(), os.path.join(datadir, "temp.pth"))
            else:
                logger.info(f"Epoch {epoch_no}, time spent: {round(time.time() - epoch_start, 1)}, "
                            f"average training loss: {sum_epoch_loss / self.num_batches_per_epoch}")

                if not self.use_val_data:
                    scheduler.step(sum_epoch_loss / self.num_batches_per_epoch)

            if optimizer.param_groups[0]["lr"] == scheduler.min_lrs[0] and scheduler.num_bad_epochs == self.patience:
                logger.info("Stopping training due to lack of improvement in training loss")
                break

        logger.info(f"Done training. Best validation MASE: {best_mase}")
        if self.use_val_data and early_stopping:
            model.load_state_dict(torch.load(os.path.join(datadir, "temp.pth")))
            os.remove(os.path.join(datadir, "temp.pth"))

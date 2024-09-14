import os
from typing import Tuple, List

import pandas as pd
from torch.utils.data import DataLoader

from run.config import InferenceConfig
from src.augmentation.augmentation import Augmentation
from src.dataset.dataset import LandmarkDetectionDataset, TrainDataset, ValidDataset, InferenceDataset

class TrainDataModule(object):

    def __init__(
        self, 
        cfg: InferenceConfig, 
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame
    ) -> None:
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df

    def prepare_loader(self) -> Tuple[DataLoader, DataLoader]:
        return self.train_dataloader(), self.valid_dataloader()

    def train_dataloader(self) -> DataLoader:
        train_dataset = TrainDataset(
            cfg=self.cfg,
            df=self.train_df,
            transform=Augmentation(self.cfg).transform_train()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=eval(self.cfg.trainer.num_workers),
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    
    def valid_dataloader(self) -> DataLoader:
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            df=self.valid_df,
            transform=Augmentation(self.cfg).transform_valid()
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=eval(self.cfg.trainer.num_workers),
            pin_memory=True,
            drop_last=True,
        )
        return valid_loader

class InferenceDataModule(object):

    def __init__(
        self, 
        cfg: InferenceConfig, 
        test_df: pd.DataFrame,
        study_ids: List[int]
    ) -> None:
        self.cfg = cfg
        self.test_df = test_df
        self.study_ids = study_ids

    def prepare_loader(self) -> DataLoader:
        return self.inference_dataloader()
    
    def prepare_detection_loader(self) -> DataLoader:
        return self.inference_detection_dataloader()

    def inference_dataloader(self) -> DataLoader:
        inference_dataset = InferenceDataset(
            cfg=self.cfg,
            df=self.test_df,
            study_ids=self.study_ids,
            transform=Augmentation(self.cfg).transform_valid()
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=eval(self.cfg.inferencer.num_workers),
            pin_memory=True,
            drop_last=True,
        )
        return inference_loader

    def inference_detection_dataloader(self) -> DataLoader:
        inference_dataset = LandmarkDetectionDataset(
            cfg=self.cfg,
            df=self.test_df,
            study_ids=self.study_ids,
            transform=Augmentation(self.cfg).transform_valid()
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=eval(self.cfg.inferencer.num_workers),
            pin_memory=True,
            drop_last=True,
        )
        return inference_loader
    
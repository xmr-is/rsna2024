import os
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader

from run.config import TrainConfig
from src.augmentation.augmentation import Augmentation
from src.dataset.dataset import TrainDataset, ValidDataset

class TrainDataModule(object):

    def __init__(
        self, 
        cfg: TrainConfig, 
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
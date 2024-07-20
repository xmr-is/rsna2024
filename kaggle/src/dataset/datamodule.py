import os
import pandas as pd
from torch.utils.data import DataLoader

from run.config import TrainConfig
from .dataset import TrainDataset, ValidDataset

class TrainDataModule(object):

    def __init__(self, cfg: TrainConfig, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df

    def train_dataloader(self) -> DataLoader:
        train_dataset = TrainDataset(
            cfg=self.cfg,
            df=self.df,
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
            df=self.df,
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
    
# class InferenceDataModule(object):

#     def __init__(self, cfg: TrainConfig, df) -> None:
#         self.cfg = cfg
#         self.df = df

#     def train_dataloader(self):
#         train_dataset = RSNA2024Dataset(
#             self.df,
#             self.transform,
#             self.phase
#         )
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.cfg.dataset.batch_size,
#             shuffle=True,
#             num_workers=self.cfg.dataset.num_workers,
#             pin_memory=True,
#             drop_last=True,
#         )
#         return train_loader
    
#     def valid_dataloader(self):
#         valid_dataset = RSNA2024Dataset(
#             self.df,
#             self.transform,
#             self.phase
#         )
#         valid_loader = DataLoader(
#             valid_dataset,
#             batch_size=self.cfg.dataset.batch_size,
#             shuffle=True,
#             num_workers=self.cfg.dataset.num_workers,
#             pin_memory=True,
#             drop_last=True,
#         )
#         return valid_loader
    

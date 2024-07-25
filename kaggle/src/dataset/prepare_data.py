import os
from pathlib import Path
from typing import Tuple
import pandas as pd

from run.config import PrepareDataConfig

class PrepareData(object):

    def __init__(self, cfg: PrepareDataConfig) -> None:
        self.cfg = cfg

    def separate_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = df[df['study_id'].isin(self.cfg.split.train_study_id)]
        valid_df = df[df['study_id'].isin(self.cfg.split.valid_study_id)]
        return train_df, valid_df

    def read_csv(self) -> pd.DataFrame:
        # #local
        # data_dir = Path(self.cfg.directory.input_dir)/'train.csv'
        #kaggle
        data_dir = Path(self.cfg.directory.base_dir)/'train.csv'
        df = pd.read_csv(data_dir)
        return self.preprocess_df(df)
    
    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(-100)
        df = df.replace(self.cfg.common.label2id)
        output_path = Path(self.cfg.directory.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return self.separate_df(df)
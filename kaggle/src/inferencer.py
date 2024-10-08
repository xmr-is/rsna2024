import os
import math
import sys
import glob
from typing import ClassVar, Any, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

import wandb
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run.config import InferenceConfig
from src.models.model import RSNA24Model
from src.dataset.datamodule import InferenceDataModule
from src.dataset.prepare_data import PrepareTestData
from src.utils.environment_helper import InferenceEnvironmentHelper

@dataclass
class Inferencer(object):
    cfg: InferenceConfig
    inference_dataloader: DataLoader

    def __post_init__(self):
        self.env = InferenceEnvironmentHelper(self.cfg)

    def predict(
            self, 
            models: List[nn.Module]
        ) -> Tuple[List[np.ndarray], List[str]]:
        predictions, row_names = self.inference(models)        
        return predictions, row_names

    def inference(
            self, 
            models: List[nn.Module]
        ) -> Tuple[List[np.ndarray], List[str]]:
        autocast = self.env.autocast()
        predictions = []
        row_names = []

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.inference_dataloader),
                total=len(self.inference_dataloader),
                desc='Inference',
                disable=False
            )
            for idx, (inputs, st_id) in pbar:         
                inputs = inputs.to(self.env.device())
                pred_per_study = np.zeros((25, 3))
                
                for condition in self.cfg.common.conditions:
                    for level in self.cfg.common.levels:
                        row_names.append(st_id[0] + '_' + condition + '_' + level)
                            
                with autocast:
                    for idx, model in enumerate(models):
                        print(f'--- Now Predicting fold {idx} ---')
                        outputs = model(inputs)[0]
                        for col in range(self.cfg.model.params.num_labels):
                            output = outputs[col*3:col*3+3]
                            pred = output.float().softmax(0).cpu().numpy()
                            pred_per_study[col] += pred / len(models)
                    predictions.append(pred_per_study)
        predictions = np.concatenate(predictions, axis=0)
                    
        return predictions, row_names
    
    def amsambles(self) -> List[nn.Module]:

        models = []
        # local
        # CKPT_PATHS = glob.glob(f'{self.cfg.directory.base_dir}/outputs/best_wll_model_fold-*.pt')
        # kaggle
        CKPT_PATHS = glob.glob(f'{self.cfg.directory.input_dir}/output/best_wll_model_fold-*.pt')
        CKPT_PATHS = sorted(CKPT_PATHS)

        for idx, cp in enumerate(CKPT_PATHS):
            print(f'loading {cp}...')
            model = RSNA24Model(
                cfg=self.cfg
            )
            model.load_state_dict(torch.load(cp))
            model.eval()
            #model.half()
            model.to(self.env.device())
            models.append(model)
        return models
            
    def make_submission(
            self, 
            predictions: List[np.ndarray], 
            row_names: List[str] 
        ) -> pd.DataFrame:
        prepare_data = PrepareTestData(self.cfg)
        labels = prepare_data.get_submission_labels()
        submission = pd.DataFrame()
        submission['row_id'] = row_names
        submission[labels] = predictions
        print(submission.head(25))
        return submission


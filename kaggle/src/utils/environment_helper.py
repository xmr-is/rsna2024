import os
import re
import random
from re import I
from typing import Any
import torch

import numpy as np

from run.config import TrainerConfig, InferencerConfig

class EnvironmentHelper(object):

    def __init__(
            self, 
            cfg: TrainerConfig, 
        ) -> None:
        self.cfg = cfg

    def set_random_seed(
        self, 
        seed: int, 
        deterministic: bool = False
    ) -> None:
        """Set seeds"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic

    def device(self) -> Any:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def autocast(self) -> Any:
        return torch.cuda.amp.autocast(
            enabled=self.cfg.trainer.use_amp, 
            dtype=torch.half
        )
    
    def scaler(self) -> Any:
        return torch.cuda.amp.GradScaler(
            enabled=self.cfg.trainer.use_amp, 
            init_scale=4096
        )
    
    
class InferenceEnvironmentHelper(object):

    def __init__(
            self, 
            cfg: InferencerConfig, 
        ) -> None:
        self.cfg = cfg

    def set_random_seed(
        self, 
        seed: int, 
        deterministic: bool = False
    ) -> None:
        """Set seeds"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic

    def device(self) -> Any:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def autocast(self) -> Any:
        return torch.cuda.amp.autocast(
            enabled=self.cfg.inferencer.use_amp, 
            dtype=torch.half
        )
    
    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]
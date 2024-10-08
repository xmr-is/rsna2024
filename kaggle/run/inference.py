import hydra
import os
import sys
from omegaconf import DictConfig
from omegaconf import OmegaConf

import pandas as pd
import yaml
import torch
import torch.nn as nn

from run.config import InferenceConfig
from src.inferencer import Inferencer
from src.models.model import RSNA24Model
from src.dataset.prepare_data import PrepareTestData
from src.dataset.datamodule import InferenceDataModule
from src.utils.environment_helper import EnvironmentHelper

@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: InferenceConfig):
    yaml_cfg = OmegaConf.to_yaml(cfg)
    
    env = EnvironmentHelper(cfg)
    env.set_random_seed(cfg.seed)

    prepare_data = PrepareTestData(cfg)
    test_df, study_ids = prepare_data.read_test_data()

    datamodule = InferenceDataModule(cfg, test_df, study_ids)
    inference_dataloader = datamodule.prepare_loader()
    
    inferencer = Inferencer(
        cfg=cfg,
        inference_dataloader=inference_dataloader,
    )
    models = inferencer.amsambles()

    predictions, row_names = inferencer.predict(models)
    
    submission = inferencer.make_submission(predictions, row_names)
    submission.to_csv(f'{cfg.directory.submission_dir}/submission.csv', index=False)

if __name__ == "__main__":
    main()
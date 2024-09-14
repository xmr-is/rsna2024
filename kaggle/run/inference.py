import hydra
import os
import sys
from omegaconf import DictConfig
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn

from run.config import InferenceConfig
from src.inferencer import Inferencer
from src.models.model import RSNA24DetectionModel, RSNA24Model
from src.dataset.prepare_data import PrepareTestData
from src.dataset.datamodule import InferenceDataModule
from src.utils.environment_helper import EnvironmentHelper
from src.utils.visualize_helper import visualize_prediction

@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: InferenceConfig):
    yaml_cfg = OmegaConf.to_yaml(cfg)
    #print(yaml_cfg)
    
    env = EnvironmentHelper(cfg)
    env.set_random_seed(cfg.seed)

    prepare_data = PrepareTestData(cfg)
    test_df, study_ids = prepare_data.read_test_data()

    datamodule = InferenceDataModule(cfg, test_df, study_ids)
    inference_detection_dataloader = datamodule.prepare_detection_loader()
    inference_dataloader = datamodule.prepare_loader()

    detection_model = RSNA24DetectionModel()

    inferencer = Inferencer(
        cfg=cfg,
        inference_dataloader=inference_detection_dataloader,
    )
    out_array = inferencer.landmark_detection(detection_model)
    
    inferencer2 = Inferencer(
        cfg=cfg,
        inference_dataloader=inference_dataloader,
    )
    models = inferencer2.amsambles()
    predictions, row_names = inferencer2.predict(models, out_array)
    
    submission = inferencer.make_submission(predictions, row_names)
    submission.to_csv(f'{cfg.directory.submission_dir}/submission.csv', index=False)

if __name__ == "__main__":
    main()
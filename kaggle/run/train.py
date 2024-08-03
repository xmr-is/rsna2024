import hydra
import os
import sys
from omegaconf import DictConfig
from omegaconf import OmegaConf

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from run.config import TrainConfig
from src.trainer import Trainer
from src.models.model import RSNA2024_ViT_HipOA, RSNA24Model
from src.dataset.prepare_data import PrepareData
from src.dataset.datamodule import TrainDataModule
from src.utils.wandb_helper import WandBHelper
from src.utils.environment_helper import EnvironmentHelper

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: TrainConfig) -> None:
    yaml_cfg = OmegaConf.to_yaml(cfg)
    print(yaml_cfg)
    
    env = EnvironmentHelper(cfg)
    env.set_random_seed(cfg.seed)

    prepare_data = PrepareData(cfg)
    train_df, valid_df = prepare_data.read_csv()

    datamodule = TrainDataModule(cfg, train_df, valid_df)
    train_dataloader, valid_dataloader = datamodule.prepare_loader()

    if cfg.model.name == 'vit_b_16':
        model = RSNA2024_ViT_HipOA(cfg).to(env.device())
    else:
        model = RSNA24Model(cfg).to(env.device())
    
    run = WandBHelper(cfg, model).wandb_config()

    optimizer=AdamW(
        model.parameters(), 
        lr=cfg.optimizer.lr, 
        weight_decay=cfg.scheduler.wd
    )

    warmup_steps = cfg.trainer.epochs/10 * len(train_dataloader) // cfg.trainer.grad_acc
    num_total_steps = cfg.trainer.epochs * len(train_dataloader) // cfg.trainer.grad_acc
    num_cycles = 0.475
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_total_steps,
        num_cycles=num_cycles
    )

    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(env.device()))
    criterion2 = nn.CrossEntropyLoss(weight=weights)

    trainer = Trainer(
        cfg=cfg,
        run=run,
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        criterion2=criterion2,
    )
    trainer.fit()
    #trainer.cv()

    run.finish()


if __name__ == "__main__":
    main()
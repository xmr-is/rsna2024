from typing import Any

import wandb
import torch.nn as nn

from run.config import TrainConfig

class WandBHelper(object):

    def __init__(self, cfg: TrainConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model

    def wandb_config(self) -> Any:
        run = wandb.init(
            mode = self.cfg.wandb,
            project = self.cfg.exp_name, 
            config = {
                #"env": self.cfg.directory,
                "model_name": self.cfg.model.name,
                "pretrained": self.cfg.model.params.pretrained,
                "in_channels": self.cfg.model.params.in_channels,
                "num_labels": self.cfg.model.params.num_labels,
                "num_classes": self.cfg.model.params.num_classes,
                "seed": self.cfg.seed,
                "debug": self.cfg.debug,
                "batch_size": self.cfg.batch_size,
                "image_size": self.cfg.dataset.image_size,
                "aug_prob": self.cfg.dataset.augmentation.apply_aug,
                "split": self.cfg.split.fold,
                "num_epochs": self.cfg.trainer.epochs,
                "early_stopping_epochs": self.cfg.trainer.early_stopping_epochs,
                "grad_acc": self.cfg.trainer.grad_acc,
                "optimizer_name": self.cfg.optimizer.name,
                "lr": self.cfg.optimizer.lr,
                "scheduler_name": self.cfg.scheduler.name,
                "wd": self.cfg.scheduler.wd,
                "criterion_name": self.cfg.criterion.name,
                "max_grad_norm": self.cfg.criterion.max_grad_norm,
            },
            entity = "xxmrkn",
            name = f"fold{self.cfg.split.fold}"
        )

        wandb.watch(self.model, log_freq=100)

        return run
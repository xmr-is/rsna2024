import os
import math
import sys
from typing import ClassVar
from collections import OrderedDict
from dataclasses import dataclass
import itertools
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



from run.config import TrainConfig
from src.utils.environment_helper import EnvironmentHelper

@dataclass
class Trainer(object):
    cfg: TrainConfig
    model: nn.Module
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    criterion: nn.modules.loss._Loss
    criterion2: nn.modules.loss._Loss
    best_loss: float = 1.2
    best_wll: float = 1.2
    es_step: int = 0 

    def fit(self) -> None:
        for epoch in range(1, self.cfg.trainer.epochs):
            self.train(epoch)
            self.valid(epoch)

    def train(self, epoch: int) -> torch.Tensor:
        self.model.train()
        total_loss = 0
        env = EnvironmentHelper(self.cfg)
        grad_scaler = env.scaler()
        autocast = env.autocast()
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc='Train',
            disable=True
        )
        for idx, (inputs, labels) in pbar:         
            inputs = inputs.to(env.device)
            labels  = labels.to(env.device)
            
            with autocast:
                loss = 0
                outputs = self.model(inputs)
                for col in range(self.cfg.model.params.num_labels):
                    pred = outputs[:,col*3:col*3+3]
                    gt = labels[:,col]
                    loss = loss + self.criterion(pred, gt) / self.cfg.model.params.num_labels

                total_loss += loss.item()
                if self.cfg.trainer.grad_acc > 1:
                    loss = loss / self.cfg.trainer.grad_acc

            if not math.isfinite(loss):
                print(f"Loss is {loss}, stopping training")
                sys.exit(1)

            pbar.set_postfix(
                OrderedDict(
                    loss=f'{loss.item()*self.cfg.trainer.grad_acc:.6f}',
                    lr=f'{self.optimizer.param_groups[0]["lr"]:.3e}'
                )
            )
            grad_scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), None or 1e9)

            if (idx + 1) % self.cfg.trainer.grad_acc == 0:
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
    
        train_loss = total_loss/len(self.train_dataloader)
        print(f'train_loss:{train_loss:.6f}') 
        return train_loss


    def valid(self, epoch: int) -> torch.Tensor:
        self.model.eval()
        env = EnvironmentHelper(self.cfg)
        autocast = env.autocast()
        total_loss = 0
        y_preds = []
        labels = []

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.valid_dataloader),
                total=len(self.valid_dataloader),
                desc='Valid',
                disable=True
            )
            for idx, (inputs, labels) in pbar:         
                inputs = inputs.to(env.device)
                labels  = labels.to(env.device)

                with autocast:
                    loss = 0
                    loss_ema = 0
                    outputs = self.model(inputs)
                    for col in range(self.cfg.model.params.num_labels):
                        pred = outputs[:,col*3:col*3+3]
                        gt = labels[:,col]

                        loss = loss + self.criterion(pred, gt) / self.cfg.model.params.num_labels
                        y_pred = pred.float()
                        y_preds.append(y_pred.cpu())
                        labels.append(gt.cpu())

                    total_loss += loss.item()   

        val_loss = total_loss/len(self.valid_dataloader)

        y_preds = torch.cat(y_preds, dim=0)
        labels = torch.cat(labels)
        val_wll = self.criterion2(y_preds, labels)

        print(f'val_loss:{val_loss:.6f}, val_wll:{val_wll:.6f}')

#             wandb.log({"train_loss": train_loss,
#                        "learning_rate": optimizer.param_groups[0]["lr"],
#                        "valid_loss": val_loss, 
#                        "valid_weighted_logloss": val_wll})

        if val_loss < self.best_loss or val_wll < self.best_wll:

            self.es_step = 0

            self.model.to(env.device)                

            if val_loss < self.best_loss:
                print(f'epoch:{epoch}, best loss updated from {self.best_loss:.6f} to {val_loss:.6f}')
                self.best_loss = val_loss

            if val_wll < self.best_wll:
                print(f'epoch:{epoch}, best wll_metric updated from {self.best_wll:.6f} to {val_wll:.6f}')
                self.best_wll = val_wll
                fname = f'{self.cfg.directory.output_dir}/best_wll_model_fold-{fold}.pt'
                torch.save(self.model.state_dict(), fname)

            self.model.to(env.device)

        else:
            self.es_step += 1
            if self.es_step >= self.cfg.trainer.early_stopping_epochs:
                print('early stopping')
            sys.exit(1)
                
                #wandb.finish()
        return self.best_wll 
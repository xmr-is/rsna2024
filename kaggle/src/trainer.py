import os
import math
import sys
from typing import ClassVar, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import itertools
import numpy as np
from sklearn.metrics import log_loss

import wandb
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from run.config import TrainConfig
from src.dataset.datamodule import TrainDataModule
from src.dataset.prepare_data import PrepareData
from src.utils.environment_helper import EnvironmentHelper

@dataclass
class Trainer(object):
    cfg: TrainConfig
    run: Any
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
        for epoch in range(1, self.cfg.trainer.epochs+1):
            print(f'----- fold ----- : {self.cfg.split.fold+1} / {self.cfg.num_folds}')
            print(f'----- epoch ----- : {epoch} / {self.cfg.trainer.epochs}')
            train_loss = self.train(epoch)
            valid_loss, valid_wll, stop_flag = self.valid(epoch)
            wandb.log({
                "Train Loss": train_loss, 
                "Leaning Rate": self.optimizer.param_groups[0]["lr"],
                "valid_loss": valid_loss, 
                "valid_weighted_logloss": valid_wll,
                "best_loss": self.best_loss, 
                "best_weighted_logloss": self.best_wll
            })
            if stop_flag:
                break 

    def cv(self) -> None:
        print(f'----- Calculate CV ----- : {self.cfg.split.fold+1}')
        y_preds, labels = self._valid()

        np.save(
            f'{self.cfg.directory.output_dir}/preds_fold{self.cfg.split.fold}.npy', 
            y_preds
        )
        np.save(
            f'{self.cfg.directory.output_dir}/labels_fold{self.cfg.split.fold}.npy', 
            labels
        )

    def train(self, epoch: int) -> None:
        self.model.train()
        total_loss = 0
        env = EnvironmentHelper(self.cfg)
        grad_scaler = env.scaler()
        autocast = env.autocast()
        pbar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc='Train',
            disable=False
        )
        for idx, (inputs, labels) in pbar:
            inputs = inputs.to(env.device())
            labels  = labels.to(env.device())
            
            with autocast:
                scs_loss = 0
                nfn_loss = 0
                ss_loss = 0
                loss = 0
                
                outputs = self.model(inputs)
                
                for col in range(5):
                    pred = outputs[0][:,col*3:col*3+3]
                    gt = labels[:,col]
                    scs_loss = scs_loss + self.criterion(pred, gt)
                for col in range(10):
                    pred = outputs[1][:,col*3:col*3+3]
                    gt = labels[:,col]
                    nfn_loss = nfn_loss + self.criterion(pred, gt)
                for col in range(10):
                    pred = outputs[2][:,col*3:col*3+3]
                    gt = labels[:,col]
                    ss_loss = ss_loss + self.criterion(pred, gt)

                loss = (scs_loss + nfn_loss + ss_loss) / self.cfg.model.params.num_labels
                total_loss  += loss.item()
                
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


    def valid(self, epoch: int) -> None:
        self.model.eval()
        env = EnvironmentHelper(self.cfg)
        autocast = env.autocast()
        total_loss = 0
        y_preds = []
        label = []
        stop_flag = False

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.valid_dataloader),
                total=len(self.valid_dataloader),
                desc='Valid',
                disable=False
            )
            for idx, (inputs, labels) in pbar:         
                inputs = inputs.to(env.device())
                labels  = labels.to(env.device())

                with autocast:
                    loss = 0
                    scs_loss = 0
                    nfn_loss = 0
                    ss_loss = 0
                    loss_ema = 0

                    outputs = self.model(inputs)
                    
                    for col in range(5):
                        pred = outputs[0][:,col*3:col*3+3]
                        gt = labels[:,col]
                        scs_loss = scs_loss + self.criterion(pred, gt)
                        y_pred_scs = pred.float()
                        y_preds.append(y_pred_scs.cpu())
                        label.append(gt.cpu())
                    for col in range(10):
                        pred = outputs[1][:,col*3:col*3+3]
                        gt = labels[:,col]
                        nfn_loss = nfn_loss + self.criterion(pred, gt)
                        y_pred_nfn = pred.float()
                        y_preds.append(y_pred_nfn.cpu())
                        label.append(gt.cpu())
                    for col in range(10):
                        pred = outputs[2][:,col*3:col*3+3]
                        gt = labels[:,col]
                        ss_loss = ss_loss + self.criterion(pred, gt)
                        y_pred_ss = pred.float()
                        y_preds.append(y_pred_ss.cpu())
                        label.append(gt.cpu())
                    
                    loss = (scs_loss + nfn_loss + ss_loss)/self.cfg.model.params.num_labels

                    total_loss += loss.item()

        val_loss = total_loss/len(self.valid_dataloader)

        y_preds = torch.cat(y_preds, dim=0)
        label = torch.cat(label)
        val_wll = self.criterion2(y_preds, label)

        print(f'val_loss:{val_loss:.6f}, val_wll:{val_wll:.6f}')

        if val_loss < self.best_loss or val_wll < self.best_wll:

            self.es_step = 0

            self.model.to(env.device())                

            if val_loss < self.best_loss:
                print(f'epoch:{epoch}, best loss updated from {self.best_loss:.6f} to {val_loss:.6f}')
                self.best_loss = val_loss

            if val_wll < self.best_wll:
                print(f'epoch:{epoch}, best wll_metric updated from {self.best_wll:.6f} to {val_wll:.6f}')
                self.best_wll = val_wll
                fname = f'{self.cfg.directory.output_dir}/best_wll_model_fold-{self.cfg.split.fold}.pt'
                torch.save(self.model.state_dict(), fname)

            self.model.to(env.device())

        else:
            self.es_step += 1
            if self.es_step >= self.cfg.trainer.early_stopping_epochs:
                print('early stopping')
                stop_flag = True
                return val_loss, val_wll, stop_flag

        return val_loss, val_wll, stop_flag
    
    def _valid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        env = EnvironmentHelper(self.cfg)
        autocast = env.autocast()
        
        fname = f'{self.cfg.directory.output_dir}/best_wll_model_fold-{self.cfg.split.fold}.pt'
        self.model.load_state_dict(
            torch.load(fname)
        )
        self.model.to(env.device())
        self.model.eval()
        
        y_preds = []
        label = []

        prepare_data = PrepareData(self.cfg)
        train_df, valid_df = prepare_data.read_csv()

        datamodule = TrainDataModule(self.cfg, train_df, valid_df)
        train_dataloader, valid_dataloader = datamodule.prepare_loader()
       
        with torch.no_grad():
            pbar = tqdm(
                enumerate(valid_dataloader),
                total=len(valid_dataloader),
                desc='Valid',
                disable=False
            )
            for idx, (inputs, labels) in pbar:
                inputs = inputs.to(env.device())
                labels = labels.to(env.device())
                    
                with autocast:
                    outputs = self.model(inputs)
                    for col in range(5):
                        pred = outputs[0][:,col*3:col*3+3]
                        gt = labels[:,col]
                        y_pred_scs = pred.float()
                        y_preds.append(y_pred_scs.cpu())
                        label.append(gt.cpu())
                    for col in range(10):
                        pred = outputs[1][:,col*3:col*3+3]
                        gt = labels[:,col]
                        y_pred_nfn = pred.float()
                        y_preds.append(y_pred_nfn.cpu())
                        label.append(gt.cpu())
                    for col in range(10):
                        pred = outputs[2][:,col*3:col*3+3]
                        gt = labels[:,col]
                        y_pred_ss = pred.float()
                        y_preds.append(y_pred_ss.cpu())
                        label.append(gt.cpu())

        return y_preds, label
from dataclasses import dataclass
import itertools
from typing import ClassVar, Any
from tqdm import tqdm
tqdm.pandas()
import torch
import torch.nn as nn

from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from run.config import TrainConfig
from src.utils.environment_helper import EnvironmentHelper

@dataclass(init=True)
class Trainer(object):

    cfg: TrainConfig
    model: None
    dataloader: None

    def _train(self):

        self.model.train()
        env = EnvironmentHelper(self.cfg)
        grad_scaler = env.scaler()
        autocast = env.autocast()
        
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.cfg.optimizer.lr, 
            weight_decay=self.cfg.scheduler.wd
        )

        warmup_steps = self.cfg.trainer.epochs/10 * len(self.dataloader) // self.cfg.trainer.grad_acc
        num_total_steps = self.cfg.trainer.epochs/10 * len(self.dataloader) // self.cfg.trainer.grad_acc
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_total_steps,
            num_cycles=num_cycles
        )

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(env.device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        best_loss = 1.2
        best_wll = 1.2
        es_step = 0 
        
        pbar = tqdm(
            enumerate(self.dataloader, start=1),
            total=len(self.dataloader),
            desc='Train',
            disable=True
        )

        for step, (inputs, labels) in pbar:         
            inputs = inputs.to(env.device)
            labels  = labels.to(env.device)
            
            if cfg.models.model.num_classes == 1:
                labels = labels.unsqueeze(1) 

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)

                if cfg.models.model.num_classes != 1:
                    f = torch.nn.Softmax(dim=1)
                    _outputs = f(outputs)
                    _, _outputs = torch.max(_outputs, 1)

                loss = self.criterion(outputs, 
                                      labels)

                running_loss += loss.item()
                train_loss = running_loss / step

            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss = f'{train_loss:0.6f}',
                             lr = f'{current_lr:0.6f}',
                             gpu_mem = f'{mem:0.2f} GB')
            
        self.scheduler.step()     
        
        return train_loss


    def _valid(self, cfg):

        if cfg.models.model.mcdropout:
            # turn ON dropout
            self.model.train()
        else:
            # turn OFF dropout
            self.model.eval()

        with torch.no_grad():
            
            dataset_size = 0
            running_loss = 0.0
            
            pbar = tqdm(enumerate(self.dataloader, start=1),
                        total=len(self.dataloader),
                        desc='Valid',
                        disable=True)

            for step, (inputs, labels, image_path, image_id) in pbar:        
                inputs  = inputs.to(ConfigurationHelper.device)
                labels  = labels.to(ConfigurationHelper.device)

                if cfg.models.model.num_classes == 1:
                    labels = labels.unsqueeze(1)

                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                
                if cfg.models.model.num_classes != 1:
                    f = torch.nn.Softmax(dim=1)
                    _outputs = f(outputs)
                    _, _outputs = torch.max(_outputs, 1)
               
                loss = self.criterion(outputs,
                                      labels)
                
                running_loss += loss.item()
                dataset_size += batch_size

                valid_loss = running_loss / step

                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(valid_loss = f'{valid_loss:0.6f}',
                                 lr = f'{current_lr:0.6f}',
                                 gpu_memory = f'{mem:0.2f} GB')
            
        return valid_loss

@dataclass
class Test(object):
    
    model: None
    fold: int
    num_sampling: int
    test_loader: None

    def test_one_epoch(self, cfg):

        if cfg.models.model.mcdropout:
            self.model.train()
        else:
            self.model.eval()
            
        with torch.no_grad():
                
            pbar = tqdm(enumerate(self.test_loader, start=1),
                        total=len(self.test_loader), 
                        desc='Test',
                        disable=True)

            for step, (inputs, labels, image_path, image_id) in pbar:                              
                inputs = inputs.to(ConfigurationHelper.device)
                labels = labels.tolist()
                
                # Predict
                outputs = self.model(inputs)

                # Classification
                if cfg.models.model.num_classes != 1:
                    f = torch.nn.Softmax(dim=1)
                    outputs = f(outputs)
                    _, _outputs = torch.max(outputs, 1)

                    ConfigurationHelper.predicts_array[self.num_sampling].extend(outputs.tolist())
                
                # Regression
                else:
                    logit = outputs.tolist()
                    
                    outputs = EvaluationHelper.threshold_config(outputs)
                    outputs = list(itertools.chain.from_iterable(outputs))
                    logit = list(itertools.chain.from_iterable(logit))
                    
                    ConfigurationHelper.predicts_array[self.num_sampling].extend(outputs)
                    ConfigurationHelper.predicts_array_float[self.num_sampling].extend(logit)

                torch.cuda.empty_cache()

                if self.num_sampling == cfg.models.inference.num_sampling-1:
                    ConfigurationHelper.ground_truth.extend(labels)
                    ConfigurationHelper.path_list.extend(image_id)
                    ConfigurationHelper.fold_id.extend([self.fold+1]*len(labels))
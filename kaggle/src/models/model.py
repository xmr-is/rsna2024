import torch
import torch.nn as nn
import timm

from run.config import ModelConfig

class RSNA24Model(nn.Module):
    def __init__(self, cfg:ModelConfig) -> None:
        # 3(Axial,sagittalT1,sagittalT2) x 
        # 5(l1-2,l2-3,l3-4,l4-5,l5-s1) x
        # 5(spinal_canal_stenosis,{left,right_}neural_foraminal_narrowing, {left,right_}subarticular_stenosis)
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            model_name=self.cfg.model.name,
            pretrained=self.cfg.model.params.pretrained, 
            features_only=False,
            in_chans=self.cfg.model.params.in_channels,
            num_classes=self.cfg.model.params.num_classes,
            global_pool='avg'
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x
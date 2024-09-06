import torch
import copy
import torch.nn as nn
from typing import Tuple
import timm

from torchvision import models

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
            #num_classes=10,
            global_pool='avg',
        )
        # self.load_weight()
        # self.init_layer()
    
    def load_weight(self):
        self.model.load_state_dict(
            torch.load(
                "/kaggle/input/spine-landmark-swin/swin_large_patch4_window12_384.ms_in22k_0.pt",
                map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        )

    def init_layer(self):
        self.model.patch_embed.proj = nn.Conv2d(
            in_channels=self.cfg.model.params.in_channels, 
            out_channels=192, 
            kernel_size=(4, 4), 
            stride=(4, 4)
        )
        self.model.head.fc = nn.Linear(
            in_features=1536,
            out_features=self.cfg.model.params.num_classes
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x

class RSNA2024Model3Heads(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super(RSNA2024Model3Heads, self).__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            model_name=self.cfg.model.name,
            pretrained=self.cfg.model.params.pretrained,
            in_chans=self.cfg.model.params.in_channels,
            features_only=False,
            global_pool='avg'
        )

        # Classifier for spinal canal stenosis
        self.scs_head = copy.deepcopy(self.model.head)
        self.scs_head.fc = nn.Sequential(
            nn.Linear(in_features=1536, out_features=768),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(in_features=768, out_features=384),
            nn.GELU(),
            nn.Linear(in_features=384, out_features=15),
        )

        # Classifier for neural foraminal narrowing
        self.nfn_head = copy.deepcopy(self.model.head)
        self.nfn_head.fc = nn.Sequential(
            nn.Linear(in_features=1536, out_features=768),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(in_features=768, out_features=384),
            nn.GELU(),
            nn.Linear(in_features=384, out_features=30),
        )

        # Classifier for subarticular stenosis
        self.ss_head = copy.deepcopy(self.model.head)
        self.ss_head.fc = nn.Sequential(
            nn.Linear(in_features=1536, out_features=768),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(in_features=768, out_features=384),
            nn.GELU(),
            nn.Linear(in_features=384, out_features=30),
        )
        
        self.model.head = nn.Identity()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logit = self.model(x)
        scs_logit = self.scs_head(logit)
        nfn_logit = self.nfn_head(logit)
        ss_logit = self.ss_head(logit)
        return scs_logit, nfn_logit, ss_logit

class RSNA2024_ViT_HipOA(nn.Module):

    def __init__(
            self,
            cfg: ModelConfig,
    ) -> None:
        super(RSNA2024_ViT_HipOA, self).__init__()
        self.cfg = cfg
        self.model = models.vit_b_16()
        self.spine_classifier = [
            nn.Linear(in_features=768,
                      out_features=384,
                      bias=True),
            nn.Linear(in_features=384,
                      out_features=self.cfg.model.params.num_classes)
        ]
        self.init_params()
        self.model.heads = nn.Sequential(*self.spine_classifier)
    

    def init_params(self) -> None:
        self.init_final_layer()
        self.load_pretrained_weight()
        self.change_first_layer()
        self.set_dropout_rate()

    def change_first_layer(self) -> None:
        self.model.conv_proj = nn.Conv2d(
            self.cfg.model.params.in_channels, 
            768, 
            kernel_size=(16, 16), 
            stride=(16, 16)
        )
        param = self.model.state_dict()
        param['conv_proj.weight'] = nn.Parameter(
            torch.randn(768,self.cfg.model.params.in_channels,16,16)
        )
    
    def init_final_layer(self) -> None:
        self.model.heads = nn.Sequential(
            nn.Linear(in_features=768,
                      out_features=7)
        )

    def set_dropout_rate(self) -> None:
        for layer in self.model.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.1
            else:
                pass

    def load_pretrained_weight(self) -> None:
        try:
            self.model.load_state_dict(
                torch.load(
                    f'/kaggle/input/rsna2024-python/kaggle/src/models/weights/vit_weight.pth',
                    map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                )
            )
            print("- Weight File Loaded !!")
        except Exception as e:
            print(e, "- Weight File Not Found !!")
    
    def forward(self, x) -> torch.Tensor:
        spine_logits = self.model(x)
        return spine_logits
import torch
import torch.nn as nn
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
            global_pool='avg'
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x

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
                    f'/Users/markun/git/rsna2024/kaggle/src/models/weights/vit_weight.pth',
                    map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                )
            )
        except Exception as e:
            print(e, "- Weight File Not Found !!")
    
    def forward(self, x) -> torch.Tensor:
        spine_logits = self.model(x)
        return spine_logits
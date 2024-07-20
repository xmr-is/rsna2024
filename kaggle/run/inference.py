import hydra
import os
import sys
from omegaconf import DictConfig

from .config import InferenceConfig

@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: InferenceConfig):
    print(cfg)

if __name__ == "__main__":
    main()
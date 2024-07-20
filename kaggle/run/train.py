import hydra
import os
import sys
from omegaconf import DictConfig

# Kaggle
# from .config import TrainConfig

# Local
from run.config import TrainConfig
from src.models.model import RSNA24Model
from src.dataset.prepare_data import PrepareData
from src.dataset.datamodule import TrainDataModule
from src.utils.environment_helper import EnvironmentHelper

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: TrainConfig):
    print(cfg)

    env = EnvironmentHelper(cfg)
    env.set_random_seed(cfg.seed)

    prepare_data = PrepareData(cfg)
    df = prepare_data.read_csv()

    datamodule = TrainDataModule(cfg, df)
    train_loader = datamodule.train_dataloader()
    valid_loader = datamodule.valid_dataloader()

    model = RSNA24Model(cfg).to(env.device())

    trainer = Trainer(
        model

    )


if __name__ == "__main__":
    main()
from dataclasses import dataclass
from typing import Any

@dataclass
class CommonConfig:
    conditions: list[str]
    levels: list[str]
    label2id: dict[str,int]

@dataclass
class DirectoryConfig:
    base_dir: str
    input_dir: str
    output_dir: str
    image_dir: str
    submission_dir: str

@dataclass
class SplitConfig:
    fold: int
    train_study_id: int
    valid_study_id: int

@dataclass
class PrepareDataConfig:
    common: CommonConfig
    directory: DirectoryConfig
    split: SplitConfig

@dataclass
class DatasetConfig:
    image_size: int
    augmentation: dict[str, Any]

@dataclass
class TrainerConfig:
    epochs: int
    early_stopping_epochs: int
    use_amp: bool
    num_workers: str
    grad_acc: int

@dataclass
class InferencerConfig:
    use_amp: bool
    num_workers: str

@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]

@dataclass
class OptimizerConfig:
    name: str
    lr: float

@dataclass
class SchedulerConfig:
    name: str
    wd: float

@dataclass
class CriterionConfig:
    name: str
    max_grad_norm: Any

@dataclass
class TrainConfig:
    exp_name: str
    debug: bool
    wandb: str
    num_folds: int
    batch_size: int
    seed: int
    common: CommonConfig
    directory: DirectoryConfig
    dataset: DatasetConfig
    prepare_data: PrepareDataConfig
    split: SplitConfig
    trainer: TrainerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    criterion: CriterionConfig

@dataclass
class InferenceConfig:
    exp_name: str
    debug: bool
    num_folds: int
    batch_size: int
    seed: int
    common: CommonConfig
    directory: DirectoryConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
    inferencer: InferencerConfig
    model: ModelConfig
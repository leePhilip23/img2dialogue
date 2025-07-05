from dataclasses import dataclass

@dataclass
class Dataset:
    path: str
    train: str
    valid: str

@dataclass
class DataLoader:
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool

@dataclass
class DataConfig:
    config: Dataset
    train: DataLoader
    valid: DataLoader

@dataclass
class ModelConfig:
    img_model: str
    small_lm: str

@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    save_pth: str

@dataclass
class MasterConfig:
    model: ModelConfig
    dataloaders: DataConfig
    training: TrainingConfig

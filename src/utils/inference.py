import wandb
import torch
from torch.utils.data import DataLoader
from .loader import CustomDataset
from .model import Model
from .config import MainConfig, TrainingConfig


def test_data(cfg: MainConfig) -> tuple[DataLoader, DataLoader]:
    """Loads the training and validation data from the dataset"""
    test_data = CustomDataset(
        cfg.config.test, 
        cfg.config.path, 
        cfg.model.img_model
    )

    test_dataloader = DataLoader(
        test_data, 
        batch_size=cfg.train.batch_size, 
        num_workers=cfg.train.num_workers, 
        pin_memory=cfg.train.pin_memory, 
        shuffle=cfg.train.shuffle
    )

    return test_dataloader


def _run_predict(
    cfg: TrainingConfig, 
    test_loader: DataLoader,
    model: Model
) -> None:
    running_loss = 0
    device = get_device()
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            running_loss += outputs.loss.item()

    return round(running_loss / len(test_loader), 3)


# TODO: Weights and Biases Setup
def run_inference(cfg: MainConfig, test_laoder: DataLoader):
    avg_loss = _run_predict(cfg, test_laoder)

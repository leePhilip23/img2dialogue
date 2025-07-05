import os
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from torch.utils.data import DataLoader
from torch.optim import AdamW
from .utils.loader import CustomDataset
from .utils.model import Model
from src import log


def _get_device() -> device:
    """Sets the device based on hardware available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model(cfg) -> Model:
    """Initializes the model with the available hardware"""
    device = _get_device()
    return Model(
        img_model=cfg.model.img_model,
        small_lm=cfg.model.small_lm
    ).to(device)
    

def _get_data(cfg) -> tuple[DataLoader, DataLoader]:
    """Loads the training and validation data from the dataset"""
    train_data = CustomDataset(
        cfg.config.train, 
        cfg.config.path, 
        cfg.model.img_model
    )
    valid_data = CustomDataset(
        cfg.config.valid, 
        cfg.config.path, 
        cfg.model.img_model
    )

    train_dataloader = DataLoader(
        train_data, 
        batch_size=cfg.train.batch_size, 
        num_workers=cfg.train.num_workers, 
        pin_memory=cfg.train.pin_memory, 
        shuffle=cfg.train.shuffle
    )
    valid_dataloader = DataLoader(
        valid_data, 
        batch_size=cfg.valid.batch_size, 
        num_workers=cfg.valid.num_workers, 
        pin_memory=cfg.valid.pin_memory, 
        shuffle=cfg.valid.shuffle
    )

    return train_dataloader, valid_dataloader


@hydra.main(config_path="conf", config_name="config.yml")
def main(cfg):
    """Main function to run the training process"""
    train_loader, valid_loader = _get_data()

    
if __name__ == '__main__':
    main()
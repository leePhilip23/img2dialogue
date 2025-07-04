import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils.loader import CustomDataset
from src import log


def get_data() -> tuple[DataLoader, DataLoader]:
    train_data = CustomDataset(
        "train", 
        "HuggingFaceM4/VisDial", 
        "Salesforce/blip-image-captioning-base"
    )
    valid_data = CustomDataset(
        "valid", 
        "HuggingFaceM4/VisDial", 
        "Salesforce/blip-image-captioning-base"
    )

    #TODO: Put config in hydra
    train_dataloader = DataLoader(train_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)

    return train_dataloader, valid_dataloader


def main():
    train_loader, valid_loader = get_data()
    
    
if __name__ == '__main__':
    main()
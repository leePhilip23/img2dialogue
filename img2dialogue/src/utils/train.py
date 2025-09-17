import torch
import mlflow
from torch.utils.data import DataLoader
from torch.optim import AdamW
from logger import log
from .loader import CustomDataset
from .model import Model
from .config import MainConfig, TrainingConfig


def train_data(cfg: MainConfig) -> tuple[DataLoader, DataLoader]:
    """Loads the training and validation data from the dataset"""
    train_data = CustomDataset(
        cfg.data_config.train, 
        cfg.data_config.path, 
        cfg.base_models.img_model
    )
    valid_data = CustomDataset(
        cfg.data_config.valid, 
        cfg.data_config.path, 
        cfg.base_models.img_model
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.data_train.batch_size,
        num_workers=cfg.data_train.num_workers,
        pin_memory=cfg.data_train.pin_memory,
        shuffle=cfg.data_train.shuffle,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=cfg.data_valid.batch_size,
        num_workers=cfg.data_valid.num_workers,
        pin_memory=cfg.data_valid.pin_memory,
        shuffle=cfg.data_valid.shuffle,
    )

    return train_dataloader, valid_dataloader


def _train_loop(
    cfg: TrainingConfig,
    epoch: int,
    model: Model,
    train_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """
    Runs one epoch of training for the given model and logs the average batch loss

    Args:
        cfg: Hydra config object containing training settings
        epoch (int): Current epoch number
        model (Model): PyTorch model to be trained
        train_loader (DataLoader): DataLoader for the training dataset
        optimizer (AdamW): Optimizer used to update model weights

    Returns:
        float: Average training loss for the epoch, rounded to 3 decimal places
    """
    model.train()
    running_loss = 0
    for i, data in enumerate(train_loader):
        img, txt, label = (
            data["img"].to(device),
            data["txt"].to(device),
            data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(img, txt, label)
        outputs.loss.backward()
        optimizer.step()
        running_loss += outputs.loss.item()

        # Calculate and print the average loss for the current batch
        avg_loss = running_loss / (i + 1)
        log.info(
            f"Epoch {epoch + 1}/{cfg.train_param.epochs}, "
            f"Batch {i+1}/{len(train_loader)}, "
            f"Avg Loss: {avg_loss:.4f}"
        )

    return round(running_loss / len(train_loader), 3)


def _valid_loop(
    cfg: TrainingConfig,
    epoch: int,
    model: Model,
    valid_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Runs one epoch of validation and logs the average batch loss

    Args:
        cfg: Hydra config object containing training settings
        epoch (int): Current epoch number
        model: PyTorch model being evaluated
        valid_loader (DataLoader): DataLoader for the validation dataset

    Returns:
        float: Average validation loss for the epoch, rounded to 3 decimal places
    """
    model.eval()
    running_loss = 0
    for i, data in enumerate(valid_loader):
        img, txt, label = (
            data["img"].to(device),
            data["txt"].to(device),
            data["label"].to(device),
        )
        outputs = model(img, txt, label)
        running_loss += outputs.loss.item()

        # Calculate and print the average loss for the current batch
        avg_loss = running_loss / (i + 1)
        log.info(
            f"Epoch {epoch + 1}/{cfg.train_param.epochs}, "
            f"Batch {i+1}/{len(valid_loader)}, "
            f"Avg Loss: {avg_loss:.4f}"
        )

    return round(running_loss / len(valid_loader), 3)


def run_training(
    cfg: TrainingConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: Model,
    device: torch.device,
    num_epochs: int,
    testing: bool = False,
) -> list | None:
    """
    Trains the model using the provided training and validation DataLoaders
    Handles training, validation, early stopping, and model checkpoint saving

    Args:
        cfg: Hydra config object containing training parameters and save path
        train_loader (DataLoader): DataLoader for the training dataset
        valid_loader (DataLoader): DataLoader for the validation dataset
    """

    # Set mlflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Vision Language Model Finetune")

    patience = 3
    patience_count = 0
    best_val_loss = float("inf")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train_param.learning_rate,
        weight_decay=cfg.train_param.weight_decay,
    )
    test_loss = []

    with mlflow.start_run():
        for epoch in range(num_epochs):
            train_loss = _train_loop(cfg, epoch, model, train_loader, optimizer, device)
            log.info(f"Epoch {epoch} Avg Training Loss: {train_loss:.4f}")
            mlflow.log_metric("Train Loss", train_loss)

            val_loss = _valid_loop(cfg, epoch, model, valid_loader, device)
            log.info(f"Epoch {epoch} Avg Validation Loss: {val_loss:.4f}")
            mlflow.log_metric("Validation Loss", val_loss)

            if testing:
                test_loss.append(train_loss)

            # Early stopping condition
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0

                # Log model to registry
                mlflow.pytorch.log_model(
                    name="CLIP-Qwen3B-0.6-VLM",
                    artifact_path="VLM",
                    registered_model_name="CLIP-Qwen3B-0.6-VLM",
                )
                log.info(f"Epoch {epoch+1}: Validation loss improved, saving best model.")
            else:
                patience_count += 1
                if patience_count >= patience:
                    log.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                    break

    if testing:
        return test_loss
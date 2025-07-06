import hydra
import torch
from torch import device
import src.train as train
import src.inference as inference
from .utils.model import Model
from .utils.config import MainConfig, ModelConfig


def get_device() -> device:
    """Sets the device based on hardware available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(cfg: ModelConfig) -> Model:
    """Initializes the model with the available hardware"""
    device = get_device()
    return Model(
        img_model=cfg.model.img_model,
        small_lm=cfg.model.small_lm
    ).to(device)


@hydra.main(config_path="conf", config_name="config.yml")
def main(cfg: MainConfig, train_val: str = "train"):
    """Main function to run the training process"""

    model = get_model()
    if train_val == "train":
        train_loader, valid_loader = train.get_data()
        train.run_training(cfg, train_loader, valid_loader, model)
    elif train_val == "inference":
        test_loader = inference.get_data()
        inference.run_inference(cfg, test_loader)
        


if __name__ == '__main__':
    main()
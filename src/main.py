import hydra
import torch
from utils.train import train_data, run_training
from utils.inference import test_data, run_inference
from utils.model import Model
from utils.config import MainConfig, ModelConfig


def get_device() -> torch.device:
    """Sets the device based on hardware available"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(cfg: ModelConfig, device: str) -> Model:
    """Initializes the model with the available hardware"""
    return Model(
        img_model=cfg.base_models.img_model,
        small_lm=cfg.base_models.slm
    ).to(device)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: MainConfig, train_val: str = "train") -> None:
    """Main function to run the training process"""
    device = get_device()
    model = get_model(cfg, device)
    if train_val == "train":
        train_loader, valid_loader = train_data(cfg)
        run_training(cfg, train_loader, valid_loader, model, device)
    elif train_val == "inference":
        test_loader = test_data(cfg)
        run_inference(cfg, test_loader, model, device)
        

if __name__ == '__main__':
    main()
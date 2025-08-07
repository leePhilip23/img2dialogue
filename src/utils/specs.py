import torch 
from .model import Model
from .config import ModelConfig

class Retrieve:
    @staticmethod
    def get_device() -> torch.device:
        """Sets the device based on hardware available"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


    @staticmethod
    def get_model(cfg: ModelConfig, device: str) -> Model:
        """Initializes the model with the available hardware"""
        return Model(
            img_model=cfg.base_models.img_model,
            small_lm=cfg.base_models.slm
        ).to(device)
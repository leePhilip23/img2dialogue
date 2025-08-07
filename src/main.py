import hydra
import torch
from utils.train import train_data, run_training
from utils.inference import test_data, run_inference
from utils.model import Model
from utils.config import MainConfig, ModelConfig
from utils.specs import Retrieve


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: MainConfig, train_val: str = "train") -> None:
    """Main function to run the training process"""
    device = Retrieve.get_device()
    model = Retrieve.get_model(cfg, device)
    if train_val == "train":
        train_loader, valid_loader = train_data(cfg)
        run_training(cfg, train_loader, valid_loader, model, device, cfg.train_param.epochs)
    elif train_val == "inference":
        test_loader = test_data(cfg)
        run_inference(cfg, test_loader, model, device)
        

if __name__ == '__main__':
    main()
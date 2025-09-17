from hydra import initialize, compose
from src.utils.specs import Retrieve
from src.utils.train import train_data, run_training


def test_model() -> None:
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config")
        device = Retrieve.get_device()
        model = Retrieve.get_model(cfg, device)
        train_loader, valid_loader = train_data(cfg)
        loss = run_training(
            cfg, 
            train_loader, 
            valid_loader, 
            model, 
            device,
            cfg.train_param.t_epochs,
            testing=True
        )
        assert loss[0] > loss[1], "Loss does not decrease"     
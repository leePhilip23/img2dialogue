from hydra import initialize, compose

def test_main_conf() -> None:
    with initialize(config_path="../../conf", version_base=None):
        cfg = compose(config_name="config")
        assert "data_config" in cfg, "Data configs does not exist"
        assert "data_train" in cfg, "Data train does not exist"
        assert "data_valid" in cfg, "Data valid does not exist"
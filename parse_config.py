from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="conf", config_name="example")
def parse_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    parse_cfg()

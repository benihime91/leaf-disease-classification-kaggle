import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src import _logger as logger
from src.core import seed_everything
from src.models import Task

warnings.filterwarnings("ignore")
OmegaConf.register_resolver("eval", lambda x: eval(x))


def main(cfg: DictConfig):
    _ = seed_everything(cfg.training.random_seed)
    task = Task(cfg)
    trainer = instantiate(cfg.trainer)
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(task)

    fig = lr_finder.plot(suggest=True)
    # create directory if does not exists
    os.makedirs(cfg.general.save_dir, exist_ok=True)
    # save lr-finder plot to memory
    _path = os.path.join(cfg.general.save_dir, f"lr-finder-plot.png")
    fig.savefig(_path)
    logger.info(f"\nSuggested LR's : {lr_finder.suggestion():.7f}")
    logger.info(f"Results saved to {_path}")


@hydra.main(config_path="conf", config_name="config")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    # run train
    cli_hydra()

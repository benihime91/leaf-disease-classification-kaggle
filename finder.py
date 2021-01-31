import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src import _logger as logger
from src.core import seed_everything
from src.models import Task

warnings.filterwarnings("ignore")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)


def main(cfg: DictConfig):
    _ = seed_everything(cfg.general.random_seed)
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
    logger.info(f"Suggested LR's : {lr_finder.suggestion():.7f}")
    logger.info(f"Results saved to {_path}")


@hydra.main(config_path="conf", config_name="effnet-base")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    # run train
    cli_hydra()

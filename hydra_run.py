import hydra
from omegaconf import DictConfig
import os
import warnings
import logging

warnings.filterwarnings('ignore')

from experiment import run
from cutmix import run as cut_mix_run
from fmix import run as fmix_run

__all__ = {"experiment": run, "fmix": fmix_run, "cutmix": cut_mix_run,}

log = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    
    print(cfg.pretty())
    run_fn = cfg.run

    run_fn(cfg, logger=log)



if __name__ == '__main__':
    run_model()

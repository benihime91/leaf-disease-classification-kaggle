import albumentations as A
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from timm.data.constants import (
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from timm.data.transforms import _pil_interp


def instantiate_transforms(cfg: DictConfig, global_config: DictConfig = None):
    "loades in individual transformations"
    if cfg._target_ == "aa":
        img_size_min = global_config.input.input_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple(
                [min(255, round(255 * x)) for x in global_config.input.mean]
            ),
        )

        if (
            global_config.input.interpolation
            and global_config.input.interpolation != "random"
        ):
            aa_params["interpolation"] = _pil_interp(global_config.input.interpolation)

        # Load autoaugment transformations
        if cfg.policy.startswith("rand"):
            return rand_augment_transform(cfg.policy, aa_params)
        elif cfg.policy.startswith("augmix"):
            aa_params["translate_pct"] = 0.3
            return augment_and_mix_transform(cfg.policy, aa_params)
        else:
            return auto_augment_transform(cfg.policy, aa_params)

    else:
        return instantiate(cfg)


def create_transform(cfg: DictConfig, global_config: DictConfig = None, verbose=False):
    "creates transoformations to be used in datasets"
    train_augs_initial = [
        instantiate_transforms(t, global_config) for t in cfg.train.before_mix
    ]
    train_augs_final = [
        instantiate_transforms(t, global_config) for t in cfg.train.after_mix
    ]
    valid_augs = [instantiate_transforms(t, global_config) for t in cfg.valid]

    if cfg.backend == "torchvision":
        compose_func = T.Compose
    elif cfg.backend == "albumentations":
        compose_func = A.Compose
    else:
        raise NameError

    # compose augmentations
    train_augs_initial = compose_func(train_augs_initial)
    train_augs_final = compose_func(train_augs_final)
    valid_augs = compose_func(valid_augs)
    return train_augs_initial, train_augs_final, valid_augs

import logging
import os

import albumentations as A
import pandas as pd
import timm
import wandb
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastcore.script import *

from src.fast_utils import *

logging.basicConfig(format="%(asctime)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)


PROJECT = "kaggle-leaf-disease-fastai-runs"


@delegates(DataBlock)
def get_data(src_pth: str, im_pth: str, curr_fold: int, bs: int = 64, **kwargs):
    print(src_pth)
    print(im_pth)
    df = pd.read_csv(src_pth)
    df["filePath"] = [os.path.join(im_pth, df["image_id"][idx]) for idx in range(len(df))]
    df["is_valid"] = [df["kfold"][n] == curr_fold for n in range(len(df))]
    df = df.sample(frac=1).reset_index(drop=True, inplace=False)

    cassava = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    splitter=ColSplitter(col="is_valid"),
                    get_x=lambda x: x["filePath"],
                    get_y=lambda x: x["label"],
                    **kwargs,)

    n_gpus = num_distrib() or 1
    workers = min(8, num_cpus() // n_gpus)
    return cassava.dataloaders(df, bs=bs, num_worker=workers)


@call_parse
def main(
    mish: Param("use mish activation", store_true),
    display: Param("print out the mode", store_true),
    lrfinder: Param("Run learning rate finder; don't train", store_true),
    src: Param("path to the stratified dataframe", str) = None,
    ims: Param("path to the image directory", str) = None,
    fold: Param("fold to train on", int) = 0,
    encoder: Param("architecture for the encoder model from timm lib", str) = None,
    cut: Param("num layers to cut from the encoder", int) = -2,
    bs: Param("batch_size for the experiment", int) = 64,
    seed: Param("seed for the experiment", int) = 42,
    dims: Param("size of input images", int) = 224,
    weights: Param("path to a torch state dict", str) = None,
    opt: Param("optimizer (adam, ranger)", str) = "adam",
    lr: Param("learning rate", float) = 1e-3,
    wd: Param("weight decay", float) = 1e-02,
    epochs: Param("number of epochs", int) = 30,
    lr_mult: Param("factor for discriminative Lrs", float) = 100.0,
    mixup: Param("mixup (0 for no mixup)", float) = 0.0,
    pct_start: Param("annealing pct start for scheduler", float) = 0.33,
    grad_accumulate: Param("gradient accumulation", int) = 0,
    sched_type: Param("LR schedule type (one_cycle, flat_cos)", str) = "one_cycle",
):
    set_seed(seed, reproducible=True)

    IN_JUPYTER, IN_NOTEBOOK, IN_IPYTHON, IN_COLAB= True, True, True, True

    if not lrfinder: run = wandb.init(project=PROJECT)

    idx = generate_random_id()

    save_name = f"{encoder}-fold={fold}-{idx}"
    save_dir = os.getcwd()

    train_augments = A.Compose(
        [
            A.RandomResizedCrop(dims, dims, p=0.5),
            A.Resize(dims, dims, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([A.Transpose(), A.VerticalFlip(), A.Transpose()], p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Cutout(p=0.5, num_holes=20),
            A.CoarseDropout(p=0.5),
        ]
    )

    valid_augments = A.Compose([A.Resize(dims, dims, p=1.0),])
    item_tfms = [AlbumentationsTransform(train_augments, valid_augments)]
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    print(f"seed: {seed}; size: {dims}; fold: {fold}; bs: {bs}; base: {encoder}")

    dls = get_data(src, ims, curr_fold=fold, bs=bs, item_tfms=item_tfms,batch_tfms=batch_tfms,)

    encoder = timm.create_model(encoder, pretrained=True)
    if mish:
        print("Using Mish activation")
        model = TransferLearningModel(encoder, dls.c, cut=cut, act=Mish())
        replace_with_mish(model)
    else:
        model = TransferLearningModel(encoder, dls.c, cut=cut)

    if weights is not None:  model.load_state_dict(torch.load(weights))

    if display:  print(model)

    if opt == "adam":
        print(f"Using Adam Optimizer, lr: {lr}, wd: {wd}, epochs: {epochs}")
        opt_func = Adam
    elif opt == "ranger":
        print(f"Using Ranger Optimizer, lr: {lr}, wd: {wd}, epochs: {epochs}")
        opt_func = ranger
    else:
        print(f"Switching to Adam Optimizer, lr: {lr}, wd: {wd}, epochs: {epochs}")
        opt_func = Adam

    if not lrfinder:
        callbacks = [WandbCallback(log_model=False, log_preds=False, seed=seed)]
    else: callbacks = None

    learn = Learner(dls, model, metrics=[accuracy], 
                loss_func=LabelSmoothingCrossEntropy(), opt_func=opt_func,
                splitter=custom_splitter, cbs=callbacks, train_bn=True,)

    learn = learn.to_native_fp16()
    learn.unfreeze()

    if lrfinder:
        # run learning rate finder
        learn.lr_find()
        learn.recorder.plot_lr_find()

    else:
        batch_cbs = []
        if grad_accumulate > 0:
            print(f"Accumulating gradients for {grad_accumulate} batches")
            batch_cbs.append(GradientAccumulation(grad_accumulate * dls.bs))
        if mixup > 0:
            print("Using MixUp")
            batch_cbs.append(MixUp(mixup))

        if sched_type == "one_cycle":
            print(f"One Cycle Annealing; pct_start: {pct_start};")
            learn.fit_one_cycle(epochs, slice(lr / lr_mult, lr), pct_start=pct_start, wd=wd, cbs=batch_cbs,)
        elif sched_type == "flat_cos":
            print(f"Flat Cos Annealing; pct_start: {pct_start};")
            learn.fit_flat_cos(epochs, slice(lr / lr_mult, lr), pct_start=pct_start, wd=wd, cbs=batch_cbs,)

        learn = learn.to_native_fp32()
        
        sdirs = os.path.join(save_dir, f"{save_name}.pt")
        torch.save(learn.model.state_dict(), sdirs)
        wandb.save(sdirs)
        print(f"weights saved as {save_name}.pt")
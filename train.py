import os

import albumentations as A
import pandas as pd
import timm
from fastai.callback.wandb import *
from fastai.vision.all import *
from fastcore.script import *
from fastprogress import fastprogress

from src.fast_utils import *

fastprogress.MAX_COLS = 80


@delegates(DataBlock)
def get_data(src_pth: str, im_pth: str, curr_fold: int, bs: int = 64, **kwargs):
    print(src_pth)
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

@patch
def plot_lr_find_modified(self:Recorder, suggestions= False,skip_end=5, lr_min=None, lr_steep=None):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs    = self.lrs    if skip_end==0 else self.lrs   [:-skip_end]
    losses = self.losses if skip_end==0 else self.losses[:-skip_end]

    if suggestions:
        lr_min_index = min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_min))
        lr_steep_index = min(range(len(lrs)), key=lambda i: abs(lrs[i]-lr_steep))

    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    if suggestions:
        ax.plot(lr_min,L(losses)[lr_min_index],'ro')
        ax.plot(lr_steep,L(losses)[lr_steep_index],'ro')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    return fig


@call_parse
def main(
    mish: Param("use mish activation", store_true),
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
    item_tfms  = [AlbumentationsTransform(train_augments, valid_augments)]
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    print(f"base: {encoder}; lr: {lr}; wd: {wd}; epochs: {epochs}; seed: {seed}; size: {dims}; fold: {fold}; bs: {bs};")

    dls = get_data(src, ims, curr_fold=fold, bs=bs, item_tfms=item_tfms,batch_tfms=batch_tfms,)

    encoder = timm.create_model(encoder, pretrained=True)
    if mish:
        print("Using Mish activation")
        model = TransferLearningModel(encoder, dls.c, cut=cut, act=Mish())
        replace_with_mish(model)
    else:
        model = TransferLearningModel(encoder, dls.c, cut=cut)
    
    apply_init(model.decoder, nn.init.kaiming_normal_)

    if weights is not None:  
        model.load_state_dict(torch.load(weights))

    if opt == "adam":
        print(f"Using Adam Optimizer")
        opt_func = Adam
    elif opt == "ranger":
        print(f"Using Ranger Optimizer")
        opt_func = ranger
    else:
        print(f"Switching to Adam Optimizer")
        opt_func = Adam


    learn = Learner(dls, model, metrics=[accuracy], 
                opt_func=opt_func, loss_func=LabelSmoothingCrossEntropy(), 
                splitter=custom_splitter,).to_native_fp16()

    if lrfinder:
        IN_NOTEBOOK = True
        IN_IPYTHON = True
        # run learning rate finder
        res = learn.lr_find()
        print(res)
        fig = learn.recorder.plot_lr_find_modified()
        plt.savefig('lr_find.png')

    else:
        batch_cbs = []
        if grad_accumulate > 0:
            print(f"Accumulating gradients for {grad_accumulate} batches")
            batch_cbs.append(GradientAccumulation(grad_accumulate * dls.bs))
        if mixup > 0:
            print("Using MixUp")
            batch_cbs.append(MixUp(mixup))

        if sched_type == "one_cycle":
            print(f"Using One Cycle Annealing with pct_start: {pct_start}")
            learn.freeze()
            learn.fit_one_cycle(1, slice(lr / lr_mult, lr), pct_start=0.99, wd=wd, cbs=batch_cbs,)
            lr/=2
            learn.unfreeze()
            learn.fit_one_cycle(epochs, slice(lr / lr_mult, lr), pct_start=pct_start, wd=wd, cbs=batch_cbs,)
        elif sched_type == "flat_cos":
            print(f"Using Flat Cos Annealing with pct_start: {pct_start}")
            learn.freeze()
            learn.fit_flat_cos(1, slice(lr / lr_mult, lr), pct_start=0.99, wd=wd, cbs=batch_cbs,)
            lr/=2
            learn.unfreeze()
            learn.fit_flat_cos(epochs, slice(lr / lr_mult, lr), pct_start=pct_start, wd=wd, cbs=batch_cbs,)

        learn = learn.to_native_fp32()
        
        sdirs = os.path.join(save_dir, save_name)
        torch.save(learn.model.state_dict(), sdirs)
        print(f"weights saved to {sdirs}.pt")

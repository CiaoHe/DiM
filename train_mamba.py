# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiM.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math
from accelerate import Accelerator
from omegaconf import OmegaConf

from download import find_model
from diffuseMamba import DiM_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# add tensorboard support
from torch.utils.tensorboard import SummaryWriter

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)
    

def get_inverse_sqrt_schedule(optimizer, num_warmup_steps:int, t_ref:int):
    # https://github.com/mmathew23/improved_edm/blob/main/train.py#L59
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0 / math.sqrt(max(1.0, (current_step - num_warmup_steps) / t_ref))
    return LambdaLR(optimizer, lr_lambda)



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    # ! Attention: set 'step_scheduler_with_optimizer' as False when using multiple gpus
    # https://github.com/huggingface/accelerate/issues/2142#issuecomment-1878705812, then you can
    # call lr_scheduler.step() after optimizer.step() in the training loop
    # else: you need parse 'global_step' to lr_scheduler.step(global_step) in the training loop
    accelerator = Accelerator() 
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiM-XL/2 --> DiM-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(experiment_dir)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiM constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiM Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    # parse optimizer config
    assert args.optimizer_config is not None, "Please provide a yaml file for optimizer config"
    opt_config = OmegaConf.load(args.optimizer_config)
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=opt_config.optimizer.lr, 
        weight_decay=opt_config.optimizer.weight_decay, 
        betas=(opt_config.optimizer.beta1, opt_config.optimizer.beta2)
    )        
    # add lr scheduler, first 20 epochs use 5*lr, then down to lr
    if opt_config.lr_scheduler.name == "multistep":
        lr_scheduler = MultiStepLR(
            opt, milestones=opt_config.lr_scheduler.milestones, gamma=opt_config.lr_scheduler.gamma)
    elif opt_config.lr_scheduler.name == "inverse_sqrt":
        lr_scheduler = get_inverse_sqrt_schedule(
            opt, num_warmup_steps=opt_config.lr_scheduler.num_warmup_steps, t_ref=opt_config.lr_scheduler.t_ref)
    elif opt_config.lr_scheduler.name == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=opt_config.lr_scheduler.T_max, eta_min=opt_config.lr_scheduler.eta_min)
    else:
        raise ValueError(f"Unsupported lr scheduler: {opt_config.lr_scheduler.name}")    
    
    # resume from checkpoint if available
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if opt_config.optimizer.clip_grad_norm>0:
                accelerator.clip_grad_norm_(model.parameters(), opt_config.optimizer.clip_grad_norm)
            opt.step()
            train_steps += 1
            lr_scheduler.step(train_steps)
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr: {opt.param_groups[0]['lr']:.2e}")
                    writer.add_scalar('Loss/train', avg_loss * accelerator.num_processes, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiM_models.keys()), default="DiM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    # optimizer config
    parser.add_argument("--optimizer-config", type=str, default=None, help="Optional path to a yaml file")
    # continue training    
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom checkpoint")
    args = parser.parse_args()
    main(args)


"""
# DiM-S/2
export HF_HOME="/cto_labs/AIDD/cache"
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_mamba.py --model DiM-S/2 --feature-path /cto_labs/AIDD/DATA/imagenet1k/train_vae --optimizer-config optim_config/multistep.yaml


# DiM-L/2
export HF_HOME="/cto_labs/AIDD/cache"
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_mambaV1.py --model DiM-L/2 --feature-path /cto_labs/AIDD/DATA/imagenet1k/train_vae --lr 4e-4
"""
import os
import random
import argparse

import numpy as np
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
import torch

from dataset_metrla import MetrLA_Dataset
from dataset_pemsbay import PemsBAY_Dataset
from trainer import Trainer1D

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="metr_la")
    # args = parser.parse_args()

    # if args.dataset == "metr_la":
        
    #     model = Unet1D(
    #         dim = 64,
    #         dim_mults = (1, 2, 4, 8),
    #         channels = 207
    #     )

    #     diffusion = GaussianDiffusion1D(
    #         model,
    #         seq_length = 24,
    #         timesteps = 50,
    #         objective = 'pred_noise'
    #     )

    #     dataset = MetrLA_Dataset(
    #         mode = "train", 
    #         val_len = 0.1, 
    #         test_len = 0.2, 
    #         missing_pattern = "point",
    #         is_interpolate = True, 
    #         target_strategy = "random"
    #     )
        
    # elif args.dataset == "pems_bay":
         
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 325
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 24,
        timesteps = 50,
        objective = 'pred_noise'
    )

    dataset = PemsBAY_Dataset(
        mode = "train", 
        val_len = 0.1, 
        test_len = 0.2, 
        missing_pattern = "point",
        is_interpolate = True, 
        target_strategy = "random"
    )

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
        wandb = True                        # whether to use wandb 
    )

    trainer.train()

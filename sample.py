import os
import argparse

import torch
from torch.utils.data import DataLoader  
from tqdm import tqdm

from models import Unet1D, GaussianDiffusion1D
from dataset_metrla import MetrLA_Dataset
from dataset_pemsbay import PemsBAY_Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="metr_la")
    parser.add_argument("--missing_pattern", type=str, default="block")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.dataset == "metr_la":
        model = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 207
        )

        diffusion = GaussianDiffusion1D(
            model,
            seq_length = 24,
            timesteps = 50,
            objective = 'pred_noise',
            auto_normalize = False
        ).to(device)
    
        dataset = MetrLA_Dataset(
            mode = "train", 
            val_len = 0.1, 
            test_len = 0.2, 
            missing_pattern = args.missing_pattern,
            is_interpolate = True, 
            target_strategy = "random"
        )
    
        scaler = torch.from_numpy(dataset.train_std).to(device).float()
        mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

        dataset_test = MetrLA_Dataset(
            mode = "test", 
            val_len = 0.1, 
            test_len = 0.2, 
            missing_pattern = args.missing_pattern,
            is_interpolate = True, 
            target_strategy = "random"
        )
    elif args.dataset == "pems_bay":
        model = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 325
        )

        diffusion = GaussianDiffusion1D(
            model,
            seq_length = 24,
            timesteps = 50,
            objective = 'pred_noise',
            auto_normalize = False
        ).to(device)
    
        dataset = PemsBAY_Dataset(
            mode = "train", 
            val_len = 0.1, 
            test_len = 0.2, 
            missing_pattern = args.missing_pattern,
            is_interpolate = True, 
            target_strategy = "random"
        )
    
        scaler = torch.from_numpy(dataset.train_std).to(device).float()
        mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

        dataset_test = PemsBAY_Dataset(
            mode = "test", 
            val_len = 0.1, 
            test_len = 0.2, 
            missing_pattern = args.missing_pattern,
            is_interpolate = True, 
            target_strategy = "random"
        )
    
    num_workers = 4
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=num_workers, shuffle=False)
    
    ckpt = torch.load(f"results/{args.dataset}/model-20.pt")
    diffusion.load_state_dict(ckpt["model"])
    
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0
    
    with tqdm(test_loader) as it:
        for batch_no, test_batch in enumerate(it):
            sampled_seq = diffusion.sample_inpaint(batch_size=64, s=test_batch)
            # sampled_seq = sampled_seq.transpose(1, 2).mean(dim=0).cpu()
            sampled_seq = sampled_seq.transpose(1, 2).median(dim=0, keepdim=True)[0].cpu()
            # sampled_seq = sampled_seq.transpose(1, 2)[0].cpu()
            
            mse_current = ((test_batch["observed_data"] - sampled_seq) * (test_batch["observed_mask"] - test_batch["gt_mask"])) ** 2 * (scaler.cpu() ** 2)
            mae_current = torch.abs((test_batch["observed_data"] - sampled_seq) * (test_batch["observed_mask"] - test_batch["gt_mask"])) * scaler.cpu()
            
            mse_total += mse_current.sum().item()
            mae_total += mae_current.sum().item()
            evalpoints_total += (test_batch["observed_mask"] - test_batch["cond_mask"]).sum().item()
        
            it.set_postfix(
                ordered_dict={
                            "mse_total": mse_total / evalpoints_total,
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                refresh=True,
            )
        
    print(f"MSE: {mse_total / evalpoints_total:.2f}")
    print(f"MAE: {mae_total / evalpoints_total:.2f}")
    
    # s = next(iter(test_loader))
    # sampled_seq = diffusion.sample_inpaint(batch_size=16, s=s)
    
    # print(s["observed_data"][0, :4, :4])
    # print((s["observed_mask"] - s["cond_mask"])[0, :4, :4])
    # print(sampled_seq.transpose(1, 2).median(dim=0)[0][:4, :4])
    
    

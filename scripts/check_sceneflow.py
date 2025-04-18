import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from config import Config, TrainingConfig, setup_hydra_configurations
from data.data_loader import fetch_dataloader
from utils_d3roma.utils import seed_everything
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from utils_d3roma.utils import Normalizer
from utils_d3roma.frame_utils import read_gen
import torch.nn.functional as F
import shutil

import torch
import numpy as np
from PIL import Image

import os
logger = get_logger(__name__, log_level="INFO") # multi-process logging

Accelerator() # hack: enable logging

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def check(config: Config):
    cfg = config.task
    logger.info(cfg.train_dataset)

    train_dataloader, val_dataloader_lst = fetch_dataloader(cfg)
    logger.info(val_dataloader_lst[0].dataset.__class__.__name__)
    
    all_dataloaders = [train_dataloader]
    all_dataloaders.extend(val_dataloader_lst)

    count = 0 
    bads = {}

    for i, dataloader in enumerate([train_dataloader]): # all_dataloaders, val_dataloader_lst
        pbar = tqdm(total=len(dataloader))
        for j, data in enumerate(dataloader):
            # print(data.keys())
            B = data['mask'].shape[0]
            for b in range(B):
                # rgb = data['normalized_rgb'][b]
                index = data['index'][b]
                path = data['path'][b]

                raw_left = path.replace("disparity", "raw_cleanpass").replace("pfm", "png").replace("right", "left")
                # raw_right= path.replace("disparity", "raw_finalpass").replace("pfm", "png").replace("left", "right")

                raw_left = np.array(read_gen(raw_left))
                gt_left = np.array(read_gen(path))

                TP = ((raw_left > 0) & (np.abs(gt_left - raw_left) <= 2)).sum()
                FP = ((raw_left > 0) & (np.abs(gt_left - raw_left) > 2)).sum()
                FN = ((raw_left == 0) & (np.abs(gt_left - raw_left) <= 2)).sum()
                precision = TP / (TP + FP)
                recall = TP / (TP + FN) # biased

                # raw_right = read_gen(raw_right)
                                 
                # if precision < 0.6 and recall < 0.7:
                if precision < 0.2:
                    bads[path] = precision
                    logger.info(f"bad image {index}: {path}")

                    if True:
                        dump_dir = "./bad_sim"
                        shutil.copy2(path, f"{dump_dir}/{j}_{b}_disp.pfm")
                        shutil.copy2(path.replace("disparity", "raw_finalpass").replace("pfm", "png"), f"{dump_dir}/{j}_{b}_raw.png")
                        shutil.copy2(path.replace("disparity", "raw_cleanpass").replace("pfm", "png"), f"{dump_dir}/{j}_{b}_raw_clean.png")
                        shutil.copy2(path.replace("disparity", "frames_finalpass").replace("pfm", "png"), f"{dump_dir}/{j}_{b}_left.png")
                        shutil.copy2(path.replace("disparity", "frames_finalpass").replace("pfm", "png").replace("left", "right"), f"{dump_dir}/{j}_{b}_right.png")
                
                    count += 1
        
            pbar.update(1)

    logger.info(f"how many bad images? {len(bads.items())}")
    with open(f'bad_his.txt', 'w') as f:
        for path,epe in bads.items():
            f.write(f"{path} {epe}\n")

if __name__ == "__main__":
    seed_everything(0)
    setup_hydra_configurations()
    check()
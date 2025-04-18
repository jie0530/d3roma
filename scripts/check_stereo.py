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
import torch.nn.functional as F

import torch
import numpy as np
from PIL import Image

logger = get_logger(__name__, log_level="INFO") # multi-process logging

Accelerator() # hack: enable logging

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def check(config: Config):
    cfg = config.task
    logger.info(cfg.train_dataset)

    from utils_d3roma.camera import DepthCamera, Realsense
    from functools import partial
    from utils import frame_utils
    sim_camera = DepthCamera.from_device("sim")
    # sim_camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
    sim_camera.change_resolution(cfg.camera_resolution)
    disp_reader = partial(frame_utils.readDispReal, sim_camera)

    # sim_disp, sim_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_simDepthImage.exr")
    # sim_disp, sim_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_simDispImage.png")
    # raw_disp, raw_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_depth.exr")

    # epe = np.abs(sim_disp[sim_valid] - raw_disp[sim_valid]).mean()
    # assert epe < 1, f"bad quality sim disp, epe={epe}"
    
    train_dataloader, val_dataloader_lst = fetch_dataloader(cfg)
    logger.info(val_dataloader_lst[0].dataset.__class__.__name__)
    
    all_dataloaders = [train_dataloader]
    all_dataloaders.extend(val_dataloader_lst)
    bad = []
    
    stats = {
        'mean': [],
        'med': [],
        'min': [],
        'max': [],
        'std': []
    }

    stats_norm = {
        'mean': [],
        'med': [],
        'min': [],
        'max': [],
        'std': []
    }
    count = 0 

    norm = Normalizer.from_config(cfg)

    bads = {}

    for i, dataloader in enumerate(val_dataloader_lst): # all_dataloaders, [train_dataloader]
        pbar = tqdm(total=len(dataloader))
        for j, data in enumerate(dataloader):
            # print(data.keys())
            B = data['mask'].shape[0]
            for b in range(B):
                mask = data['mask'][b]
                # sim_mask = data['sim_mask'][b]

                disp = data['raw_disp'][b] 
                disp_norm = data["normalized_disp"][b]
                # rgb = data['normalized_rgb'][b]
                index = data['index'][b]
                path = data['path'][b]

                # sim_disp = data["sim_disp_unnorm"][b]
                # sim_valid = data["sim_mask"][b].bool()

                stats['mean'].append(disp.mean().item())
                stats['med'].append(disp.median().item())
                stats['min'].append(disp.min().item())
                stats['max'].append(disp.max().item())
                stats['std'].append(disp.std().item())

                stats_norm['mean'].append(disp_norm.mean().item())
                stats_norm['med'].append(disp_norm.median().item())
                stats_norm['min'].append(disp_norm.min().item())
                stats_norm['max'].append(disp_norm.max().item())
                stats_norm['std'].append(disp_norm.std().item())

                # sim_disp, sim_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_simDepthImage.exr")
                # sim_disp, sim_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_simDispImage.png")
                # raw_disp, raw_valid, min_disp, max_disp = disp_reader("datasets/HssdIsaacStd/train/102344049/kitchentable/1500_depth.exr")

                # epe = torch.abs(sim_disp[sim_valid] - disp[sim_valid]).mean()
                if True: #&epe > 2.:
                    # print(f"bad quality sim disp, epe={epe}, {data['path']}")
                    # bads[data['path'][b]] = epe

                    if "normalized_rgb" in data:
                        rgb = data['normalized_rgb'][b:b+1]
                        Image.fromarray(((rgb[0]+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_{j}_rgb.png")

                    if True:
                        left = data['left_image'][b:b+1]
                        Image.fromarray(((left[0]+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_{j}_left.png")

                        right = data['right_image'][b:b+1]
                        Image.fromarray(((right[0]+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_{j}_right.png")

                        H, W = disp.shape[-2:]
                        device = left.device

                        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                        xx = xx.unsqueeze(0).repeat(1, 1, 1).to(device)
                        yy = yy.unsqueeze(0).repeat(1, 1, 1).to(device)

                        # raw_disp = data['raw_disp'][b]
                        xx = (xx - disp) / ((W  - 1) / 2.) - 1 
                        yy = yy / ((H - 1) / 2.) - 1
                        grid = torch.stack((xx, yy), dim=-1)
                        warp_left_image = F.grid_sample(right, grid, align_corners=True, mode="bilinear", padding_mode="border")
                        warp_left_image[0][mask.repeat(3,1,1)<1.0] = -1
                        Image.fromarray(((warp_left_image[0]+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_{j}_warped_right.png")
                        loss = F.l1_loss(left[..., 0:], warp_left_image, reduction='mean')
                        logger.info(f"raw disp loss: {loss.item()}")
                        
                        sim_disp = norm.denormalize(data["sim_disp"])[b] 
                        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                        xx = xx.unsqueeze(0).repeat(B, 1, 1).to(device)
                        yy = yy.unsqueeze(0).repeat(B, 1, 1).to(device)
                        xx = (xx - sim_disp) / ((W  - 1) / 2.) - 1 
                        yy = yy / ((H - 1) / 2.) - 1
                        sim_grid = torch.stack((xx, yy), dim=-1)
                        warp_left_image_sim = F.grid_sample(right, sim_grid, align_corners=True, mode="bilinear", padding_mode="border")
                        # warp_left_image_sim[0][mask.repeat(3,1,1)<1.0] = -1 for sparse dataset
                        warp_left_image_sim[0][mask.repeat(3,1,1)<1.0] = -1
                        Image.fromarray(((warp_left_image_sim[0]+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_{j}_warped_right_sim.png")
                        loss_sim = F.l1_loss(left[..., 0:], warp_left_image_sim, reduction='mean')
                        logger.info(f"sim disp loss: {loss_sim.item()}")
                                 
                """ if True or mask.sum() / mask.numel() < 0.98:
                    bad.append(path)
                    logger.info(f"bad image {index}: {path}")

                    if True:
                        # low, high = torch.quantile(data['depth'][b], torch.tensor((0.02, 0.98))) # gt depth
                        # d = (data['depth'][b] - low) / (high - low)
                        # Image.fromarray(mask[0].cpu().numpy().astype(np.uint8)*255).save(f"{index}_mask.png")
                        # Image.fromarray((d[0].clamp(0,1)*255).cpu().numpy().astype(np.uint8)).save(f"{index}_depth_p.png")
                        Image.fromarray(((rgb+1) * 127.5).cpu().numpy().astype(np.uint8).transpose(1,2,0)).save(f"{index}_rgb.png") """
                
                count += 1
                if count % 1000 == 0:
                    print("stats_raw...")
                    print(f"tatal={len(stats['mean'])}")
                    for k, vals in stats.items():
                        print(f"{k}: {np.mean(vals)}")    
                    print("stats_norm...")
                    for k, vals in stats_norm.items():
                        print(f"{k}: {np.mean(vals)}")
                        
            #     break
            # break
            pbar.update(1)

    print(f"tatal={len(stats['mean'])}")
    print("stats_raw...")
    for k, vals in stats.items():
        print(f"{k}: {np.mean(vals)}")
    print("stats_norm...")
    for k, vals in stats_norm.items():
        print(f"{k}: {np.mean(vals)}")

    # print("stats:", stats)
    logger.info(f"how many bad images? {len(bads.items())}")
    with open(f'bad_his.txt', 'w') as f:
        for path,epe in bads.items():
            f.write(f"{path} {epe}\n")

if __name__ == "__main__":

    seed_everything(0)
    setup_hydra_configurations()
    check()
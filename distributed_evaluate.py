import os, sys
import math
import argparse
import torch
import logging
from tqdm import tqdm

from core.custom_pipelines import GuidedLatentDiffusionPipeline
from accelerate import Accelerator, PartialState
from core.stereo_matching import StereoMatching
from core.guidance import FlowGuidance
import numpy as np
from utils.utils import seed_everything
from config import TrainingConfig, create_sampler
from diffusers import UNet2DModel, DDIMScheduler
from utils.utils import InputPadder, metrics_to_dict, pretty_json
from accelerate.logging import get_logger
from utils.camera import  plot_error_map
from evaluate import eval_batch
from data.stereo_datasets import *
from data.mono_datasets import *

import hydra
from config import Config, TrainingConfig, create_sampler, setup_hydra_configurations

logger = get_logger(__name__, log_level="INFO")

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def run_distributed_eval(base_cfg: Config):
    if base_cfg.seed != -1:
        seed_everything(base_cfg.seed) # for reproducing

    accelerator = Accelerator() # hack: enable logging

    config = base_cfg.task
    assert len(config.eval_dataset) == 1, "only support single dataset for evaluation"

    inputPadder = InputPadder(config.image_size, divis_by=8)
    # config.camera # hack init default camera

    patrained_path = f"{config.resume_pretrained}"
    if os.path.exists(patrained_path):
        logger.info(f"load weights from {patrained_path}")
        """ pipeline = GuidedLatentDiffusionPipeline.from_pretrained(patrained_path).to("cuda")
        # model = UNet2DConditionModel.from_pretrained(patrained_path)

        from diffusers import DDIMScheduler
        ddim = DDIMScheduler.from_config(dict(
            beta_schedule = config.beta_schedule, # "scaled_linear",
            beta_start = config.beta_start, # 0.00085,
            beta_end = config.beta_end, # 0.012,
            clip_sample = config.clip_sample, # False,
            num_train_timesteps = config.num_train_timesteps, # 1000,
            prediction_type = config.prediction_type, # #"v_prediction",
            set_alpha_to_one = False,
            skip_prk_steps = True,
            steps_offset = 1,
            trained_betas = None
        ))
        pipeline.scheduler = ddim """

        from core.custom_pipelines import GuidedDiffusionPipeline, GuidedLatentDiffusionPipeline
        clazz_pipeline = GuidedLatentDiffusionPipeline if config.ldm else GuidedDiffusionPipeline
        pipeline = clazz_pipeline.from_pretrained(patrained_path).to("cuda")
        pipeline.guidance.flow_guidance_mode=config.flow_guidance_mode

        pipeline.scheduler = create_sampler(config, train=False)
    else:
        raise ValueError(f"patrained path not exists: {patrained_path}")

    if config.eval_output:
        eval_output_dir = f"{config.resume_pretrained}/{config.eval_output}"
    else:
        eval_output_dir = f"{config.resume_pretrained}/dist.{config.eval_dataset[0]}.g.{config.guide_source}.b{config.eval_num_batch}.{config.eval_split}"

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)
    
    logger.logger.addHandler(logging.FileHandler(f"{eval_output_dir}/eval.log"))
    logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"eval output dir: {eval_output_dir}")

    from data.data_loader import create_dataset
    val_dataset = create_dataset(config, config.eval_dataset[0], split = config.eval_split)
    # print(f"eval_batch_size={config.eval_batch_size}"); exit(0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config.eval_batch_size,
                                                shuffle=True,
                                                pin_memory=False, 
                                                drop_last=False)
    
    """ if type(model.sample_size) == list:
        model.sample_size[0] = inputPadder.padded_size[0]
        model.sample_size[1] = inputPadder.padded_size[1] """

    # distributed evaluation
    val_dataloader = accelerator.prepare(val_dataloader)

    pbar = tqdm(total=len(val_dataloader), desc="Eval", disable=not accelerator.is_local_main_process, position=0)
    disable_bar = not accelerator.is_local_main_process
    distributed_state = PartialState()

    w = config.flow_guidance_weights[0]
    if accelerator.is_local_main_process:
        logger.info(f"guided by {config.guide_source}")

    disp_metrics = []
    depth_metrics = []
    total = 0
    for i, batch in enumerate(val_dataloader):
        if config.eval_num_batch > 0 and i >= config.eval_num_batch:
            break
        
        normalized_rgbs = batch["normalized_rgb"]
        gt_images = batch["normalized_disp"]
        raw_disps = batch["raw_disp"]
        left_images = batch["left_image"] if "left_image" in batch else None
        right_images = batch["right_image"] if "right_image" in batch else None
        depth_images = batch["depth"] if "depth" in batch else None
        gt_masks = batch["mask"]
        fxb = batch["fxb"]
        sim_disps = batch["sim_disp"] if "sim_disp" in batch else None
        
        B = normalized_rgbs.shape[0]
        # assert not torch.any(gt_images[gt_masks.to(torch.bool)] == 0.0), "dataset bug"
        if config.guide_source is None:
            pass

        elif config.guide_source == "raft-stereo":
            pass

        elif config.guide_source == "stereo-match":
            pass

        elif config.guide_source == "raw-depth":
            guidance_image = batch["raw_depth"] # raw
            valid = guidance_image > 0

        elif config.guide_source == "gt":
            guidance_image = batch["depth"] # gt
            valid = guidance_image > 0
        else:
            raise ValueError(f"Unknown guidance mode: {config.guide_source}")

        if config.guide_source is not None:
            pipeline.guidance.prepare(guidance_image, valid, "depth") # disp
            pipeline.guidance.flow_guidance_weight = w

        pred_disps, metrics_, uncertainties, error, intermediates = eval_batch(config, pipeline, disable_bar,  fxb, normalized_rgbs, 
                                                                               raw_disps, gt_masks, left_images, right_images, sim_disps)
        metrics = metrics_to_dict(*metrics_)
        logger.info(f"metrics(w={w}):{pretty_json(metrics)}")

        disp_err = torch.from_numpy(metrics_[0]).to(distributed_state.device) # to be gathered
        depth_err = torch.from_numpy(metrics_[1]).to(distributed_state.device)

        if config.plot_error_map:
            fname = lambda name: f"{eval_output_dir}/idx{i}_w{w}_pid{distributed_state.process_index}_{name}"
            error_map = plot_error_map(error)
            error_map.save(fname("error.png"))
        
        # gather all batch results
        gathered_disp_err = accelerator.gather_for_metrics(disp_err)
        gathered_depth_err = accelerator.gather_for_metrics(depth_err)

        disp_metrics.extend(gathered_disp_err) 
        depth_metrics.extend(gathered_depth_err)
        total += gathered_disp_err.shape[0]

        pbar.update(1)

    # whole val set results
    gathered_metrics = metrics_to_dict(torch.vstack(disp_metrics).cpu().numpy(), torch.vstack(depth_metrics).cpu().numpy())
    logger.info(f"final metrics:{pretty_json(gathered_metrics)}")
    logger.info(f"total evaluated {total} samples, please check if correct")

if __name__ == "__main__":
    setup_hydra_configurations()
    run_distributed_eval()
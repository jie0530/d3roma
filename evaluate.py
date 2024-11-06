import os, sys
import cv2
import math
import argparse
import torch
import logging
import random
from tqdm import tqdm
from data.stereo_datasets import *
from data.mono_datasets import *

from accelerate import Accelerator
from core.stereo_matching import StereoMatching
from core.guidance import FlowGuidance
import numpy as np
from utils.utils import seed_everything, make_image_grid__, debug_stat
from config import TrainingConfig, create_sampler, setup_hydra_configurations
from diffusers import UNet2DModel, UNet2DConditionModel
from utils.utils import RunningAverager, compute_errors, metrics_to_dict, compute_scale_and_shift, normalize_rgb, pretty_json, ensemble_depths, viz_cropped_pointcloud
from accelerate.logging import get_logger
from utils.camera import Realsense, plot_uncertainties, plot_denoised_images, plot_error_map
from utils.utils import InputPadder
import matplotlib.pyplot as plt
import hydra
from config import Config

logger = get_logger(__name__, log_level="INFO")

def evaluate_intermediate_metrics(config, fxb, intermediates, gt_disps, gt_masks, gt_indexes):
    """ intermediates: B,T,H,W 
        gt_disp: B,1,H,W
        gt_masks: B,1,H,W
    """
    pred_origs = intermediates.images_pred_orig.cpu()
    B,N,H,W = pred_origs.shape
    gt_disps = gt_disps.repeat(1, N, 1, 1)
    # gt_masks = gt_masks.repeat(1, N, 1, 1)
    
    if fxb is not None:
        fxb = fxb.view(B,1).repeat(B, N, 1)

    if config.eval_dataset[0] == "SceneFlow":
        scale = 960. / float(config.camera_resolution.split("x")[0])
        min_, max_ = 0.5/scale, 192.0/scale
        mask = (pred_origs >= min_) & (pred_origs < max_) & (gt_masks.repeat(1, N, 1, 1).bool())
        pred_origs = pred_origs.clamp(min_, max_)
    else:
        pred_origs = pred_origs.clamp(min=0.25) # prevent 0 disparities
        mask = gt_masks.repeat(1, N, 1, 1).bool()

    disp_metrics, depth_metrics = compute_errors(gt_disps.view(-1,H,W).numpy(), 
                             pred_origs.view(-1,H,W).numpy(), 
                             config.prediction_space,
                             mask.view(-1,H,W).numpy().astype(bool), 
                             fxb.view(-1,1).cpu().numpy() if fxb is not None else None)
    disp_metrics = np.reshape(disp_metrics, (B, N, -1))
    depth_metrics = np.reshape(depth_metrics, (B, N, -1))
    return disp_metrics, depth_metrics
    
def denormalize(config, pred_disps, raw_disp=None, mask=None):
    from utils.utils import Normalizer
    norm = Normalizer.from_config(config)
    pred_disps_unnormalized = norm.denormalize(pred_disps, raw_disp, mask)

    return pred_disps_unnormalized

def eval_batch(config, pipeline, disable_bar, fxb=None, normalized_rgb=None, raw_disp=None, mask=None, 
            left_image=None, right_image=None, sim_disp=None, raw_depth=None, **kwargs):
    """ raw_disp: unnormalized gt data 
        sim_disp: simulated disp from stereo/rgbd cameras
    """

    image_shape = normalized_rgb.shape if normalized_rgb is not None else left_image.shape
    inputPadder = InputPadder(image_shape, divis_by=config.divis_by)
    normalized_rgb, left_image, right_image, sim_disp, raw_depth = inputPadder.pad(normalized_rgb, left_image, right_image, sim_disp, raw_depth)
    
    denorm = partial(denormalize, config)
    """ # hack here
    if hasattr(pipeline.guidance, 'disp_sm') and pipeline.guidance.disp_sm is not None and \
                tuple(pipeline.guidance.disp_sm.shape[-2:]) != inputPadder.padded_size:
        pipeline.guidance.disp_sm = inputPadder.pad(pipeline.guidance.disp_sm)[0]
        pipeline.guidance.valid_sm = inputPadder.pad_zero(pipeline.guidance.valid_sm)[0]

        # # TODO encode to latent space
        # pipeline.guidance.disp_sm = encode_disp(pipeline.vae, guidance.disp_sm, 0.18215) # B,4,H,W   """

    final_pred_disps = []
    
    assert config.num_inference_rounds > 0, "num_inference_rounds should be greater than 0"

    for r in range(config.num_inference_rounds): # repeat R times
        pipeline.set_progress_bar_config(desc=f"Denoising(w={pipeline.guidance.flow_guidance_weight:.1f}), round={r}", 
                                         disable=disable_bar, 
                                         leave=config.num_inference_rounds>1, 
                                         position=2)
        out = pipeline(normalized_rgb, left_image, right_image, sim_disp, raw_depth, mask,
                num_inference_steps=config.num_inference_timesteps,
                num_intermediate_images=config.num_intermediate_images, # T
                add_noise_rgb=config.noise_rgb,
                depth_channels=config.depth_channels,
                cond_channels=config.cond_channels,
                denorm = denorm
            )
        out.images = inputPadder.unpad(out.images)
        final_pred_disps.append(out.images)
    
    # only visualize the last round results
    out.images_pred_orig = denorm(inputPadder.unpad(out.images_pred_orig), raw_disp, mask)
    out.images_perturbed_orig = denorm(inputPadder.unpad(out.images_perturbed_orig), raw_disp, mask)
    out.images_pred_prev = denorm(inputPadder.unpad(out.images_pred_prev), raw_disp, mask)
    out.images_sampled_prev = denorm(inputPadder.unpad(out.images_sampled_prev), raw_disp, mask)
    out.images_purturbed_pred_prev = denorm(inputPadder.unpad(out.images_purturbed_pred_prev), raw_disp, mask)

    pred_disps = torch.concat(final_pred_disps, dim=1)  # [B,R,H,W]
    # pred_disps = (pred_disps + 1.0) * 0.5 # -> ~ [0,1], more numeraically stable when doing ssi 
    # pred_disps = pred_disps + 0.5 # (pred_disps + 1.0) * 0.5 # -> ~ [0,1], actually, it does not matter when doing ssi 

    if config.ensemble:
        # pred_disps = pred_disps.mean(dim=1, keepdim=True)
        aligned_images = [ensemble_depths(pred_disps[i])[0] for i in range(pred_disps.shape[0])]
        pred_disps = torch.stack(aligned_images).unsqueeze(1)
    
    pred_disps_unnormalized = denormalize(config, pred_disps, raw_disp, mask)
    gt_disp_unnormalized = raw_disp.squeeze(1).cpu().numpy() # b,h,w
    pred_disps_unnormalized = torch.mean(pred_disps_unnormalized, dim=1).cpu().numpy() # b,h,w
    
    if config.eval_dataset[0] == "SceneFlow":
        scale = 960. / float(config.camera_resolution.split("x")[0])
        min_, max_ = 0.5/scale, 192.0/scale
        mask = (gt_disp_unnormalized >= min_) & (gt_disp_unnormalized < max_)
        pred_disps_unnormalized = np.clip(pred_disps_unnormalized, min_, max_) # for numerical stability
    else:
        mask = mask.squeeze(1).cpu().numpy().astype(bool)  # b,h,w

    metrics = compute_errors(gt_disp_unnormalized, 
                             pred_disps_unnormalized,
                             config.prediction_space,
                             mask, 
                             fxb.cpu().numpy() if fxb is not None else None)

    if pred_disps.shape[1] // config.depth_channels > 1:
        uncertainties = np.zeros_like(gt_disp_unnormalized)
        uncertainties[mask] = np.std(pred_disps.cpu().numpy(), axis=1)[mask]
    else:
        uncertainties = None

    error_map = np.zeros_like(gt_disp_unnormalized)
    error_map[mask] = np.abs(gt_disp_unnormalized[mask] - pred_disps_unnormalized[mask]) # B,H,W
    return pred_disps_unnormalized, metrics, uncertainties, error_map, out

def eval_sliced_batch(pipeline, config, mini_batch, stat, eval_output_dir, abnoraml_indexes, i, j):
    print(f"mini_batch_size:", mini_batch["normalized_disp"].shape)

    normalized_rgbs = mini_batch["normalized_rgb"].cuda() if "normalized_rgb" in mini_batch else None
    normalized_disps = mini_batch["normalized_disp"].cuda()
    raw_disps = mini_batch["raw_disp"].cuda()
    left_images = mini_batch["left_image"].cuda() if "left_image" in mini_batch else None
    right_images = mini_batch["right_image"].cuda() if "right_image" in mini_batch else None
    depth_images = mini_batch["depth"].cuda() if "depth" in mini_batch else None
    gt_masks = mini_batch["mask"].cuda()
    raw_depth = mini_batch["raw_depth"].cuda() if "raw_depth" in mini_batch else None
    sim_disps = mini_batch["sim_disp"].cuda() if "sim_disp" in mini_batch else None
    fxb = mini_batch["fxb"].cuda() if "fxb" in mini_batch else None
    
    B = normalized_disps.shape[0]

    # assert not torch.any(gt_images[gt_masks.to(torch.bool)] == 0.0), "dataset bug"
    for p, fidx in zip(mini_batch["path"], mini_batch['index']):
        logger.info(f"i={i}, index={fidx.item()}, file path={p}")

    # logger.info(f"guided by {config.guide_source}")
    if config.guide_source is None:
        pass

    elif config.guide_source == "raw-depth":
        
        assert raw_depth is not None, f"guide source not exists"
        assert np.sum(config.flow_guidance_weights) > 0, f"guidance source is set to {config.guide_source}, but no weights!"

    else:
        raise ValueError(f"Unknown guidance mode: {config.guide_source}")
    
    logger.info("guided by %s" % config.guide_source)

    # if config.guide_source is not None:
    #     pipeline.guidance.prepare(guidance_image, valid, "depth") # config.prediction_space

    inter_metrics = {}
    for w in config.flow_guidance_weights:
        pipeline.guidance.flow_guidance_weight = w
        logger.info(f"flow_guidance_weight={w}")
        if w > 0:
            logger.info(f"flow_guidance_mode={config.flow_guidance_mode}")

        pred_disps, metrics_, uncertainties, error, intermediates = eval_batch(config, pipeline, False, fxb, 
                                                                            normalized_rgbs, raw_disps, gt_masks, 
                                                                            left_images, right_images, sim_disps, raw_depth)
        if False: # FOR DEBUG ONLY
            accuracy = np.array([f"{dep[2]:.3f}" for dep in metrics_[1]])
            logger.info("-"*10)
            logger.info(accuracy)
            for j, a in enumerate(accuracy):
                if float(a) < 0.86:
                    logger.info(f"anomal detected (index={mini_batch['index'][j]}): {mini_batch['path'][j]}")
                    ind = mini_batch['index'][j].item()
                    abnoraml_indexes[ind] = float(a)
            logger.info([p])
            logger.info("-"*10)

        metrics = metrics_to_dict(*metrics_)
        logger.info(f"metrics(w={w}):{pretty_json(metrics)}")
        running_avg, anomal = stat.append(metrics, w)

        fname = lambda name: f"{eval_output_dir}/idx{i}_{j}_s.{config.guide_source}_m.{config.flow_guidance_mode[:3]}_w{w}_{name}"
        if uncertainties is not None:
            var = plot_uncertainties(uncertainties)
            var.save(fname("var.png"))

        error_map = plot_error_map(error)
        error_map.save(fname("error.png"))

        if config.plot_denoised_images:
            grid = plot_denoised_images(config, 
                                        intermediates, 
                                        pred_disps, 
                                        **mini_batch)
            grid.save(fname("denoise.png")) # x_{t-1}

        if config.write_pcd:
            instrinsic = mini_batch["K"].cuda()
            device = mini_batch["device"]

            # depth_images[gt_masks] = 0.0
            for b in range(B):
                camera = DepthCamera.from_device(device[b])
                H, W = normalized_rgbs[b].shape[-2:]
                camera.change_resolution(f"{W}x{H}")
                assert camera.resolution == (H, W)

                K = instrinsic[b].cpu().numpy()
                rgb = ((normalized_rgbs[b].cpu().permute(1,2,0).numpy()+1.0)*127.5).astype(np.uint8).clip(0,255)
                gt_depth_np = depth_images[b,0].cpu().numpy() # [H,W]
                gt_masks_np = gt_masks[b,0].cpu().numpy().astype(bool)
                gt_depth_np[~gt_masks_np] = 0.0
                gt_depth_np = camera.transform_depth_to_rgb_frame(gt_depth_np) #if not alreay aligned
                viz_cropped_pointcloud(K, rgb, gt_depth_np, fname=fname(f"b{b}_gt.ply"))

                if config.prediction_space == "disp":
                    pred_depth = np.zeros_like(pred_disps[b])
                    pred_mask_np = (pred_disps[b] > camera.min_disp) & (pred_disps[b] < camera.max_disp)  # np.ones_like(pred_depth).astype(bool)# gt_masks[b,0].cpu().numpy().astype(bool)
                    pred_depth[pred_mask_np] = fxb[b].cpu().numpy() / pred_disps[b][pred_mask_np] # [H,W]

                    # hack for simulation
                    if config.eval_dataset[0] == "HssdIsaacStd" and config.eval_split == "simulation2":
                        cv2.imwrite(f"datasets/HssdIsaacStd/simulation2/{mini_batch['index'][b]:04d}_pred.exr", pred_depth)

                elif config.prediction_space == "depth":
                    pred_mask_np = (pred_disps[b] > camera.min_depth) & (pred_disps[b] < camera.max_depth)
                    pred_depth = pred_disps[b]
                
                pred_depth = camera.transform_depth_to_rgb_frame(pred_depth)
                viz_cropped_pointcloud(K, rgb, pred_depth, fname=fname(f"b{b}_pred.ply"))

        if config.plot_intermediate_metrics:
            inter_metrics[w] = evaluate_intermediate_metrics(config, fxb, intermediates, mini_batch["raw_disp"], mini_batch["mask"], mini_batch["index"])

    # plot intermediate metricss
    if config.plot_intermediate_metrics:
        for b in range(B):
            figure, axis = plt.subplots(2, 2, figsize=(12, 8))
            figure.tight_layout(pad=4)
            for w, (disp_metrics, depth_metrics) in inter_metrics.items():
                # x = np.arange(0, config.num_inference_timesteps, config.num_inference_timesteps // disp_metrics.shape[1])
                x = np.arange(0, disp_metrics.shape[1])

                epe = disp_metrics[b, :, 0]
                a1 = depth_metrics[b, :, 0]
                rmse = depth_metrics[b, :, 3] 
                mae = depth_metrics[b, :, 5]

                axis[0, 0].plot(x, epe, label=f"w={w:.1f}")
                axis[0, 0].set_title("Disparity EPE (↓)")
                axis[0, 1].plot(x, a1, label=f"w={w:.1f}")
                axis[0, 1].set_title("Depth a1 (↑)")
                axis[1, 0].plot(x, rmse, label=f"w={w:.1f}")
                axis[1, 0].set_title("Depth RMSE (↓)")
                axis[1, 1].plot(x, mae, label=f"w={w:.1f}")
                axis[1, 1].set_title("Depth MAE (↓)")

            for ax in axis.flat:
                ax.set(xlabel='T - t')
                ax.legend(loc="upper right")
        
            metric_fname = lambda i, b, fidx, j: f"{eval_output_dir}/idx{i}_{j}_b{b}_file{fidx}_g{config.flow_guidance_mode[:4]}.w{w}_metrics.png"
            figure.savefig(metric_fname(i, b, mini_batch['index'][b].item(), j))
            plt.close(figure)

    return pred_disps, metrics_, running_avg, anomal

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def run_eval(base_cfg: Config):
    if base_cfg.seed != -1:
        seed_everything(base_cfg.seed) # for reproducing
    
    Accelerator() # hack: enable logging

    config = base_cfg.task
    assert len(config.eval_dataset) == 1, "only support single dataset for evaluation"

    patrained_path = f"{config.resume_pretrained}"
    if os.path.exists(patrained_path):
        logger.info(f"load weights from {patrained_path}")
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
        if config.ssi:
            ssi_str = "safe_ssi" if config.safe_ssi else "ssi"
        else:
            ssi_str = "no_ssi"
        eval_output_dir = f"{config.resume_pretrained}/{config.eval_dataset[0]}.g.{config.guide_source}.b{config.eval_batch_size}.{config.eval_split}.{ssi_str}"

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)

    logger.logger.addHandler(logging.FileHandler(f"{eval_output_dir}/eval.log"))
    # logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"eval output dir: {eval_output_dir}")
    logger.info('RUN ========================================')
    logger.info(' '.join(sys.argv))
    logger.info('END ========================================')

    assert len(config.eval_dataset) == 1

    from data.data_loader import create_dataset
    val_dataset = create_dataset(config, config.eval_dataset[0], split = config.eval_split)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config.eval_batch_size,
                                                shuffle=True,
                                                pin_memory=False, 
                                                drop_last=False)
    
    stat = RunningAverager(horizon=5)
    # detect_anomaly={
    #     "disp_epe": 50,
    #     "depth_mae": 50
    # }
    abnoraml_indexes = {}

    pbar = tqdm(val_dataloader, desc="Eval", position=0)
    for i, batch in enumerate(pbar):
        if config.eval_num_batch > 0 and i >= config.eval_num_batch:
            break

        if config.coarse_to_fine:
            pred_disps = np.empty((batch["raw_disp"].shape[0], batch["raw_disp"].shape[2], batch["raw_disp"].shape[3]))
            metrics = []
            for j in range(4):
                mini_batch = {}
                for key, val in batch.items():
                    if key == "raw_disp" \
                        or key == "normalized_disp" \
                        or key == "sim_disp" \
                        or key == "left_image" \
                        or key == "right_image" \
                        or key == "mask":

                        if j == 0:
                            mini_batch[key] = val[:,:,0:270,0:480]
                        elif j == 1:
                            mini_batch[key] = val[:,:,270:,0:480]
                        elif j == 2:
                            mini_batch[key] = val[:,:,0:270,480:]
                        elif j == 3:
                            mini_batch[key] = val[:,:,270:,480:]
                    else:
                        mini_batch[key] = val

                
                pred_disps_mini, metrics_mini, running_avg, anomal = eval_sliced_batch(pipeline, config, mini_batch, stat, eval_output_dir, abnoraml_indexes, i, j)
                if j == 0:
                    pred_disps[:, 0:270, 0:480] = pred_disps_mini
                elif j == 1: 
                    pred_disps[:, 270:, 0:480] = pred_disps_mini
                elif j == 2:
                    pred_disps[:, 0:270, 480:] = pred_disps_mini
                elif j == 3:
                    pred_disps[:, 270:, 480:] = pred_disps_mini
                # pred_disps.append(pred_disps_mini)
                metrics.append(metrics_mini[0])
            metrics_ = np.mean(metrics, axis=0)

            def viz_normalizer(x, i, apply_mask=True, low_p=0, high_p=100):
                # assert x.min() >= 0.0, "bug"
                valid =  x > 0.0
                low, high = np.percentile(x[valid], (low_p, high_p))
                x[valid] = (x[valid] - low) / (high - low + 1e-10) 
                x[~valid] = 0.0
                return x
            gray_to_jet = lambda x:  x #(cmap_jet(x/255.0)*255.)[...,:3].astype(np.uint8)
            depth_to_grayscale = lambda x, i: (viz_normalizer((x+1)/2, i, apply_mask=False, low_p=2, high_p=98).clip(0, 1) * 255.0).astype(np.uint8) # from ~[-1,1] -> [0,255]
            for k in range(pred_disps.shape[0]):
                Image.fromarray(gray_to_jet(depth_to_grayscale(pred_disps[k], i))).save(f"{eval_output_dir}/idx{i}_{k}_pred_disp.png")
        else:
            pred_disps, metrics_, running_avg, anomal = eval_sliced_batch(pipeline, config, batch, stat, eval_output_dir, abnoraml_indexes, i, 0)   

        if anomal > 0.:
            logger.warning(f"Anomal sample detected! ratio={anomal}, metric={metrics_}")
            pbar.set_description(f"Eval: anomaly detected!")
        else:
            # pbar.set_description(f"Eval: a3={running_avg['depth_a3']:.2f}, rel={running_avg['depth_rel']:.4f}")
            pbar.set_description(f"Eval: epe={running_avg['disp_epe']:.2f}") #, mae={running_avg['depth_mae']:.4f}

        if (i+1) % 10 == 0:
            logger.info(pretty_json(stat.dump()))

    if i > 1: 
        logger.info(pretty_json(stat.dump()))
        logger.info("abnormal indexes: ")
        logger.info(abnoraml_indexes)

if __name__ == "__main__":
    setup_hydra_configurations()
    run_eval()
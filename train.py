import os
import sys
import torch
import logging
import numpy as np
from core.scheduler_ddpm import MyDDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import (
    get_cosine_schedule_with_warmup, 
    get_constant_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from dataclasses import asdict
from diffusers import UNet2DModel, DDIMScheduler
from functools import partial
from accelerate import Accelerator, PartialState
from tqdm.auto import tqdm
from data.data_loader import fetch_dataloader
from torch.optim.lr_scheduler import ConstantLR
from core.custom_pipelines import GuidedLatentDiffusionPipeline, GuidedDiffusionPipeline
from core.guidance import FlowGuidance
from core.resample import create_named_schedule_sampler
from accelerate.logging import get_logger
from evaluate import eval_batch
from utils.utils import pyramid_noise_like, flatten, pretty_json, metrics_to_dict, InputPadder
from utils.camera import plot_uncertainties, plot_denoised_images, plot_error_map, plot_loss_terms
from utils.losess import mse_to_vlb
from config import TrainingConfig, create_sampler
from utils.utils import seed_everything
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
import random
from PIL import Image
import hydra
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from config import get_output_dir, set_debug, setup_hydra_configurations
from config import Config, TrainingConfig

logger = get_logger(__name__, log_level="INFO") # multi-process logging

class StepCounter:
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.loss_history = np.zeros([100], dtype=np.float32)
        self.loss_count = 0
        self.min_eval = np.inf

    def save_step(self, epoch, global_step, local_step):
        self.epoch = epoch
        self.global_step = global_step
        self.local_step = local_step

    def step_eval(self, eval_metric):
        if eval_metric < self.min_eval:
            self.min_eval = eval_metric
            return True
        else:
            return False

    def queue_loss(self, loss):
        if self.loss_count == 100:
            self.loss_history[:-1] = self.loss_history[1:]
            self.loss_history[-1] = loss
        else:
            self.loss_history[self.loss_count] = loss
            self.loss_count += 1

    def state_dict(self):
        return dict(epoch=self.epoch, 
                    global_step=self.global_step, 
                    local_step=self.local_step, 
                    loss_count=self.loss_count, 
                    loss_history=self.loss_history.tolist(),
                    min_eval=self.min_eval)
    
    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.local_step = state_dict['local_step']
        self.loss_count = state_dict['loss_count']
        self.loss_history = np.array(state_dict['loss_history'])
        self.min_eval = state_dict["min_eval"]

    def avg_loss(self):
        return np.inf if self.loss_count < 100 else np.sum(self.loss_history) / self.loss_count

def create_pipeline(accelerator, config, model, vae=None, tokenizer=None, text_encoder=None):
    flow_guidance = FlowGuidance(config.flow_guidance_weights[0], config.perturb_start_ratio, config.flow_guidance_mode)
    if config.ldm:
        ddim = DDIMScheduler.from_config(dict(
            beta_schedule = config.beta_schedule,
            beta_start = config.beta_start,
            beta_end = config.beta_end,
            clip_sample = config.clip_sample,
            num_train_timesteps = config.num_train_timesteps,
            prediction_type = config.prediction_type,
            set_alpha_to_one = False,
            skip_prk_steps = True,
            steps_offset = 1,
            trained_betas = None
        ))
        return GuidedLatentDiffusionPipeline(unet=accelerator.unwrap_model(model), 
                                    vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
                                    scheduler=ddim, guidance=flow_guidance)
    else:
        scheduler = create_sampler(config, train=False)
        return GuidedDiffusionPipeline(unet=accelerator.unwrap_model(model), 
                                scheduler=scheduler, #noise_scheduler_infer,
                                guidance=flow_guidance)

def __encode_empty_text(tokenizer, text_encoder):
    """
    Encode text embedding for empty prompt
    """
    prompt = ""
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    return text_encoder(text_input_ids)[0]

def encode_disp(vae, x, depth_latent_scale_factor):
    """ x: B,1,H,W 
        output: B,4,H/f,W/f
    """
    disp_in = x.repeat(1,3,1,1)
    return encode_rgb(vae, disp_in, depth_latent_scale_factor)

def encode_rgb(vae, x, rgb_latent_scale_factor):
    """
    Encode RGB image into latent.

    Args:
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.

    Returns:
        `torch.Tensor`: Image latent.
    """
    # encode
    h = vae.encoder(x)
    moments = vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    # scale latent
    rgb_latent = mean * rgb_latent_scale_factor
    return rgb_latent

def train_step(accelerator, config: TrainingConfig, model, optimizer, lr_scheduler, noise_scheduler, t_sampler, step_counter,
               normalized_disp = None, normalized_rgb=None, mask=None, left_image=None, right_image=None, sim_disp=None, 
               tokenizer=None, text_encoder=None, vae=None,
               empty_text_embed = None, 
               **kwargs):
    """ normalized_disp: B,1,H,W 
        normalized_rgb: B,C,H,W
        mask: B,1,H,W
        left_image/right_image: B,3,H,W
        sim_disp: B,1,H,W  simulated depth from stereo/rgbd cameras 
    """
    if config.clip_sample and (normalized_disp.max() > config.clip_sample_range or normalized_disp.min() < -config.clip_sample_range):
        # logger.warning(f"out of clip range: max={normalized_disp.max()}, min={normalized_disp.min()}")
        normalized_disp = torch.clamp(normalized_disp, -config.clip_sample_range, config.clip_sample_range)
        mask = mask * (normalized_disp.abs() < config.clip_sample_range).to(torch.float32)

    if not config.ldm:
        assert normalized_disp.shape[1] == config.depth_channels, "depth channel inconsistency"

    losses = {}
    normalized_disp_ori = normalized_disp

    inputPadder = InputPadder(normalized_disp.shape, divis_by=config.divis_by)
    # print("L96:", normalized_disp.shape)
    normalized_disp, normalized_rgb, left_image, right_image, sim_disp = inputPadder.pad(normalized_disp, normalized_rgb, left_image, right_image, sim_disp)
    mask = inputPadder.pad_zero(mask)[0]

    if config.ldm: 
        def __decode_depth(depth_latent, depth_latent_scale_factor=0.18215):
            """
            Decode depth latent into depth map.

            Args:
                depth_latent (`torch.Tensor`):
                    Depth latent to be decoded.

            Returns:
                `torch.Tensor`: Decoded depth map.
            """
            # scale latent
            depth_latent = depth_latent / depth_latent_scale_factor
            # decode
            z = vae.post_quant_conv(depth_latent)
            stacked = vae.decoder(z)
            # mean of output channels
            depth_mean = stacked.mean(dim=1, keepdim=True)
            return depth_mean
    
        """ move to latent space """
        # normalized_rgb = (normalized_rgb + 1) * 0.5 # rescale to [0,1] before going to latent space
        # logger.warning(f"{normalized_rgb.max().item()}, {normalized_rgb.mean().item()}, {normalized_rgb.min().item()}, {normalized_rgb.std().item()}")
        
        if left_image is not None:
            left_image_latent = encode_rgb(vae, left_image, 0.18215)
        else:
            left_image_latent = None

        if right_image is not None:
            right_image_latent = encode_rgb(vae, right_image, 0.18215)
        else:
            right_image_latent = None
        
        # print("L105:", normalized_rgb.shape)
        # normalized_disp = (normalized_disp + 1) * 0.5  # rescale to [0,1] before going to latent space
        # logger.warning(f"{normalized_disp.max().item()}, {normalized_disp.mean().item()}, {normalized_disp.min().item()}, {normalized_disp.std().item()}")
        normalized_disp_latent = encode_disp(vae, normalized_disp, 0.18215) # B,4,H,W  
        
        if sim_disp is not None:
            sim_disp_latent = encode_disp(vae, sim_disp, 0.18215) # B,4,H,W
        else:
            sim_disp_latent = None

    left_image = left_image_latent if config.ldm else left_image
    right_image = right_image_latent if config.ldm else right_image

        # # sanity check
        # decode_disp = __decode_depth(normalized_disp_latent, 0.18215) 
        # decode_disp = decode_disp.clamp(-1,1)
        # if (err:=torch.abs(decode_disp - normalized_disp)).mean() >= 2e-2:
        #     logger.warning(f"abnormal disp encoder/decoder mae: {err.mean().item()}")
        #     logger.warning(f"{err.max().item()}, {err.mean().item()}, {err.min().item()}, {err.std().item()}")

    if config.loss_type == "l1":
        loss_fn = nn.L1Loss(reduction='none')
    elif config.loss_type == "mse":
        loss_fn = nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"loss type {config.loss_type} not supported")

    # sample epsilon
    if config.noise_strategy == 'pyramid':
        noise = pyramid_noise_like(normalized_disp_latent if config.ldm else normalized_disp)

    elif config.noise_strategy == 'randn':
        noise = torch.randn(normalized_disp_latent.shape if config.ldm else normalized_disp.shape).to(normalized_disp.device)
    else:
        raise NotImplementedError
    
    bs = noise.shape[0]

    # sample t
    """ timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bs,), device=normalized_disp.device
    ).long() # T """
    timesteps, weights = t_sampler.sample(bs, noise.device)
    
    x_clean = normalized_disp_latent if config.ldm else normalized_disp
    cond_sim = sim_disp_latent if config.ldm else sim_disp
    
    if "rgb" in config.cond_channels:
        if config.ldm:
            normalized_rgb_latent = encode_rgb(vae, normalized_rgb, 0.18215) # B,4,H,W 
        cond_rgb = normalized_rgb_latent if config.ldm else normalized_rgb
        if config.noise_rgb:
            noise_rgb = torch.randn(cond_rgb.shape).to(cond_rgb.device)
            noisy_rgb = noise_scheduler.add_noise(cond_rgb, noise_rgb, timesteps)
            final_rgb = cond_rgb * 0.5 + noisy_rgb * 0.5
        else:
            final_rgb = cond_rgb
    
    # forward diffusion process
    noisy_images = noise_scheduler.add_noise(x_clean, noise, timesteps) # B,1,H,W
    # q_means, _ = noise_scheduler.posterior_mean_variance(x_clean, noisy_images, timesteps)
    
    abnormal_loss_detected = False
    with accelerator.accumulate(model):
        if config.cond_channels == "rgb":
            inputs = torch.cat([noisy_images, final_rgb], dim=1)
        elif config.cond_channels == "rgb+raw":
            inputs = torch.cat([noisy_images, final_rgb, cond_sim], dim=1)
        elif config.cond_channels == "left+right":
            inputs = torch.cat([noisy_images, left_image, right_image], dim=1)
        elif config.cond_channels == "left+right+raw": # 8
            inputs = torch.cat([noisy_images, left_image, right_image, cond_sim], dim=1)
        elif config.cond_channels == "rgb+left+right":
            inputs = torch.cat([noisy_images, final_rgb, left_image, right_image], dim=1)
        elif config.cond_channels == "rgb+left+right+raw":
            inputs = torch.cat([noisy_images, final_rgb, left_image, right_image, cond_sim], dim=1)
        else:
            raise NotImplementedError
        
        with accelerator.autocast():
            if config.ldm:
                # Batched empty text embedding
                if empty_text_embed is None:
                    empty_text_embed = __encode_empty_text(tokenizer, text_encoder)
                empty_text_embed = empty_text_embed.repeat(
                    (noisy_images.shape[0], 1, 1)
                )
                model_output = model(inputs, timesteps, empty_text_embed, return_dict=False)[0]
            else:
                model_output = model(inputs, timesteps, return_dict=False)[0]

            if config.prediction_type == "epsilon":
                network_loss = loss_fn(model_output * mask, noise * mask) # SNR weighting, diff-13 sec 4
                # p_means, _ = noise_scheduler.posterior_mean_variance(normalized_disp, model_output, timesteps)
            elif config.prediction_type == "v_prediction":
                v = noise_scheduler.get_velocity(x_clean, noise, timesteps) # diff-22 section 2.4
                
                # # print(mask.shape)
                # # print(normalized_disp_ori.shape)
                # network_loss = loss_fn(pred_disp * mask, normalized_disp_ori * mask)

                network_loss = loss_fn(model_output, v)
                # p_means, _ = noise_scheduler.p_mean_variance(model_output, noisy_images, timesteps)

                pred_v = model_output.detach()

                # debug
                def extract(arr, indices):
                    return arr[indices]

                alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                alpha_prod_t = extract(alphas_cumprod, timesteps).view(bs, 1, 1, 1)
                beta_prod_t = extract(1 - alphas_cumprod, timesteps).view(bs, 1, 1, 1)
                x0_latent = (alpha_prod_t**0.5) * noisy_images - (beta_prod_t**0.5) * v
                x0_hat_latent = (alpha_prod_t**0.5) * noisy_images - (beta_prod_t**0.5) * pred_v 
                
                assert (x0_latent - normalized_disp_latent).max() < 1e-4

                # x0 = __decode_depth(x0_latent, 0.18215) # B,1,H,W # rescaled to [0, 1] before decoding, now pred_disp is in [0,1]
                # x0 = x0.clamp(0, 1)
                # Image.fromarray((x0[0]/x0[0].max()*255).cpu().numpy().astype(np.uint8).transpose(1,2,0)[...,0]).save("disp_de_x0.png")
                # Image.fromarray(((normalized_rgb[0].cpu().numpy()+1)*127.5).astype(np.uint8).transpose(1,2,0)).save("rgb_aug.png")
                # Image.fromarray(mask[0].cpu().numpy().astype(np.uint8).transpose(1,2,0)[...,0]*255).save("mask_aug.png")
                # Image.fromarray(((normalized_disp_ori[0].cpu().numpy()+1)*127.5).astype(np.uint8).transpose(1,2,0)[...,0]).save("disp_x0.png")

                x0_hat = __decode_depth(x0_hat_latent, 0.18215) # B,1,H,W # rescaled to [0, 1] before decoding, now pred_disp is in [0,1]
                x0_hat = x0_hat.clamp(-1, 1)
                # Image.fromarray((x0_hat[0]/x0_hat[0].max()*255).cpu().numpy().astype(np.uint8).transpose(1,2,0)[...,0]).save("pred_disp.png")
                losses['recon_mae'] = torch.abs(x0_hat - normalized_disp).mean()

                # log images FIXME NOT TESTED
                global_step = step_counter.global_step
                if (accelerator.is_main_process and \
                    ((global_step < 1000 and (global_step+1) % 100 == 0) or \
                    (global_step < config.val_every_global_steps and (global_step+1) % (config.val_every_global_steps//10) == 0) or \
                    (global_step+1) % (config.val_every_global_steps//4) == 0)):
                    from utils.utils import Normalizer
                    norm = Normalizer.from_config(config)
                    # TensorBoardTracker
                    accelerator.get_tracker("tensorboard").log_images({
                        "train/sample_pred": x0_hat[:8,...].detach().repeat(1,3,1,1).cpu().numpy(),
                        "train/sample_gt": normalized_disp_ori[:8,...].repeat(1,3,1,1).cpu().numpy()
                    }, global_step)

            elif config.prediction_type == "v_pred_depth":
                # v = noise_scheduler.get_velocity(normalized_disp_latent, noise, timesteps) # diff-22 section 2.4
                pred_v = model_output

                # debug
                def extract(arr, indices):
                    return arr[indices]

                alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                alpha_prod_t = extract(alphas_cumprod, timesteps).view(bs, 1, 1, 1)
                beta_prod_t = extract(1 - alphas_cumprod, timesteps).view(bs, 1, 1, 1)
                # x0_latent = (alpha_prod_t**0.5) * noisy_images - (beta_prod_t**0.5) * v
                x0_hat_latent = (alpha_prod_t**0.5) * noisy_images - (beta_prod_t**0.5) * pred_v 
                
                # assert (x0_latent - normalized_disp_latent).max() < 1e-4
                x0_hat = __decode_depth(x0_hat_latent, 0.18215) # B,1,H,W # rescaled to [0, 1] before decoding, now pred_disp is in [0,1]
                network_loss = loss_fn(x0_hat * mask, normalized_disp_ori * mask)
            
                losses['recon_mae'] = torch.abs(x0_hat.clamp(-1, 1).detach() - normalized_disp).mean()
                
            elif config.prediction_type == "sample":
                # network_loss = loss_fn(model_output * mask, normalized_disp * mask)
                channel_weights = [1, 1, 1]
                network_loss = 0
                
                for c_ in range(config.depth_channels):
                    network_loss += loss_fn(model_output[:,c_:c_+1] * mask, normalized_disp[:,c_:c_+1] * mask) * channel_weights[c_]
                
                network_loss /= config.depth_channels
                # p_means = noise_scheduler.add_noise(model_output, noise, timesteps) )
                
                global_step = step_counter.global_step
                if (accelerator.is_main_process and \
                    ((global_step < 1000 and (global_step+1) % 100 == 0) or \
                    (global_step < config.val_every_global_steps and (global_step+1) % (config.val_every_global_steps//10) == 0) or \
                    (global_step+1) % (config.val_every_global_steps//4) == 0)):
                    from utils.utils import Normalizer
                    norm = Normalizer.from_config(config)
                    # TensorBoardTracker
                    if config.ssi: 
                        sample_pred = model_output[:8,...].detach().repeat(1,3,1,1).cpu().numpy() / norm.s + norm.t
                        sample_gt = normalized_disp[:8,...].repeat(1,3,1,1).cpu().numpy() / norm.s + norm.t
                    else:
                        sample_pred  = norm.denormalize(model_output[:8,...].detach())
                        sample_gt = norm.denormalize(normalized_disp[:8,...])
                        sample_pred = sample_pred / torch.amax(sample_pred, keepdim=True, dim=list(range(1,len(sample_pred.shape))))
                        sample_gt = sample_gt / torch.amax(sample_gt, keepdim=True, dim=list(range(1,len(sample_pred.shape))))
                    accelerator.get_tracker("tensorboard").log_images({ 
                        "train/sample_pred": sample_pred,
                        "train/sample_gt": sample_gt
                    }, global_step)

            else:
                raise NotImplementedError
        
            loss = (network_loss * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).mean() 
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                breakpoint()

            losses['loss'] = network_loss.mean(list(range(1, len(normalized_disp.shape)))).detach()
            losses['mse'] = (losses['loss'] ** 2 if config.loss_type == "l1" else losses['loss'])
            # losses['vlb'] = [mse_to_vlb(t, mse, noise_scheduler.posterior_log_variance_clipped) for t, mse in enumerate(zip(timesteps, losses['mse']))]

            # gather training losses and metrics
            gathered_timesteps = accelerator.gather_for_metrics(timesteps)
            gathered_losses = accelerator.gather_for_metrics(losses)
            accelerator.wait_for_everyone()

            if loss.item() / step_counter.avg_loss() > 100: # or other cases
                logger.warning("-"*20, main_process_only=False)
                logger.warning(f"->weird loss {loss.item()} !", main_process_only=False)
                logger.warning(f"global_step={global_step}", main_process_only=False)
                logger.warning(f"losses:{losses['loss'].cpu()}", main_process_only=False)
                logger.warning(f"loss_history:{step_counter.loss_history}", main_process_only=False)
                logger.warning(f"loss_count={step_counter.loss_count}", main_process_only=False)
                logger.warning(kwargs["path"], main_process_only=False)
                logger.warning(kwargs["index"], main_process_only=False)
                logger.warning(f"lr={lr_scheduler.get_last_lr()[0]}", main_process_only=False)
                logger.warning("-"*20, main_process_only=False)
                
                # loss *= 0.0 # hack to discarding this batch
                abnormal_loss_detected = True

                logger.warning("try to save checkpoint ...",  main_process_only=False)
                if not (os.path.exists(f"{config.output_dir}/checkpoints") and len(os.listdir(f"{config.output_dir}/checkpoints")) > 2):
                    logger.warning("wait for everyone",  main_process_only=False)
                    
                    accelerator.save_state(f"{config.output_dir}/checkpoints/{global_step}_eve")
                    logger.warning(f"saving checkpoint at {global_step}")
                else:
                    logger.warning(f"checkpoints already exists, skip saving")
            
        # !! exiting autocast context manager
        accelerator.backward(loss)

        if abnormal_loss_detected:
            total_norm = 0.0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            logger.warning(f"->total grad_norm={total_norm}", main_process_only=False)

            from utils.utils import Normalizer
            norm = Normalizer.from_config(config)
            # TensorBoardTracker
            if config.ssi: 
                sample_pred = model_output[:8,...].detach().repeat(1,3,1,1) / norm.s + norm.t
                sample_gt = normalized_disp[:8,...].repeat(1,3,1,1) / norm.s + norm.t
            else:
                sample_pred  = norm.denormalize(model_output[:8,...].detach())
                sample_gt = norm.denormalize(normalized_disp[:8,...])
                sample_sim = norm.denormalize(sim_disp[:8,...])
                sample_pred = sample_pred / torch.amax(sample_pred, keepdim=True, dim=list(range(1,len(sample_pred.shape))))
                sample_gt = sample_gt / torch.amax(sample_gt, keepdim=True, dim=list(range(1,len(sample_pred.shape))))
                sample_sim = sample_sim / torch.amax(sample_sim, keepdim=True, dim=list(range(1,len(sample_pred.shape))))
            
            dump_dir = f"{config.output_dir}/checkpoints/{global_step}_dump"
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir, exist_ok=True)

            torch.save(left_image, f"{dump_dir}/left_image.pt")
            torch.save(right_image, f"{dump_dir}/right_image.pt")
            torch.save(normalized_disp, f"{dump_dir}/normalized_disp.pt")
            torch.save(normalized_rgb, f"{dump_dir}/normalized_rgb.pt")
            torch.save(sim_disp, f"{dump_dir}/sim_disp.pt")

            for i in range(network_loss.shape[0]):
                Image.fromarray((sample_gt[i,0].cpu().numpy()*255.).astype(np.uint8)).save(f"{dump_dir}/gt_{i}.png")
                Image.fromarray((sample_pred[i,0].cpu().numpy()*255.).astype(np.uint8)).save(f"{dump_dir}/pred_{i}.png")
                Image.fromarray((sample_sim[i,0].cpu().numpy()*255.).astype(np.uint8)).save(f"{dump_dir}/sim_{i}.png")
                Image.fromarray((((left_image[i].permute((1,2,0)).cpu().numpy()+1)*0.5)*255.).astype(np.uint8)).save(f"{dump_dir}/left_image_{i}.png")
                Image.fromarray((((right_image[i].permute((1,2,0)).cpu().numpy()+1)*0.5)*255.).astype(np.uint8)).save(f"{dump_dir}/right_image_{i}.png")
                Image.fromarray((((normalized_rgb[i].permute((1,2,0)).cpu().numpy()+1)*0.5)*255.).astype(np.uint8)).save(f"{dump_dir}/normalized_rgb_{i}.png")
                
            import shutil
            for i, p in enumerate(kwargs["path"]):
                shutil.copy2(p, f"{dump_dir}/{i}_disp.pfm")
                shutil.copy2(p.replace("disparity", "raw_finalpass").replace("pfm", "png"), f"{dump_dir}/{i}_raw.png")
                shutil.copy2(p.replace("disparity", "frames_finalpass").replace("pfm", "png"), f"{dump_dir}/{i}_left.png")
                shutil.copy2(p.replace("disparity", "frames_finalpass").replace("pfm", "png").replace("left", "right"), f"{dump_dir}/{i}_right.png")

            logger.warning(f"->lr={lr_scheduler.get_last_lr()[0]}", main_process_only=False)
            logger.warning(f"->global_step={global_step}", main_process_only=False)
            logger.warning(f"->epoch={step_counter.epoch}", main_process_only=False)
        
        if (abnormal_loss_detected and \
            not (os.path.exists(f"{config.output_dir}/checkpoints") and len(os.listdir(f"{config.output_dir}/checkpoints")) > 2)
        ):
            accelerator.wait_for_everyone()
            accelerator.save_state(f"{config.output_dir}/checkpoints/{global_step}_storm")
            logger.warning(f"saving checkpoint at {global_step}")
            breakpoint() # hack: stop training 

        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.clip_grad_norm).item()
            """ if grad_norm > 1:
                logger.warning(f"gradient clipped from {grad_norm} to 1.") """
        else:
            grad_norm = -1

        optimizer.step()
        optimizer.zero_grad()

        if not accelerator.optimizer_step_was_skipped:
            lr_scheduler.step()
                
    losses['loss'] = network_loss.mean(list(range(1, len(normalized_disp.shape)))).detach()
    losses['mse'] = (losses['loss'] ** 2 if config.loss_type == "l1" else losses['loss']) # BUG!

    # losses['vlb'] = [mse_to_vlb(t, mse, noise_scheduler.posterior_log_variance_clipped) for t, mse in enumerate(zip(timesteps, losses['mse']))]

    # gather training losses and metrics
    gathered_timesteps = accelerator.gather_for_metrics(timesteps)
    gathered_losses = accelerator.gather_for_metrics(losses)
    return loss.detach().item(), {'t': gathered_timesteps, 'losses': gathered_losses, 'gradn': grad_norm, 'abnormal_loss_detected': abnormal_loss_detected}

def train(accelerator: Accelerator, config: TrainingConfig, model: UNet2DModel, noise_scheduler: MyDDPMScheduler, 
        optimizer, train_dataloader, val_dataloader_lst, lr_scheduler, tokenizer = None, text_encoder = None, vae = None):
    if accelerator.is_main_process:
        """ if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id """
        accelerator.init_trackers("logs")
    
    step_counter = StepCounter()
    # prepare every objects relevant to training
    model, optimizer, train_dataloader, lr_scheduler, *val_dataloader_lst = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, *val_dataloader_lst
    )
    accelerator.register_for_checkpointing(step_counter)

    if config.resume_ckpt is not None and os.path.exists(config.resume_ckpt):
        logger.info(f"resuming checkpoints {config.resume_ckpt}")
        accelerator.load_state(config.resume_ckpt)
        skipped_dataloader = accelerator.skip_first_batches(train_dataloader, step_counter.local_step)
        resume_skipped_dataloader = True
    else:
        resume_skipped_dataloader = False
        
    # tokenizer = tokenizer.to(model.device)
    if text_encoder is not None:
        text_encoder = text_encoder.to(model.device)
    if vae is not None:
        vae = vae.to(model.device)

    # snr = noise_scheduler.alphas_cumprod / ( 1-noise_scheduler.alphas_cumprod)
    # t_sampler = create_named_schedule_sampler("snr", (snr ** 0.5 + 1).cpu().numpy())
    t_sampler = create_named_schedule_sampler("uniform", config.num_train_timesteps)
    t_sampler_mse = create_named_schedule_sampler("loss-second-moment", config.num_train_timesteps)
    # t_sampler_vlb = create_named_schedule_sampler("loss-second-moment", config.num_train_timesteps)

    distributed_state = PartialState()
    
    last_epoch = step_counter.epoch
    last_global_step = step_counter.global_step
    last_local_step = step_counter.local_step

    global_step = last_global_step # counts every step of  batch_size*accumulation_steps*num_processes samples
    
    # for epoch in range(config.num_epochs):
    for epoch in range(last_epoch, config.num_epochs):
        data_loader = train_dataloader if not resume_skipped_dataloader else skipped_dataloader 

        progress_bar = tqdm(total=len(data_loader), disable=not accelerator.is_local_main_process, position=0)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            local_step = (last_local_step+1 if resume_skipped_dataloader else 0) + step

            loss, gathered = train_step(accelerator, config, model, optimizer, lr_scheduler, noise_scheduler, t_sampler, step_counter,
                                        tokenizer = tokenizer, text_encoder = text_encoder, vae = vae, **batch)
           
            if not gathered['abnormal_loss_detected']: #loss > 0:
                step_counter.queue_loss(loss)
                t_sampler_mse.update_with_all_losses(gathered['t'], gathered['losses']['mse'])
                # t_sampler_vlb.update_with_all_losses(gathered['t'], gathered['losses']['vlb'])
            else:
                # breakpoint() # bug happens
                logger.critical("abnormal loss detected !")
            
            # if loss < 0: # skipped
            #     logger.warning(f"skipped batch {n} due to loss < 0")
            #     continue
            # logger.info(f"keeps training n={n}")
            
            if accelerator.sync_gradients: # step % config.gradient_accumulation_steps == 0:
                progress_bar.update(config.gradient_accumulation_steps)
                logs = {"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "gradn": gathered['gradn'], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(flatten(logs, "train", "/"), step=global_step) # FIXME
                
                if (
                    # (global_step < 1000 and (global_step+1) % 100 == 0) or \
                    # (global_step < config.val_every_global_steps and (global_step+1) % (config.val_every_global_steps//10) == 0) or \
                    (global_step+1) % config.val_every_global_steps == 0):
                    logger.info(f"Eval at epoch {epoch} global step {global_step}")

                    def get_filepath(dir, epoch, step, dataset, b, name):
                        return f"{dir}/samples/{dataset}/epoch{epoch:04d}_step{step}_pid{distributed_state.process_index}_b{b}_{name}"
                    
                    plot_loss_terms(t_sampler_mse.weights(), f"{config.output_dir}/metrics/step{global_step}_losses.png")
                    
                    pipeline = create_pipeline(accelerator, config, model, vae, tokenizer, text_encoder).to("cuda")
                    for val_dataloader in val_dataloader_lst:
                        val_dataset_name = val_dataloader.dataset.__class__.__name__
                        fname = partial(get_filepath, config.output_dir, epoch, global_step, val_dataset_name)
                        os.makedirs(f"{config.output_dir}/samples/{val_dataset_name}", exist_ok=True)
                        
                        disp_metrics = []
                        depth_metrics = []
                        val_progress_bar = tqdm(total=len(val_dataloader), disable=not accelerator.is_local_main_process, position=1)
                        val_progress_bar.set_description(f"Eval at epoch {epoch} global_step: {global_step}")
                        total_eval = 0
                        for i, batch in enumerate(val_dataloader):
                            disable_bar = not accelerator.is_local_main_process
                            pred_disps_ss, metrics_, uncertainties, error, denoised_images = eval_batch(config, pipeline, disable_bar, **batch)
                            # metrics = metrics_to_dict(*metrics_) # B,5 + B,6
                            # logger.info(f"metrics:{pretty_json(metrics)}")
                            if i == 0:
                                if uncertainties is not None:
                                    var = plot_uncertainties(uncertainties)
                                    var.save(fname(i, "var.png"))
                                error_map = plot_error_map(error)
                                error_map.save(fname(i, "error.png"))
                                grid = plot_denoised_images(config, denoised_images, pred_disps_ss, **batch)
                                grid.save(fname(i, "denoise.png"))

                            # gather whole batch results
                            # gathered_items = accelerator.gather_for_metrics(metrics_)
                            # disp_metrics.extend(gathered_items[0::2]) 
                            # depth_metrics.extend(gathered_items[1::2])

                            disp_err = torch.from_numpy(metrics_[0]).to(distributed_state.device) # extract to be gathered
                            depth_err = torch.from_numpy(metrics_[1]).to(distributed_state.device)

                            # gather all batch results
                            gathered_disp_err = accelerator.gather_for_metrics(disp_err)
                            gathered_depth_err = accelerator.gather_for_metrics(depth_err)

                            disp_metrics.extend(gathered_disp_err) 
                            depth_metrics.extend(gathered_depth_err)
                            total_eval += gathered_disp_err.shape[0]

                            val_progress_bar.update(1)
                            if config.eval_num_batch != -1 and (i+1) >= config.eval_num_batch:
                                break

                        # log whole val set results
                        gathered_metrics = metrics_to_dict(torch.vstack(disp_metrics).cpu().numpy(), torch.vstack(depth_metrics).cpu().numpy())
                        if val_dataset_name == config.eval_dataset[0]:
                            is_lowest = step_counter.step_eval(gathered_metrics["disp"]["epe"])
                        logger.info(f"metrics:{pretty_json(gathered_metrics)}")
                        logger.info(f"total evaluated {total_eval} samples, please check if correct")
                        accelerator.log(flatten(flatten(gathered_metrics), f"val_{val_dataset_name}", "/"), step=global_step)

                        if val_dataset_name == config.eval_dataset[0] and is_lowest:
                            logger.info(f"saving best model: {gathered_metrics['disp']['epe']} at epoch {epoch}, global step {global_step}")
                            pipeline.save_pretrained(f"{config.output_dir}/best")

                global_step += 1

            step_counter.save_step(epoch, global_step, local_step)

        # save model after every epoch
        accelerator.wait_for_everyone()
        # save checkpoint after every epoch
        accelerator.save_state(f"{config.output_dir}/checkpoints/last")
        logger.warning(f"saving checkpoint at {global_step}")

        if accelerator.is_main_process: 
            pipeline = create_pipeline(accelerator, config, model, vae, tokenizer, text_encoder)
            # always override newest epoch
            # pipeline.save_pretrained(config.output_dir)
            
            # save model every n epoch
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir + f"/epoch_{epoch:04d}")

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def run_train(base_cfg: Config):
    if base_cfg.seed != -1:
        seed_everything(base_cfg.seed) # for reproducing

    config = base_cfg.task
    if base_cfg.debug:
        set_debug(config)

    output_dir = get_output_dir(base_cfg)
    config.output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/metrics", exist_ok=True)

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=output_dir
    )

    fileHandler = logging.FileHandler(output_dir + "/train.log")
    logger.logger.addHandler(fileHandler)
    logger.info(f"training task: {config.name}")
    logger.info(f"output dir: {output_dir}")
    logger.info(f"running model: debug={base_cfg.debug}")
    from omegaconf import OmegaConf
    logger.info("configurations:")
    logger.info(pretty_json(OmegaConf.to_container(config)))
    logger.info('RUN ========================================')
    logger.info(' '.join(sys.argv))
    logger.info('END ========================================')

    train_dataloader, val_dataloader_lst = fetch_dataloader(config)

    clazz_unet = UNet2DConditionModel if config.ldm else UNet2DModel
    if not config.resume_pretrained:
        if config.ldm:
            # sanity check
            assert config.depth_channels == 4, "ldm only support 4 channels"
            assert config.mixed_precision == "no", "had not handle grad scaler yet"

            logger.info("load pretrained UNet from stable diffusion")
            model = UNet2DConditionModel.from_pretrained("checkpoint/stable-diffusion/unet")

            new_conv_in_channels = 4
            dup = len(config.cond_channels.split("+")) + 1
            new_conv_in_channels = dup * 4

            # adapt the first layer
            origin_conv_in = model.conv_in
            origin_state_dict = origin_conv_in.state_dict()
            weight = origin_state_dict["weight"].repeat(1, dup, 1, 1) / dup
            bias = origin_state_dict["bias"] / dup

            new_conv_in = torch.nn.Conv2d(new_conv_in_channels, origin_conv_in.out_channels, 
                kernel_size=origin_conv_in.kernel_size, 
                stride=origin_conv_in.stride, 
                padding=origin_conv_in.padding
            ) # @see L283 of unet_2d_condition.py

            new_conv_in.load_state_dict({'weight': weight, 'bias': bias})
            model.conv_in = new_conv_in
            model._internal_dict["in_channels"] = new_conv_in_channels # hack here
            model.config.in_channels = new_conv_in_channels
        else:
            
            if config.ssi:
                assert config.num_chs == 1
            if config.cond_channels == "left+right+raw":
                in_channels = 6+2*config.depth_channels
            elif config.cond_channels == "rgb+raw":
                in_channels = 3+2*config.depth_channels
            elif config.cond_channels == "rgb+left+right":
                in_channels = 9+config.depth_channels
            elif config.cond_channels == "rgb+left+right+raw":
                in_channels = 9+2*config.depth_channels
            else:
                raise ValueError(f"{config.cond_channels} not supported")

            model = UNet2DModel(
                sample_size=list(config.image_size), # h,w
                in_channels=in_channels,
                out_channels=config.depth_channels,
                layers_per_block=2,
                block_out_channels=tuple(config.block_out_channels),
                down_block_types=(
                    "DownBlock2D", # a regular ResNet downsampling block
                    "DownBlock2D", # "AttnDownBlock2D",
                    "DownBlock2D", # "AttnDownBlock2D",
                    "DownBlock2D", # "AttnDownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D", # "AttnUpBlock2D",
                    "UpBlock2D", # "AttnUpBlock2D",
                    "UpBlock2D", # "AttnUpBlock2D",
                    "UpBlock2D",
                )
            )

    elif os.path.exists(f"{config.resume_pretrained}"):
        patrained_path = f"{config.resume_pretrained}/unet"
        logger.info(f"resume unets from checkpoint: {patrained_path}")
        model = clazz_unet.from_pretrained(patrained_path)

    else:
        logger.error(f"resume patrained path not exists: {config.resume_pretrained}")
        exit(1)

    if config.ldm:
        tokenizer = CLIPTokenizer.from_pretrained("checkpoint/stable-diffusion/tokenizer")
        logger.info("load pretrained tokenizer")

        text_encoder = CLIPTextModel.from_pretrained("checkpoint/stable-diffusion/text_encoder")
        logger.info("load pretrained text-encoder and freeze")
        for param in text_encoder.parameters():
            param.requires_grad = False

        logger.info("load pretrained vae and freeze")
        vae = AutoencoderKL.from_pretrained("checkpoint/stable-diffusion/vae")
        for param in vae.parameters():
            param.requires_grad = False
    else:
        tokenizer = None
        text_encoder = None
        vae = None

        # adjust resolution
        inputPadder = InputPadder(config.image_size, divis_by=config.divis_by)
        model.sample_size[0] = inputPadder.padded_size[0]
        model.sample_size[1] = inputPadder.padded_size[1]

    """ model = UNet(
        in_channel=4,
        out_channel=1,
        inner_channel=64,
        channel_mults=[1,2,4,8],
        attn_res=[16],
        num_head_channels=32,
        res_blocks=2,
        dropout=0.2,
        image_size=352,
    ) """ # not done yet to try palatte

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"total trainable params: {count_parameters(model) / 1e6} M")
    noise_scheduler = create_sampler(config, train=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )

    steps_per_epoch = len(train_dataloader) // (config.gradient_accumulation_steps * accelerator.num_processes)
    total_num_steps = steps_per_epoch  * config.num_epochs
    
    if config.lr_scheduler == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, config.lr_warmup_steps) #ConstantLR(optimizer, factor=1)
    elif config.lr_scheduler == "cosine":
        """ see test_mini_train.py how to set num_training_steps
        """
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_num_steps*accelerator.num_processes,
            num_cycles=config.num_cycles
        )
    elif config.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_num_steps*accelerator.num_processes,
        )
    else:
        raise NotImplementedError

    train(accelerator, config, model, noise_scheduler, optimizer, train_dataloader,
           val_dataloader_lst, lr_scheduler, tokenizer, text_encoder, vae)
    accelerator.end_training()

if __name__ == "__main__":
    setup_hydra_configurations()
    run_train()

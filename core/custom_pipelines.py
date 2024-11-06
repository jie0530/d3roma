import torch
import numpy as np
import PIL
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers import DiffusionPipeline
from diffusers.pipelines import ImagePipelineOutput
from typing import Optional, Union, List
from diffusers.utils.torch_utils import randn_tensor

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from core.scheduler_ddpm import MyDDPMScheduler
from core.scheduler_ddim import MyDDIMScheduler
from typing import Callable

def resize_max_res(img, max_edge_resolution: int):
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    img: B,C,H,W
    """
    original_height, original_width  = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    round_to_factor_8 = lambda x: int(np.ceil(x / 8) * 8)

    new_width = round_to_factor_8(int(original_width * downscale_factor))
    new_height = round_to_factor_8(int(original_height * downscale_factor))

    resized_img = F.resize(img, (new_width, new_height), interpolation=InterpolationMode.BILINEAR)
    return resized_img

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



@dataclass
class GuidedPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        
        image_xt: B,H,W,T intermediate denoised steps


    """
    #### B,H,W final denoised results
    images: Union[List[PIL.Image.Image], np.ndarray] # x_0

    #### B,H,W,T
    images_pred_orig: Optional[np.ndarray] = None # \hat{x_0}
    images_perturbed_orig: Optional[np.ndarray] = None # \hat{x_0} + w * perturbation
    images_pred_prev: Optional[np.ndarray] = None # \miu_t{x_t, x_0}
    images_purturbed_pred_prev: Optional[np.ndarray] = None # \miu_t{x_t, x_0} + w * perturbation
    images_sampled_prev: Optional[np.ndarray] = None # x_{t-1} = \miu_t{x_t, x_0} + \delta_t * N(0,1)

    #### debug
    # debug_k: Optional[np.ndarray] = None # k_t
    
class GuidedDiffusionPipeline(DiffusionPipeline):

    def __init__(self, unet, scheduler, guidance):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, guidance=guidance)
    
    @torch.no_grad()
    def __call__(self, rgb_images = None, left_images = None, right_images = None, 
                    sim_disp = None, raw_depth = None, mask = None,
                    num_inference_steps: int = 128,
                    num_intermediate_images: int = 128,
                    add_noise_rgb: bool = False,
                    depth_channels: int =1,
                    cond_channels: str = "rgb",
                    denorm: Callable = None,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None):
        """ in & out images are preferably on the same device 
        """
        assert rgb_images is not None or left_images is not None
        assert num_intermediate_images > 0 and num_inference_steps % num_intermediate_images == 0

        plot_every_intermediate_steps = num_inference_steps // num_intermediate_images # T = N // S
        image_shape = rgb_images.shape if rgb_images is not None else left_images.shape
        batch_size = image_shape[0]

        # Sample gaussian noise to begin loop
        if type(self.unet.sample_size) == int:
            image = randn_tensor((batch_size, depth_channels, self.unet.sample_size, self.unet.sample_size), generator=generator)    
        elif type(self.unet.sample_size ) == tuple or type(self.unet.sample_size) == list:
            image = randn_tensor((batch_size, depth_channels, image_shape[-2],  image_shape[-1]), generator=generator)
        else:
            raise ValueError("sample_size must be int or tuple of ints")
                
        image = image.to(self.device)
        if rgb_images is not None:
            rgb_images = rgb_images.to(self.device)
        if left_images is not None:
            left_images = left_images.to(self.device)
        if right_images is not None:
            right_images = right_images.to(self.device)
        if sim_disp is not None:
            sim_disp = sim_disp.to(self.device)
        
        # preparations
        self.scheduler.set_timesteps(num_inference_steps)

        # if self.guidance.flow_guidance_weight > 0.0:
        #     self.guidance.start_stereo_match(left_images, right_images)

        # reverser diffusion
        images_pred_origs = []
        images_perturbed_origs = []
        images_pred_prevs = []
        images_sampled_prevs = []
        images_purturbed_prevs = []
        
        step = 0
        for t in self.progress_bar(self.scheduler.timesteps):
            # if t > 10: continue
            if add_noise_rgb:
                noise_rgb = torch.randn(rgb_images.shape).to(rgb_images.device)
                timesteps = (t * torch.ones(rgb_images.shape[0]).to(rgb_images.device)).long()
                noisy_rgb = self.scheduler.add_noise(rgb_images, noise_rgb, timesteps)
                final_rgb = rgb_images * 0.5 + noisy_rgb * 0.5
            else:
                final_rgb = rgb_images

            # model_input = torch.concat([image, final_rgb], dim=1)
            if cond_channels == "rgb":
                model_input = torch.cat([image, final_rgb], dim=1)
            elif cond_channels == "rgb+raw":
                model_input = torch.cat([image, final_rgb, sim_disp], dim=1)
            elif cond_channels == "rgb+right":
                model_input = torch.cat([image, left_images, right_images], dim=1)
            elif cond_channels == "left+right+raw":
                model_input = torch.cat([image, left_images, right_images, sim_disp], dim=1)
            elif cond_channels == "rgb+left+right":
                model_input = torch.cat([image, final_rgb, left_images, right_images], dim=1)
            elif cond_channels == "rgb+left+right+raw":
                model_input = torch.cat([image, final_rgb, left_images, right_images, sim_disp], dim=1)
            else:
                raise NotImplementedError
    
            model_output = self.unet(model_input, t).sample 
            g = self.scheduler.step(model_output, t, image) # , guidance=None self.guidance
            image = g.prev_sample #  [B,1,H,W]
            
            if (step + 1) % plot_every_intermediate_steps == 0:
                images_pred_origs.append(image.clamp(-1,1))#g.images_pred_orig)
                images_perturbed_origs.append(image.clamp(-1,1))#g.images_perturbed_orig)
                images_pred_prevs.append(image.clamp(-1,1))#g.images_pred_prev)
                images_purturbed_prevs.append(image.clamp(-1,1))#g.images_pred_prev)
                images_sampled_prevs.append(image.clamp(-1,1))#g.prev_sample)
                

            step += 1  
            # Experimental: Langevin dynamics
            # image = self.guidance.optimize(image, left_images, right_images, min_disp, max_disp)
            # image = self.unet(model_input, t).sample # directly outputs \tilde{miu}_t{x_t, x_0}

        images_pred_origs = torch.concat(images_pred_origs, dim=1)#.cpu().numpy()
        images_perturbed_origs = torch.concat(images_perturbed_origs, dim=1)
        images_pred_prevs = torch.concat(images_pred_prevs, dim=1)#.cpu().numpy()
        images_purturbed_prevs = torch.concat(images_purturbed_prevs, dim=1)#.cpu().numpy()
        images_sampled_prevs = torch.concat(images_sampled_prevs, dim=1)#.cpu().numpy()

        image = image.clamp(-1, 1)
        # image = image.cpu().numpy()

        return GuidedPipelineOutput(images=image, # B,dc,H,W
                                    images_pred_orig=images_pred_origs, # B,dc*T,H,W
                                    images_perturbed_orig=images_perturbed_origs, # B,dc*T,H,W
                                    images_pred_prev=images_pred_prevs, # B,dc*T,H,W
                                    images_purturbed_pred_prev=images_purturbed_prevs, # B,dc*T,H,W
                                    images_sampled_prev=images_sampled_prevs, # B,dc*T,H,W 
                                    ) 


class GuidedLatentDiffusionPipeline(DiffusionPipeline):
    
    def __init__(self, unet, vae, tokenizer, text_encoder, scheduler, guidance):
        super().__init__()
        self.register_modules(unet=unet, 
                              vae=vae, 
                              tokenizer=tokenizer, 
                              text_encoder=text_encoder, 
                              scheduler=scheduler, 
                              guidance=guidance)
        
    @torch.no_grad()
    def __call__(self, rgb_images, left_images = None, right_images = None, 
                    sim_disp = None, raw_depth = None,
                    raw_mask=None, empty_text_embed = None,
                    num_inference_steps: int = 128,
                    num_intermediate_images: int = 128,
                    add_noise_rgb: bool = False,
                    depth_channels: int =1,
                    cond_channels: str = "rgb",
                    denorm: Callable = None,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None):
        """ in & out images are preferably on the same device 
        """
        assert num_intermediate_images > 0 and num_inference_steps % num_intermediate_images == 0

        plot_every_intermediate_steps = num_inference_steps // num_intermediate_images # T = N // S
        
        # processing_res = 768 # FIXME
        # rgb_images = resize_max_res(
        #     rgb_images, max_edge_resolution=processing_res
        # )

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
            z = self.vae.post_quant_conv(depth_latent)
            stacked = self.vae.decoder(z)
            # mean of output channels
            depth_mean = stacked.mean(dim=1, keepdim=True)
            return depth_mean

        
        input_shape = rgb_images.shape if rgb_images is not None else (
            left_images.shape if left_images is not None else (
                right_images.shape if right_images is not None else (
                    sim_disp.shape if sim_disp is not None else None
                )
            )
        )
        assert input_shape is not None

        # Batched empty text embedding
        if empty_text_embed is None:
            empty_text_embed = __encode_empty_text(self.tokenizer, self.text_encoder)
        empty_text_embed = empty_text_embed.repeat(
            (input_shape[0], 1, 1)
        ) # [B, 2, 1024]

        """ # Sample gaussian noise to begin loop
        batch_size = cond_images.shape[0]
    

        # Sample gaussian noise to begin loop
        if type(self.unet.sample_size) == int:
            image = randn_tensor((batch_size, depth_channels, self.unet.sample_size, self.unet.sample_size), generator=generator)    
        elif type(self.unet.sample_size ) == tuple or type(self.unet.sample_size) == list:
            image = randn_tensor((batch_size, depth_channels, cond_images.shape[-2],  cond_images.shape[-1]), generator=generator)
        else:
            raise ValueError("sample_size must be int or tuple of ints") """
        
        image = randn_tensor((input_shape[0], 4, input_shape[2]//8, input_shape[3]//8), generator=generator)
        image = image.to(self.device)

        if rgb_images is not None:
            rgb_images = rgb_images.to(self.device)
            cond_images = encode_rgb(self.vae, rgb_images, 0.18215)
        
        if left_images is not None:
            left_images = left_images.to(self.device)
            left_images_latent = encode_rgb(self.vae, left_images, 0.18215)
                
        if right_images is not None:
            right_images = right_images.to(self.device)
            right_images_latent = encode_rgb(self.vae, right_images, 0.18215)

        if sim_disp is not None:
            sim_disp = sim_disp.to(self.device)
            sim_disp_latent = encode_disp(self.vae, sim_disp, 0.18215)
        
        # preparations
        self.scheduler.set_timesteps(num_inference_steps)

        # if self.guidance.flow_guidance_weight > 0.0:
        #     self.guidance.start_stereo_match(left_images, right_images)

        # # GUIDE in latent space FIXME
        # if hasattr(self.guidance, "disp_sm"):
        #     self.guidance.disp_sm_lat = encode_disp(self.vae, self.guidance.disp_sm, 0.18215) # B,4,H,W  
        #     self.guidance.valid_sm_lat = torch.ones_like(self.guidance.disp_sm_lat)

        # reverser diffusion
        images_pred_origs = []
        images_perturbed_origs = []
        images_pred_prevs = []
        images_sampled_prevs = []
        images_purturbed_prevs = []
        
        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # if t > 10: continue
            if "rgb" in cond_channels:
                if add_noise_rgb:
                    noise_rgb = torch.randn(cond_images.shape).to(cond_images.device)
                    timesteps = (t * torch.ones(cond_images.shape[0]).to(cond_images.device)).long()
                    noisy_rgb = self.scheduler.add_noise(cond_images, noise_rgb, timesteps)
                    final_rgb = cond_images * 0.5 + noisy_rgb * 0.5
                else:
                    final_rgb = cond_images

            # model_input = torch.concat([image, final_rgb], dim=1)
            if cond_channels == "rgb": #4:
                model_input = torch.cat([image, final_rgb], dim=1)
            elif cond_channels == "rgb+raw":
                cond_sims = encode_disp(self.vae, sim_disp, 0.18215)
                model_input = torch.cat([image, final_rgb, cond_sims], dim=1)
            elif cond_channels == "left+right":
                model_input = torch.cat([image, left_images_latent, right_images_latent], dim=1)
            elif cond_channels == "left+right+raw":
                model_input = torch.cat([image, left_images_latent, right_images_latent, sim_disp_latent], dim=1)
            elif cond_channels == "rgb+left+right":
                model_input = torch.cat([image, final_rgb, left_images_latent, right_images_latent], dim=1)
            elif cond_channels == "rgb+left+right+raw":
                model_input = torch.cat([image, final_rgb, left_images_latent, right_images_latent, sim_disp_latent], dim=1)
            else:
                raise ValueError(f"Unknown cond_channels: {cond_channels}")
    
            """LATENT model_output = self.unet(model_input, t, empty_text_embed).sample 
            if type(self.scheduler) == MyDDPMScheduler or type(self.scheduler) == MyDDIMScheduler:
                g = self.scheduler.step(model_output, t, image, 
                                        guidance=self.guidance, 
                                        decoder=__decode_depth,
                                        raw_depth=sim_disp,
                                        raw_mask=raw_mask,
                                        left_image=left_images,
                                        right_image=right_images)
            else:
                g = self.scheduler.step(model_output, t, image)

            image = g.prev_sample #  [B,4,H/f,W/f]
            if (step + 1) % plot_every_intermediate_steps == 0:
                d_hat = __decode_depth(g.pred_original_sample).clamp(-1,1)
                images_pred_origs.append(d_hat)#(__decode_depth(g.images_pred_orig).clamp(0,1)*2 -1
                images_perturbed_origs.append(d_hat)#(__decode_depth(g.images_perturbed_orig).clamp(0,1)*2 -1)
                images_pred_prevs.append(d_hat)#.append(__decode_depth(g.images_pred_prev).clamp(0,1)*2 -1)
                images_purturbed_prevs.append(d_hat)#.append(__decode_depth(g.images_pred_prev).clamp(0,1)*2 -1)
                images_sampled_prevs.append(d_hat)#.append(__decode_depth(g.prev_sample).clamp(0,1)*2 -1) """

            model_output = self.unet(model_input, t, empty_text_embed).sample 
            if self.scheduler.__class__.__name__ == "MyDDIMScheduler":
                assert self.guidance.flow_guidance_mode != "imputation", "latent diffusion does not support imputation mode"
                g = self.scheduler.step(model_output, 
                                        t, image, guidance=self.guidance,
                                        decoder=__decode_depth,
                                        denormer=denorm,
                                        raw_mask=raw_mask,
                                        left_image=left_images,
                                        right_image=right_images,
                                        raw_depth=raw_depth)
            else:
                g = self.scheduler.step(model_output, t, image)
                
            image = g.prev_sample #  [B,1,H,W]
           
            if (step + 1) % plot_every_intermediate_steps == 0:
                d_hat = __decode_depth(g.pred_original_sample).clamp(-1,1)
                images_pred_origs.append(d_hat)#(__decode_depth(g.images_pred_orig).clamp(0,1)*2 -1
                images_perturbed_origs.append(d_hat)#(__decode_depth(g.images_perturbed_orig).clamp(0,1)*2 -1)
                images_pred_prevs.append(d_hat)#.append(__decode_depth(g.images_pred_prev).clamp(0,1)*2 -1)
                images_purturbed_prevs.append(d_hat)#.append(__decode_depth(g.images_pred_prev).clamp(0,1)*2 -1)
                images_sampled_prevs.append(d_hat)#.append(__decode_depth(g.prev_sample).clamp(0,1)*2 -1)

            if step == num_inference_steps -1: # hack for d435 on real at galbot
               image_final = __decode_depth(g.pred_original_sample).clamp(-1,1)

            step += 1  
            # Experimental: Langevin dynamics
            # image = self.guidance.optimize(image, left_images, right_images, min_disp, max_disp)
            # image = self.unet(model_input, t).sample # directly outputs \tilde{miu}_t{x_t, x_0}

        images_pred_origs = torch.concat(images_pred_origs, dim=1)#.cpu().numpy()
        images_perturbed_origs = torch.concat(images_perturbed_origs, dim=1)
        images_pred_prevs = torch.concat(images_pred_prevs, dim=1)#.cpu().numpy()
        images_purturbed_prevs = torch.concat(images_purturbed_prevs, dim=1)#.cpu().numpy()
        images_sampled_prevs = torch.concat(images_sampled_prevs, dim=1)#.cpu().numpy()

        # images_pred_origs = F.resize(images_pred_origs, input_size)
        # images_perturbed_origs = F.resize(images_perturbed_origs, input_size)
        # images_pred_prevs = F.resize(images_pred_prevs, input_size)
        # images_purturbed_prevs = F.resize(images_purturbed_prevs, input_size)
        # images_sampled_prevs = F.resize(images_sampled_prevs, input_size)

        image = __decode_depth(image).clamp(-1,1) #.clamp(0, 1)*2-1


        """ batch_size, channels, *remaining_dims = image.shape
        image = image.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = image.abs()  # "a certain percentile absolute pixel value"
        s = torch.quantile(abs_sample, 1.0, dim=1)
        s = torch.clamp(
            s, min=1, max=1.2
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        image = torch.clamp(image, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"
        image = image.reshape(batch_size, channels, *remaining_dims) """

        # image = image.clamp(0, 1) # clamp to [0,1]
        # image = (image - 0.5) * 2 # [0,1] to [-1, 1] for compatibility
        # image = image.cpu().numpy()
        # image = F.resize(image, input_size)
        
        return GuidedPipelineOutput(images=image_final, #_final, # B,dc,H,W
                                    images_pred_orig=images_pred_origs, # B,dc*T,H,W
                                    images_perturbed_orig=images_perturbed_origs, # B,dc*T,H,W
                                    images_pred_prev=images_pred_prevs, # B,dc*T,H,W
                                    images_purturbed_pred_prev=images_purturbed_prevs, # B,dc*T,H,W
                                    images_sampled_prev=images_sampled_prevs, # B,dc*T,H,W 
                                    ) 
        


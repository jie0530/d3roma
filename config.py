from dataclasses import dataclass, field
from utils.camera import DepthCamera, RGBDCamera, Realsense

from diffusers import DDPMScheduler, HeunDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler
from core.scheduler_ddpm import MyDDPMScheduler
from core.scheduler_ddim import MyDDIMScheduler
from typing import List, Union, Optional, Tuple
from omegaconf import MISSING, OmegaConf
from omegaconf import DictConfig, OmegaConf, ValidationError
from hydra.core.config_store import ConfigStore

supported_samplers = {
    'ddpm': DDPMScheduler,
    'euler': EulerDiscreteScheduler,
    'heun': HeunDiscreteScheduler,
    'ddim': DDIMScheduler,
    'my_ddim': MyDDIMScheduler,
    'my_ddpm': MyDDPMScheduler
}

@dataclass
class Augment:
    resizedcrop: dict = field(default_factory=lambda: {
        'scale': [2, 2], 
        'ratio': [1.33333333333333,1.33333333333333333333]
    })
    hflip: str = "h" # off
    #==== raft stereo augmentation ====#    
    min_scale: float = 0 # -0.2
    max_scale: float = 0 # 0.4
    saturation_range: List[float] = field(default_factory=lambda: [0, 1.4])
    gamma: List[float] = field(default_factory=lambda: [1,1,1,1])
    yjitter: bool =False

@dataclass
class TrainingConfig:
    name: Optional[str] = "your task name here"
    tag: str = "" # your tag here
    camera_resolution: str = "320x256" # "224x128" # WxH dataset camera resolution, default "640x360"
    image_size: Tuple[int] = field(default_factory=lambda: (256, 320)) # (128, 224) #(352, 640) # [h,w] training image size
    divis_by: int = 32
    # image_size: tuple = (126, 224) # (128, 224) #(352, 640) # [h,w] training image size
    depth_channels: int = 1
    cond_channels: str = "rgb" # "rgb+raw" # "left+right" # "rgb+left+right"  # "left+right+raw" # "left+right+raw"
    train_batch_size: int = 12 # 16
    eval_batch_size: int = 12
    eval_num_batch: int = 2 # if set to -1, will evaluate whole val set

    num_epochs: int = 1000
    gradient_accumulation_steps: int = 3
    clip_grad_norm: float = 1.0
    
    lr_warmup_steps: int = 500
    val_every_global_steps: int = 1000
    save_model_epochs: int = 10
    mixed_precision: str = "no"  # `no` for float32, `fp16` for automatic mixed precision
    
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_model_id: str = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    # seed: int = 0

    train_dataset: List[str] = field(default_factory=lambda: ['NYUv2'])  #"std_100k" #  
    eval_dataset: List[str] = field(default_factory=lambda: ['NYUv2'])  #"std_100k" #  
    dataset_weight: List[int] = field(default_factory=lambda: [1])
    dataset_variant: str = "default"

    #### training settings
    ldm: bool = True
    prediction_space: str = "depth" # or "disp" ?
    ssi: bool = False
    # data normalizer
    normalize_mode: str = "average"
    num_chs: int = 3
    ch_bounds: List[float] = field(default_factory=lambda: [256, 256, 256])#[64, 64, 128]
    ch_gammas: List[float] = field(default_factory=lambda: [1/3., 1/3., 1/3. ])#[1., 1/3, 1/3]
    norm_t: float = 0.5
    norm_s: float = 2.0

    num_train_timesteps: int = 128 #1000 # diff-11
    num_inference_timesteps: int = 128 #1000 # diff-11
    num_inference_rounds: int = 1
    noise_strategy: str = 'randn' # ['randn', 'pyramid']
    loss_type: str = "l1" # "mse"
    learning_rate: float = 1e-4
    clip_gradient: bool = False

    #### scheduler
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    num_cycles: int = 1 
    beta_schedule: str = "squaredcos_cap_v2" # "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    noise_rgb: bool = False
    
    sampler: str = "my_ddpm"
    prediction_type: str = "v_prediction" # "sample" #  "epsilon" # 

    #### guidance settings
    flow_guidance_weights: List[float] = field(default_factory=lambda: [0.0])
    perturb_start_ratio: float = 1.0 # @deprecated
    guide_source: Optional[Union[str, None]] = None # "raw|stereo-match"
    flow_guidance_mode: str = "imputation"
    
    #### evaluation settings
    eval_output: str = ""
    eval_split: str = "val" # "test"
    write_pcd: bool = False
    num_intermediate_images: int = 8
    plot_mask: bool = False
    plot_error_map: bool = True
    plot_denoised_images: bool = True
    plot_intermediate_images: bool = False
    plot_intermediate_metrics: bool = False
    experiment_dir: str = "experiments"
    safe_ssi: bool = False # do ransac when align scales, only valid when ssi is on, should be turn off when training
    ransac_error_threshold: float = 0.6 # squared error, 0.6 works for nyu
    ensemble: bool = False
    coarse_to_fine: bool = False
    
    #### resume checkpoints
    resume_pretrained: Optional[str] = ""
    resume_ckpt: Optional[str] = ""

    #### experiment output directory, will be overriden automatically
    output_dir: Optional[str] = ""

    augment: Augment=field(default_factory=Augment) #Augment= MISSING #

    ### networks
    block_out_channels: Tuple[int] = field(default_factory=lambda: (128, 128, 256, 256, 512, 512))
    lr_scheduler: Optional[str] = "cosine"

@dataclass
class Config:
    debug: bool = False
    seed: int = -1
    task: TrainingConfig = MISSING

def setup_hydra_configurations():
    # setup hydra configurations
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    cs = ConfigStore.instance()
    cs.store(
        group="task",
        name="cfg",
        node=TrainingConfig
    )

def get_output_dir(base_config: Config):
    config = base_config.task
    ssi = "ssi" if config.ssi else "nossi"
    datasets = "_".join(config.train_dataset)
    weights = "_".join(format(x, ".1f") for x in config.flow_guidance_weights)
    tag = "" if config.tag=="" else f"-{config.tag}"

    return f"{config.experiment_dir}/{config.name}{tag}.dep{config.depth_channels}.lr{config.learning_rate:.0e}.{config.prediction_type}.{ssi}.{config.beta_schedule}.{config.noise_strategy}." + \
            f"{config.sampler}{config.num_train_timesteps}." + \
            f"{datasets}.{config.image_size[0]}x{config.image_size[1]}.{config.cond_channels}." + \
            f"w{weights}" + ("_debug" if base_config.debug else "")

def set_debug(config: TrainingConfig):
    config.val_every_global_steps = 10 #1000#
    config.save_model_epochs = 1
    config.train_batch_size = 1
    config.eval_batch_size = 1
    config.beta_schedule = "linear"
    config.beta_start = 1e-4
    config.beta_end = 2e-1
    # config.dataset = "nyu_depth_v2" # "std_debug" #720x360
    config.num_train_timesteps = 128 # 128#
    config.num_inference_timesteps = 128 # 128#
    config.num_intermediate_images = 4
    # config.output_dir = f"{config.output_dir}_debug"

def create_sampler(config, train=True):
    if config.sampler not in supported_samplers.keys():
        raise ValueError("Sampler not found")

    opt = {
        "num_train_timesteps": config.num_train_timesteps if train else config.num_inference_timesteps
    }
    
    if train:
        assert "ddim" not in config.sampler, "DDIM should not be used for training"
    
    opt["clip_sample"] = config.clip_sample
    opt["prediction_type"] = config.prediction_type
    opt["beta_schedule"] =  config.beta_schedule
    opt["beta_start"] = config.beta_start
    opt["beta_end"] = config.beta_end
    opt["num_train_timesteps"] = config.num_train_timesteps
    
    if config.sampler == "my_ddpm" or config.sampler == "ddpm":
        opt["clip_sample_range"] = config.clip_sample_range
        opt["thresholding"] = config.thresholding
        opt["dynamic_thresholding_ratio"] = config.dynamic_thresholding_ratio
    elif config.sampler == "my_ddim" or config.sampler == "ddim":
        opt["set_alpha_to_one"] = False
        opt["skip_prk_steps"] = True
        opt["steps_offset"] = 1
        opt["trained_betas"] = None
    else:
        raise ValueError("Sampler may not be configured properly?!")
    
    return supported_samplers[config.sampler].from_config(opt)

########### TESTING BELOW, INGNORE #############

def plot_iddpm_figure_1():
    def distortion(delta, sqared_err):
        # return ( math.log(1/math.sqrt(2*math.pi)) - math.log(delta) - 0.5 * sqared_err / delta**2)
        log_scales = th.FloatTensor([0.5 * math.log(delta)]) # 0.5 * log_variance
        centered_x = 0.95/256/256
        x = th.FloatTensor([0.5])
        
        inv_stdv = th.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = approx_standard_normal_cdf(min_in)
        log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = th.where(
            x < -0.999,
            log_cdf_plus,
            th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs
    
    # config.set_debug()
    T = 4000
    config.num_train_timesteps = config.num_inference_timesteps = T
    config.beta_schedule = "squaredcos_cap_v2"
    scheduler = create_sampler(config)
    print(distortion(scheduler.betas[0], 0.95**2))
    print(normal_kl(scheduler.alphas_cumprod[-1]**0.5*2, math.log(1-scheduler.alphas_cumprod[-1]), 0, 0)) # Section 4 ddpm: keep SNR at X_T ~= 1e-5 (=10**-5)

    vlb = []
    for t in range(T):
        vlb.append(
            normal_kl(scheduler.alphas_cumprod[t]**0.5*2, math.log(1-scheduler.alphas_cumprod[t]), 0, 0)
        )
    
    x = np.linspace(0, 1, T)
    y = scheduler.betas_tilde / scheduler.betas
    plt.plot(x, y, label="4000")

    T = 1000
    config.num_train_timesteps = config.num_inference_timesteps = T
    scheduler = create_sampler(config)
    print(distortion(scheduler.betas[0], 0.95**2))
    x2 = np.linspace(0, 1, T)
    y2 = scheduler.betas_tilde / scheduler.betas
    
    plt.plot(x2, y2, label="1000")
    print(normal_kl(scheduler.alphas_cumprod[-1]**0.5, math.log(1-scheduler.alphas_cumprod[-1]), 0, 0)) # Section 4 ddpm: keep SNR at X_T ~= 1e-5 (=10**-5)

    T = 128
    config.num_train_timesteps = config.num_inference_timesteps = T
    scheduler = create_sampler(config)
    alphas_cumprod_128 = scheduler.alphas_cumprod
    print(distortion(scheduler.betas[0], 0.95**2))
    x3 = np.linspace(0, 1, T)
    y3 = scheduler.betas_tilde / scheduler.betas
    plt.plot(x3, y3, label="128")
    plt.xlabel("t/T")
    plt.ylabel("~beta_t/beta_t")
    plt.legend(loc="upper right")
    plt.savefig("Figure 1.png")  # Figure 1 in iDDPM
    print(normal_kl(scheduler.alphas_cumprod[-1], math.log(1-scheduler.alphas_cumprod[-1]), 0, 0)) # Section 4 ddpm: keep SNR at X_T ~= 1e-5 (=10**-5)

def plot_iddpm_figure_2():
    T = 128
    config.num_train_timesteps = config.num_inference_timesteps = T
    scheduler = create_sampler(config)
    x = np.linspace(0, 1, T)
    vlbs = []
    for t in range(T):
        vlbs.append(
            normal_kl(0, math.log(1-scheduler.alphas_cumprod[t]), 0, 0)
        )
    
def plot_iddpm_figure_5():
    T = 1000
    config.num_train_timesteps = config.num_inference_timesteps = T
    config.beta_schedule = "linear"
    scheduler = create_sampler(config)
    alphas_cumprod_linear = scheduler.alphas_cumprod

    T = 1000
    config.num_train_timesteps = config.num_inference_timesteps = T
    config.beta_schedule = "squaredcos_cap_v2"
    scheduler = create_sampler(config)
    alphas_cumprod_cosine = scheduler.alphas_cumprod
    
    x = np.linspace(0, 1, T)
    plt.figure()
    plt.plot(x, alphas_cumprod_linear, label="linear")
    plt.plot(x, alphas_cumprod_cosine, label="cosine")
    plt.legend(loc="upper right")
    plt.xlabel("diffusion step t/T")
    plt.ylabel("alpha bar")
    plt.savefig("Figure 5.png")

def plot_snr():
    T = 128
    config.num_train_timesteps = T
    config.beta_schedule = "linear"
    scheduler = create_sampler(config)
    plt.figure()
    
    x = np.linspace(0, T, T)
    snr_linear = scheduler.alphas_cumprod / ( 1-scheduler.alphas_cumprod)
    # plt.plot(x, snr_linear, label="SNR Linear")
    plt.plot(x, snr_linear ** 0.5, label="sqrt SNR Linear")
    # plt.plot(x, th.log(snr_linear), label="log SNR Linear")

    config.beta_schedule = "squaredcos_cap_v2"
    scheduler = create_sampler(config)
    
    x = np.linspace(0, T, T)
    snr_cosine = scheduler.alphas_cumprod / ( 1-scheduler.alphas_cumprod)
    # plt.plot(x, snr_cosine, label="SNR cosine")
    plt.plot(x, snr_cosine ** 0.5, label="sqrt SNR cosine")
    # plt.plot(x, th.log(snr_cosine), label="log SNR cosine") 
    plt.xlabel("t/T")
    plt.ylabel("SNR")
    plt.legend(loc="upper right")
    plt.savefig("Figure_SNR.png")

def plot_sample_t():
    T = 128
    config.num_train_timesteps = T
    config.beta_schedule = "squaredcos_cap_v2"
    scheduler = create_sampler(config)
    snr_cosine = scheduler.alphas_cumprod / ( 1-scheduler.alphas_cumprod)
    from core.resample import create_named_schedule_sampler
    t_sampler = create_named_schedule_sampler("snr", (snr_cosine ** 0.5 + 1).cpu().numpy())
    timestemps, weights = t_sampler.sample(128, "cpu")
    # print(timestemps, weights)
    plt.figure()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(timestemps, bins=T)
    axs[1].hist(weights, bins=T)
    # print(weights.mean())
    plt.savefig("Figure_sampled_t.png")

if __name__ == "__main__": # DEBUG & PLOT schdulers
    config = TrainingConfig()

    from utils.losess import *

    import matplotlib.pyplot as plt
    import numpy as np
    import torch as th
    import math

    # plot_iddpm_figure_1() 
    # plot_iddpm_figure_2()
    # plot_iddpm_figure_5()
    plot_snr()
    plot_sample_t()

    """ 
    # resolution is irrelanvent for predicting depth
    print(config.camera.resolution_str)
    print(config.camera.resolution)
    print(config.camera.fxb)

    fxb = config.camera.fxb #* 2.5
    disp = fxb / 0.75
    print(disp)
    disp_2 = fxb / (0.75 + 0.001)
    print(disp_2 - disp)
    print(f"{(disp_2 - disp) / disp * 100} %") """




    


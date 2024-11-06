import os
import cv2
import torch
import numpy as np
from functools import partial
from utils.camera import Realsense

class D3RoMa():
    def __init__(self, overrides=[], camera=None):
        from config import TrainingConfig, setup_hydra_configurations
        self.camera: Realsense = camera

        setup_hydra_configurations()
        from hydra import compose, initialize
        with initialize(version_base=None, config_path="conf", job_name="inference"):
            base_cfg = compose(config_name="config.yaml", overrides=overrides)

        if base_cfg.seed != -1:
            from utils.utils import seed_everything
            seed_everything(base_cfg.seed) # for reproducing

        config: TrainingConfig = base_cfg.task
        self.camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
        self.pipeline =  self._load_pipeline(config)

        self.eval_output_dir = "_outputs"
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir, exist_ok=True)

        from utils.utils import Normalizer
        self.normer = Normalizer.from_config(config)
        self.config = config
        

    def _load_pipeline(self, config):
        patrained_path = f"{config.resume_pretrained}"
        if os.path.exists(patrained_path):
            print(f"load weights from {patrained_path}")
            
            from core.custom_pipelines import GuidedDiffusionPipeline, GuidedLatentDiffusionPipeline
            clazz_pipeline = GuidedLatentDiffusionPipeline if config.ldm else GuidedDiffusionPipeline
            pipeline = clazz_pipeline.from_pretrained(patrained_path).to("cuda")
            # model = UNet2DConditionModel.from_pretrained(patrained_path)
            pipeline.guidance.flow_guidance_mode=config.flow_guidance_mode

            if config.sampler == "my_ddim":
                from core.scheduler_ddim import MyDDIMScheduler
                my_ddim = MyDDIMScheduler.from_config(dict(
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
                pipeline.scheduler = my_ddim
                print(f"Careful! sampler is overriden to {config.sampler}")
        else:
            raise ValueError(f"patrained path not exists: {patrained_path}")
        
        return pipeline

    @torch.no_grad()
    def infer(self, left: np.ndarray, right: np.ndarray, raw_depth: np.ndarray=None, rgb:np.ndarray=None):
        """Depth restoration with left, right and raw depth
        
        Args:
            left (np.ndarray): left (IR) image
            right (np.ndarray): right (IR) image 
            raw (np.ndarray): raw depth image from camera sensors, unit is meter (optional)
            rgb (np.ndarray): RGB image (optional) for point cloud visualization only

        Returns:
            np.ndarray: restored depth image, unit is meter
        """
        assert len(left.shape) == len(right.shape)
        assert left.dtype == right.dtype == np.uint8

        if raw_depth is None or rgb is None:
            raise NotImplementedError("no worry, i will implement this soon")
        
        # assert raw.dtype == np.float32
        # if len(raw.shape) == 2:
        #     raw = raw[...,None]

        if len(left.shape) == 2:
            # grayscale images
            left = np.tile(left[...,None], (1, 1, 3))
            right = np.tile(right[...,None], (1, 1, 3))
        else:
            left = left[..., :3]
            right = right[..., :3]
        
        left = cv2.resize(left, self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)

        left = torch.from_numpy(left).permute(2, 0, 1).float()
        right = torch.from_numpy(right).permute(2, 0, 1).float()

        if rgb is not None:
            rgb = cv2.resize(rgb, self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            
        raw_depth = cv2.resize(raw_depth, dsize=self.camera.resolution[::-1], interpolation=cv2.INTER_NEAREST)
        if len(raw_depth.shape) == 3 and raw_depth.shape[-1] == 3:
            raw_depth = raw_depth [...,0]
        if len(raw_depth.shape) == 2:
            raw_depth = raw_depth[...,None]
        raw_depth = torch.from_numpy(raw_depth).permute(2, 0, 1).float()

        assert self.config.prediction_space == "disp", "not implemented"
        raw_disp = torch.zeros_like(raw_depth)
        raw_valid = (raw_depth > 0)
        raw_disp[raw_valid] = self.camera.fxb_depth / raw_depth[raw_valid]
        
        normalized_raw_disp = self.normer.normalize(raw_disp)[0] # normalized sim disp
        assert left.shape[1] % 8 == 0 and left.shape[2] % 8 == 0, "image size must be multiple of 8"
        
        normalize_rgb_fn = lambda x: (x / 255. - 0.5) * 2

        normalized_rgb = normalize_rgb_fn(rgb).to("cuda")
        left_image = normalize_rgb_fn(left).to("cuda")
        right_image = normalize_rgb_fn(right).to("cuda")
        raw_disp = raw_disp.to("cuda")

        def denormalize(config, pred_disps, raw_disp=None, mask=None):
            from utils.utils import Normalizer
            norm = Normalizer.from_config(config)

            if config.ssi:
                # assert config.depth_channels == 1, "fixme"
                B, R, H, W = pred_disps.shape
                # scale-shift invariant evaluation, consider using config.safe_ssi if the ssi computation is not stable
                batch_pred = pred_disps.reshape(-1, H*W) # BR, HW
                batch_gt = raw_disp.repeat(1, R, 1, 1).reshape(-1, H*W) # BR, HW
                batch_mask = mask.repeat(1, R, 1, 1).reshape(-1, H*W)
                if config.safe_ssi:
                    from utils.ransac import RANSAC
                    regressor = RANSAC(n=0.1, k=10, d=0.2, t=config.ransac_error_threshold)
                    regressor.fit(batch_pred, batch_gt, batch_mask)
                    st = regressor.best_fit
                    print(f"safe ssi in on: n=0.1, k=10, d=0.2, t={config.ransac_error_threshold}")
                else:
                    print("directly compute ssi")
                    from utils.utils import compute_scale_and_shift
                    st = compute_scale_and_shift(batch_pred, batch_gt, batch_mask) # BR, HW

                s, t = torch.split(st.view(B, R, 1, 2), 1, dim=-1)
                pred_disps_unnormalized = pred_disps * s + t
            else:
                pred_disps_unnormalized = norm.denormalize(pred_disps)

            return pred_disps_unnormalized
        
        denorm = partial(denormalize, self.config)
        mask = (raw_disp > 0).float().to("cuda")
        self.pipeline.set_progress_bar_config(desc=f"Denoising")

        # batchify
        normalized_rgb = normalized_rgb.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        left_image = left_image.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        right_image = right_image.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        normalized_raw_disp = normalized_raw_disp.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        raw_disp = raw_disp.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        mask = mask.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)

        pred_disps = self.pipeline(normalized_rgb, left_image, right_image, normalized_raw_disp, raw_disp, mask,
                num_inference_steps=self.config.num_inference_timesteps,
                num_intermediate_images=self.config.num_intermediate_images, # T
                add_noise_rgb=self.config.noise_rgb,
                depth_channels=self.config.depth_channels,
                cond_channels=self.config.cond_channels,
                denorm = denorm
            ).images
        
        if pred_disps.shape[0] > 1: # B is actually num_inference_rounds
            uncertainties = np.zeros_like(raw_disp)
            uncertainties[mask] = np.std(pred_disps.cpu().numpy(), axis=0)[mask]
        else:
            uncertainties = None

        pred_disps_unnormalized = denormalize(self.config, pred_disps, raw_disp, mask)
        pred_disps_unnormalized = pred_disps_unnormalized.mean(dim=0)
        
        if True:
            from utils.utils import compute_errors, metrics_to_dict, pretty_json
            metrics = compute_errors(raw_disp[0].cpu().numpy(), 
                                pred_disps_unnormalized.cpu().numpy(),
                                self.config.prediction_space,
                                mask[0].cpu().numpy().astype(bool), 
                                [self.camera.fxb_depth])
            
            metrics = metrics_to_dict(*metrics)
            print((f"metrics:{pretty_json(metrics)}"))

        pred_disps_unnormalized = pred_disps_unnormalized[0].cpu().numpy()
        pred_depth = np.zeros_like(pred_disps_unnormalized)
        pred_mask = (pred_disps_unnormalized > 0)
        pred_depth[pred_mask] = self.camera.fxb_depth / pred_disps_unnormalized[pred_mask]
        return pred_depth


if __name__ == "__main__":
    from utils.camera import Realsense
    camera = Realsense.default_real("fxm")
    overrides = [
        "task=eval_ldm_mixed",
        "task.resume_pretrained=experiments/ldm_sf-mixed.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.nossi.my_ddpm1000.SceneFlow_Dreds_HssdIsaacStd.180x320.cond7-raw+left+right.w0.0/epoch_0199",
        "task.eval_num_batch=1",
        "task.image_size=[360,640]", 
        "task.eval_batch_size=1",
        "task.num_inference_rounds=1",
        "task.num_inference_timesteps=10", "task.num_intermediate_images=5",
        "task.write_pcd=true"
    ]
    """ if False: # turn on guidance
        overrides += [
            "task.sampler=my_ddim", 
            "task.guide_source=raw-depth", 
            "task.flow_guidance_mode=gradient", 
            "task.flow_guidance_weights=[1.0]"
        ] """

    droma = D3RoMa(overrides, camera)

    from PIL import Image
    from hydra.utils import to_absolute_path
    left = np.array(Image.open(to_absolute_path("./assets/examples/0000_ir_l.png")))
    right = np.array(Image.open(to_absolute_path("./assets/examples/0000_ir_r.png")))
    raw = np.array(Image.open(to_absolute_path("./assets/examples/0000_depth.png"))) * 1e-3
    rgb = np.array(Image.open(to_absolute_path("./assets/examples/0000_rgb.png")))


    pred_depth = droma.infer(left, right, raw, rgb)

    import matplotlib.pyplot as plt
    cmap_spectral = plt.get_cmap('Spectral')
    pred_depth_normalized = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    Image.fromarray((cmap_spectral(pred_depth_normalized)*255.)[...,:3].astype(np.uint8)).save(f"{droma.eval_output_dir}/pred.png")

    if droma.config.write_pcd:
        from utils.utils import viz_cropped_pointcloud
        gt_depth_np = raw # [H,W]
        gt_masks_np = raw > 0
        gt_depth_np[~gt_masks_np] = 0.0
        gt_depth_np = camera.transform_depth_to_rgb_frame(gt_depth_np) #if not alreay aligned
        viz_cropped_pointcloud(camera.K.arr, rgb, gt_depth_np, fname=f"{droma.eval_output_dir}/raw.ply")
                           
        pred_depth = camera.transform_depth_to_rgb_frame(pred_depth)
        viz_cropped_pointcloud(camera.K.arr, rgb, pred_depth, fname=f"{droma.eval_output_dir}/pred.ply")

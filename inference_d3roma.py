import os
import cv2
import torch
import numpy as np
from functools import partial
import pathlib
from utils_d3roma.camera import Realsense

def denormalize(config, pred_disps, raw_disp=None, mask=None):
    from utils_d3roma.utils import Normalizer
    norm = Normalizer.from_config(config)

    if config.ssi:
        # assert config.depth_channels == 1, "fixme"
        B, R, H, W = pred_disps.shape
        # scale-shift invariant evaluation, consider using config.safe_ssi if the ssi computation is not stable
        batch_pred = pred_disps.reshape(-1, H*W) # BR, HW
        batch_gt = raw_disp.repeat(1, R, 1, 1).reshape(-1, H*W) # BR, HW
        batch_mask = mask.repeat(1, R, 1, 1).reshape(-1, H*W)
        if config.safe_ssi:
            from utils_d3roma.ransac import RANSAC
            regressor = RANSAC(n=0.1, k=10, d=0.2, t=config.ransac_error_threshold)
            regressor.fit(batch_pred, batch_gt, batch_mask)
            st = regressor.best_fit
            print(f"safe ssi in on: n=0.1, k=10, d=0.2, t={config.ransac_error_threshold}")
        else:
            print("directly compute ssi")
            from utils_d3roma.utils import compute_scale_and_shift
            st = compute_scale_and_shift(batch_pred, batch_gt, batch_mask) # BR, HW

        s, t = torch.split(st.view(B, R, 1, 2), 1, dim=-1)
        pred_disps_unnormalized = pred_disps * s + t
    else:
        pred_disps_unnormalized = norm.denormalize(pred_disps)

    return pred_disps_unnormalized

class D3RoMa():
    def __init__(self, overrides=[], camera=None, variant="left+right+raw"):
        assert variant in ["left+right+raw", "rgb+raw"], "not released yet"

        from config import TrainingConfig, setup_hydra_configurations
        self.camera: Realsense = camera

        setup_hydra_configurations()
        from hydra import compose, initialize
        with initialize(version_base=None, config_path="conf", job_name="inference"):
            base_cfg = compose(config_name="config.yaml", overrides=overrides)

        if base_cfg.seed != -1:
            from utils_d3roma.utils import seed_everything
            seed_everything(base_cfg.seed) # for reproducing

        config: TrainingConfig = base_cfg.task
        self.camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
        self.pipeline =  self._load_pipeline(config)

        self.eval_output_dir = f"_outputs.{variant}"
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir, exist_ok=True)

        from utils_d3roma.utils import Normalizer
        self.normer = Normalizer.from_config(config)
        self.config = config
        self.variant = variant

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
    def infer_with_rgb_raw(self, rgb: np.ndarray, raw_depth: np.ndarray):
        """Depth restoration with RGB and raw depth (RGB and depth SHOULD be aligned!)
        
        Args:
            rgb (np.ndarray): RGB image or gray image
            raw (np.ndarray): raw depth image from camera sensors, unit is meter

        Returns:
            np.ndarray: restored depth image, unit is meter
        """
        # print(f"rgb.shape: {rgb.shape}, raw_depth.shape: {raw_depth.shape}")
        assert rgb.dtype == np.uint8
        if len(rgb.shape[:2]) != len(raw_depth.shape[:2]):
            rgb = cv2.resize(rgb, dsize=raw_depth.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        if len(rgb.shape) == 2: #如果RGB图像是灰度图像（即只有两个维度），则将其复制三份以模拟RGB图像（三个通道）。
            # grayscale images
            rgb = np.tile(rgb[...,None], (1, 1, 3))
        else:
            rgb = rgb[..., :3]
        
        rgb = cv2.resize(rgb, self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() #改变数据的维度顺序以及数据类型

        if len(raw_depth.shape) == 2:
            raw_depth = raw_depth[...,None] #如果原始深度图像只有两个维度，则增加一个维度以匹配后续处理的期望格式。
        raw_depth = torch.from_numpy(raw_depth).permute(2, 0, 1).float() #改变原始深度图像的维度顺序以及数据类型

        assert self.config.prediction_space == "disp", "not implemented"
        raw_disp = torch.zeros_like(raw_depth)
        raw_valid = (raw_depth > 0) #根据原始深度图像中的有效深度值（大于0），计算对应的视差值，并存储在raw_disp中
        raw_disp[raw_valid] = self.camera.fxb_depth / raw_depth[raw_valid] #视差值的计算使用了相机焦距和基线距离（self.camera.fxb_depth）与深度值的倒数关系
        
        # normalized_raw_disp = self.normer.normalize(raw_disp)[0]
        return self.run_pipeline(None, None, raw_disp, rgb)

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
        
        assert left.shape[1] % 8 == 0 and left.shape[2] % 8 == 0, "image size must be multiple of 8"
        return self.run_pipeline(left, right, raw_disp, rgb)
        
    def run_pipeline(self, left_image, right_image, raw_disp, rgb):
        device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu" #
        normalize_rgb_fn = lambda x: (x / 255. - 0.5) * 2 #归一化到[-1, 1]的范围内
        
        #  batchify
        if rgb is not None:
            normalized_rgb = normalize_rgb_fn(rgb).to(device)
            normalized_rgb = normalized_rgb.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)

        if left_image is not None and right_image is not None:
            left_image = normalize_rgb_fn(left_image).to(device)
            right_image = normalize_rgb_fn(right_image).to(device)

            left_image = left_image.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
            right_image = right_image.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)

        raw_disp = raw_disp.to(device)
        normalized_raw_disp = self.normer.normalize(raw_disp)[0] # normalized sim disp
        normalized_raw_disp = normalized_raw_disp.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)

        raw_disp = raw_disp.unsqueeze(0).repeat(self.config.num_inference_rounds, 1, 1, 1)
        mask = (raw_disp > 0).float()

        denorm = partial(denormalize, self.config) # 用于后续将预测结果从归一化空间转换回原始空间。
        self.pipeline.set_progress_bar_config(desc=f"Denoising")
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred_disps = self.pipeline(normalized_rgb, left_image, right_image, normalized_raw_disp, raw_disp, mask,
                    num_inference_steps=self.config.num_inference_timesteps,
                    num_intermediate_images=self.config.num_intermediate_images, # T
                    add_noise_rgb=self.config.noise_rgb,
                    depth_channels=self.config.depth_channels,
                    cond_channels=self.config.cond_channels,
                    denorm = denorm
                ).images
        # 如果进行了多次推理（num_inference_rounds > 1），则计算预测视差图像在掩码区域内的不确定性（标准差）。
        if pred_disps.shape[0] > 1: # B is actually num_inference_rounds
            uncertainties = np.zeros_like(raw_disp)
            uncertainties[mask] = np.std(pred_disps.cpu().numpy(), axis=0)[mask]
        else:
            uncertainties = None
        # 使用denormalize函数将预测的视差图像从归一化空间转换回原始空间，并对多次推理的结果取平均。
        pred_disps_unnormalized = denormalize(self.config, pred_disps, raw_disp, mask)
        pred_disps_unnormalized = pred_disps_unnormalized.mean(dim=0)
        
        if False: #误差结果不打印
            from utils_d3roma.utils import compute_errors, metrics_to_dict, pretty_json
            # 使用compute_errors函数计算原始视差图和预测视差图之间的误差，并将误差指标转换为字典格式后打印出来。
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
    from utils_d3roma.camera import Realsense
    from utils_d3roma.realsense import RealSenseRGBDCamera
    import time
    import rospy
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import struct
    import ros_numpy
    rospy.init_node('pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher('raw_pcl', PointCloud2, queue_size=10)
    pub_pred = rospy.Publisher('pred_pcl', PointCloud2, queue_size=10)
    # 创建保存图片的目录
    output_dir = "inference_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化相机
    camera_config = Realsense.default_real("d435_right")
    overrides = [
        # uncomment if you choose variant left+right+raw
        # "task=eval_ldm_mixed",
        # "task.resume_pretrained=experiments/ldm_sf-mixed.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.nossi.my_ddpm1000.SceneFlow_Dreds_HssdIsaacStd.180x320.cond7-raw+left+right.w0.0/epoch_0199",
        
        # uncomment if you choose variant rgb+raw
        "task=eval_ldm_mixed_rgb+raw",
        # "task.resume_pretrained=experiments/ldm_sf-241212.2.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.ddpm1000.Dreds_HssdIsaacStd_ClearPose.180x320.rgb+raw.w0.0/epoch_0056",
        "task.resume_pretrained=/home/wsco/jie_ws/src/d3roma/experiments/ldm/epoch_0056",
        # rest of the configurations
        "task.eval_num_batch=1",
        "task.image_size=[360,640]", 
        # "task.image_size=[180,320]",
        "task.eval_batch_size=1",
        "task.num_inference_rounds=1",
        "task.num_inference_timesteps=5", "task.num_intermediate_images=1",
        "task.write_pcd=true"
    ]
    
    camera = RealSenseRGBDCamera(serial = "236522072295")
    for _ in range(30): 
        camera.get_rgbd_image()
    print("Initialization Finished.")

    droma = D3RoMa(overrides, camera_config, variant="rgb+raw")
    
    
    # 使用异步处理
    import threading
    import queue

    frame_queue = queue.Queue(maxsize=10)

    def inference_worker():
        while True:
            frames = frame_queue.get()
            if frames is None:
                break
            rgb_frame, depth_aligned = frames
            pred_depth = droma.infer_with_rgb_raw(rgb_frame, depth_aligned)
            
                
            # # 可视化处理
            # # 对深度图进行归一化以便显示
            # import matplotlib.pyplot as plt
            # cmap_spectral = plt.get_cmap('Spectral')
            
            # # 原始深度图可视化
            # valid = (depth_aligned > 0.2) & (depth_aligned < 5)
            # raw_depth_normalized = np.zeros_like(depth_aligned)
            # raw_depth_normalized[valid] = (depth_aligned[valid] - depth_aligned[valid].min()) / (depth_aligned[valid].max() - depth_aligned[valid].min())
            # raw_depth_vis = (cmap_spectral(raw_depth_normalized)*255.)[...,:3].astype(np.uint8)
            
            # # 预测深度图可视化
            # pred_depth_normalized = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
            # pred_depth_vis = (cmap_spectral(pred_depth_normalized)*255.)[...,:3].astype(np.uint8)
            
            # 获取当前时间戳
            # timestamp = time.time()
            # cloud = camera.create_point_cloud(rgb_frame, depth_aligned, camera_config.K.arr, voxel_size = 0.005, fname=f"{output_dir}/raw_{timestamp}.ply")
            # cloud, points, colors = camera.create_point_cloud(rgb_frame, pred_depth, camera_config.K.arr, voxel_size = 0.005, fname=f"{output_dir}/pred_{timestamp}.ply")

            # cloud, points, colors = camera.create_point_cloud(rgb_frame, pred_depth, camera_config.K.arr, voxel_size = 0.005)

            # 发布点云数据到ROS话题
            # cloud_array = camera.merge_xyz_rgb(points, colors)
            # pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            #     cloud_array, rospy.Time.now(), "cam_right_link"
            # )
            # pub.publish(pointcloud_msg)

            # # 创建并保存原始点云（同时发布到ROS）
            # cloud = camera.create_point_cloud(
            #     rgb_frame, 
            #     depth_aligned,
            #     camera_config.K.arr, 
            #     voxel_size=0.005, 
            #     publish_ros=True
            # )
            
            # # 创建并保存预测点云（同时发布到ROS）
            # cloud = camera.create_point_cloud(
            #     rgb_frame, 
            #     pred_depth, 
            #     camera_config.K.arr, 
            #     voxel_size=0.005, 
            #     publish_ros=True
            # )

            # # 保存RGB图像
            # cv2.imwrite(f"{output_dir}/rgb_{timestamp}.png", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            # # 保存原始深度图
            # cv2.imwrite(f"{output_dir}/raw_depth_{timestamp}.png", raw_depth_vis)
            
            # # 保存预测深度图
            # cv2.imwrite(f"{output_dir}/pred_depth_{timestamp}.png", pred_depth_vis)
            
            # # 保存原始深度数据（可选）
            # np.save(f"{output_dir}/raw_depth_data_{timestamp}.npy", depth_aligned)
            
            # # 保存预测深度数据（可选）
            # np.save(f"{output_dir}/pred_depth_data_{timestamp}.npy", pred_depth)
            
            print(f"Saved results for timestamp: {timestamp}")
            
            # # 保存原始点云
            # from utils_d3roma.utils import viz_cropped_pointcloud
            # pcd_rgbd_raw = viz_cropped_pointcloud(
            #     camera_config.K.arr, 
            #     rgb_frame, 
            #     depth_aligned, 
            #     fname=f"{output_dir}/raw_{timestamp}.ply"
            # )
            # # 保存预测点云
            # pcd_rgbd_pred = viz_cropped_pointcloud(
            #     camera_config.K.arr, 
            #     rgb_frame, 
            #     pred_depth, 
            #     fname=f"{output_dir}/pred_{timestamp}.ply"
            # )

            # def pointcloud_to_ros_msg(pcd):
            #     """
            #     将 Open3D 的点云数据转换为 ROS 的 PointCloud2 消息
            #     """
            #     points = np.asarray(pcd.points)
            #     colors = np.asarray(pcd.colors)
            
            # 按'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # 启动推理线程
    # inference_thread = threading.Thread(target=inference_worker)
    # inference_thread.start()
    
    while True:
        # 从相机读取RGB和深度图像
        rgb_frame, depth_aligned = camera.get_rgbd_image()
        # frame_queue.put((rgb_frame, depth_aligned))
        
        pred_depth = droma.infer_with_rgb_raw(rgb_frame, depth_aligned)
            
        timestamp = time.time()
        raw_pcd = camera.rgbd_to_pointcloud(rgb_frame, depth_aligned, camera_config.K.arr, camera.cam_extrinsic(), downsample_factor=1, 
                                      fname=f"{output_dir}/raw_{timestamp}.ply"
                                    ) 
        
        # 发布点云数据到ROS话题
        cloud_array = camera.merge_xyz_rgb(raw_pcd.points, raw_pcd.colors)
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            # cloud_array, rospy.Time.now(), "cam_right_color_optical_frame"
            cloud_array, rospy.Time.now(), "base_link"
        )
        pub.publish(pointcloud_msg)  
        
        
        pred_pcd = camera.rgbd_to_pointcloud(rgb_frame, pred_depth, camera_config.K.arr, camera.cam_extrinsic(), downsample_factor=1, 
                #   fname=f"{self.output_dir}/pred_{timestamp}.ply"
                ) 
        
        cloud_array = camera.merge_xyz_rgb(pred_pcd.points, pred_pcd.colors)
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            # cloud_array, rospy.Time.now(), "cam_right_color_optical_frame"
            cloud_array, rospy.Time.now(), "base_link"
        )
        pub_pred.publish(pointcloud_msg)  
        print(f"Published pred pcl at {timestamp}")
        
        
        # pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg)
        # if cloud_array.any == pcd.any:
        #     print("*****************************sfsdfds")
        # pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        # pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        # pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        # pcd_rgb[..., 0] = pcd["r"]
        # pcd_rgb[..., 1] = pcd["g"]
        # pcd_rgb[..., 2] = pcd["b"]
        # pcd_rgb = pcd_rgb[pcd_xyz_mask]

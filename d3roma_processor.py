#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import threading
import queue
import time
# import sys
# sys.path.append('/home/wsco/jie_ws/src/d3roma/')
from inference_d3roma import D3RoMa
from utils_d3roma.camera import Realsense
from utils_d3roma.utils import viz_cropped_pointcloud
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import ros_numpy
import open3d as o3d
    

class D3RoMaProcessor:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('d3roma_processor', anonymous=True)
        
        # 初始化CV桥接
        self.bridge = CvBridge()
        
        # 初始化D3RoMa
        self.camera_config = Realsense.default_real("d435")
        overrides = [
            "task=eval_ldm_mixed_rgb+raw",
            "task.resume_pretrained=/home/wsco/jie_ws/src/d3roma/experiments/ldm/epoch_0056",
            "task.eval_num_batch=1",
            "task.image_size=[480,640]",
            "task.eval_batch_size=1",
            "task.num_inference_rounds=1",
            "task.num_inference_timesteps=5",
            "task.num_intermediate_images=1",
            "task.write_pcd=true"   
        ]
        self.droma = D3RoMa(overrides, self.camera_config, variant="rgb+raw")
        
        # 创建帧队列
        self.frame_queue = queue.Queue(maxsize=10)
        
        # 初始化存储变量
        self.rgb_frame = None
        self.depth_frame = None
        self.latest_timestamp = None
        
        # 创建锁
        self.lock = threading.Lock()
        
        # self.pub = rospy.Publisher('fused_pcl', PointCloud2, queue_size=10)
        self.pub_raw = rospy.Publisher('raw_pcl', PointCloud2, queue_size=10)
        self.pub_pred = rospy.Publisher('pred_pcl', PointCloud2, queue_size=10)
        
        # 订阅相机话题
        self.rgb_sub = rospy.Subscriber(
            "/cam_right/color/image_raw",  # 根据实际话题名称修改
            Image,
            self.rgb_callback
        )
        
        self.depth_sub = rospy.Subscriber(
            "/cam_right/aligned_depth_to_color/image_raw",  # 根据实际话题名称修改
            Image,
            self.depth_callback
        )
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_worker)
        self.process_thread.start()
        
        # 启动推理线程
        # self.inference_thread = threading.Thread(target=self.inference_worker)
        # self.inference_thread.start()
        
        print("D3RoMa Processor initialized")
        
        self.output_dir = "/home/wsco/jie_ws/src/d3roma/inference_results"
        
        
    def create_point_cloud(self, colors, depths, cam_intrinsics, voxel_size = 0.005, fname = None):
        """
        color, depth => point cloud
        """
        h, w = depths.shape
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        
        # colors = o3d.geometry.Image(np.ascontiguousarray(depths).astype(np.float32))
        # depths = o3d.geometry.Image(np.ascontiguousarray(colors).astype(np.uint8))

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(voxel_size) # 下采样
        
        # # 将点云转换到base_link坐标系
        # xyz = [-0.783746, 0.437297, 0.246427+0.018]
        # rpy = [-1.96669+1.5+1.5, 0.0312856+23, 3.09404-90-5]
        
        
        # xyz = [-0.783746, 0.437297, 0.246427]
        # rpy = [-1.96669, 0.0312856, 3.09404]
        # # 转换为弧度
        # rpy = np.radians(rpy)

        # # 转换为齐次变换矩阵
        # base_to_camera = self.xyz_rpy_to_homogeneous_matrix(xyz, rpy)
        # camera_to_base = np.linalg.inv(base_to_camera)
        # cloud.transform(camera_to_base)
        
        # if show:
        #     o3d.visualization.draw_geometries([cloud])
        if fname is not None:
            o3d.io.write_point_cloud(fname, cloud)
        
        # points = np.array(cloud.points).astype(np.float32)
        # colors = np.array(cloud.colors).astype(np.float32)

        # WORKSPACE_MIN = np.array([-1.2, -0.5, 0])
        # WORKSPACE_MAX = np.array([-0.5, 0.5, 0.3])
        # IMG_MEAN = np.array([0.485, 0.456, 0.406])
        # IMG_STD = np.array([0.229, 0.224, 0.225])

        # x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        # y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        # z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        # mask = (x_mask & y_mask & z_mask)
        # points = points[mask]
        # colors = colors[mask]
        # imagenet normalization
        # colors = (colors - IMG_MEAN) / IMG_STD
        
        # # final cloud
        # cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
        # if show:
        #     o3d.visualization.draw_geometries([cloud_final])
        return cloud.points, cloud.colors     
    
    
    def merge_xyz_rgb(self, xyz, rgb):
        # 将点云的空间坐标(xyz)和颜色信息(rgb)合并成一个结构化数组
        # 将RGB颜色值打包成一个32位的浮点数
        # 用于创建ROS点云消息
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.uint8)

        rgb_packed = np.asarray(
            (rgb[:, 0].astype(np.uint32) << 16)
            | (rgb[:, 1].astype(np.uint32) << 8)
            | rgb[:, 2].astype(np.uint32),
            dtype=np.uint32,
        ).view(np.float32)

        structured_array = np.zeros(
            xyz.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ],
        )
        structured_array["x"] = xyz[:, 0]
        structured_array["y"] = xyz[:, 1]
        structured_array["z"] = xyz[:, 2]
        structured_array["rgb"] = rgb_packed

        return structured_array
    
    def cam_intrinsics(self):        
        return np.array([
            [604.988525390625, 0, 325.60302734375, 0],
            [0, 604.2501831054688, 251.7237548828125, 0],
            [0, 0, 1, 0]
        ])
    
    
    def rgb_callback(self, msg):
        """处理RGB图像回调"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {e}")
    
    def depth_callback(self, msg):
        """处理深度图像回调"""
        try:
            # 将ROS深度图像消息转换为OpenCV格式
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
            
    def rgbd_to_pointcloud(
        self, 
        color:np.ndarray, 
        depth:np.ndarray, 
        intrinsic:np.ndarray, 
        # extrinsic:np.ndarray=np.eye(4), 
        downsample_factor:float=1,
        fname:str=None,
        pcl_type:str="pred"
    ):
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = depth.astype(np.float32)

        # downsample image
        color = cv2.resize(color, (int(640 / downsample_factor), int(480 / downsample_factor))).astype(np.int8)
        depth = cv2.resize(depth, (int(640 / downsample_factor), int(480 / downsample_factor)))

        if pcl_type == "raw":
            depth /= 1000.0  # from millimeters to meters
        # depth[depth < self.min_depth_m] = 0
        # depth[depth > self.max_depth_m] = 0
        

        rgbd_image = o3d.geometry.RGBDImage()
        rgbd_image = rgbd_image.create_from_color_and_depth(o3d.geometry.Image(color),
            o3d.geometry.Image(depth), depth_scale=1.0, convert_rgb_to_intensity=False)

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(640 / downsample_factor), int(480 / downsample_factor), 
            intrinsic[0, 0] / downsample_factor, intrinsic[1, 1] / downsample_factor, 
            intrinsic[0, 2] / downsample_factor, intrinsic[1, 2] / downsample_factor)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d) #, extrinsic=extrinsic)
        # 在保存点云前添加坐标系变换
        # cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd)
        
        return pcd        
    
    def process_worker(self):
        """处理线程，将同步的RGB和深度图像放入队列"""
        # rate = rospy.Rate(30)  # 30Hz
        while not rospy.is_shutdown():
                if self.rgb_frame is not None and self.depth_frame is not None:
                    # 将同步的帧放入队列
                    # self.frame_queue.put((self.rgb_frame, self.depth_frame), block=False)
                    timestamp = time.time()
                    rgb_image = cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2RGB)
                    # 确保深度图像的类型为float32
                    depth_image = self.depth_frame.astype(np.float32)
                    pred_depth = self.droma.infer_with_rgb_raw(rgb_image, depth_image)
                    
                    # 预测深度还原，反归一化
                    pred_depth = pred_depth * 2.4
                    
                    # raw_pcd = self.rgbd_to_pointcloud(self.rgb_frame, self.depth_frame, self.cam_intrinsics(), downsample_factor=1, 
                    #                             #   fname=f"{self.output_dir}/pred_{timestamp}.ply"
                    # pcl_type="raw"
                    #                             ) 
                    
                    # # 发布点云数据到ROS话题
                    # cloud_array = self.merge_xyz_rgb(raw_pcd.points, raw_pcd.colors)
                    # pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                    #     cloud_array, rospy.Time.now(), "cam_right_color_optical_frame"
                    # )
                    # self.pub_raw.publish(pointcloud_msg)  
                    
                    pred_pcd = self.rgbd_to_pointcloud(rgb_image, pred_depth, self.cam_intrinsics(), downsample_factor=1, 
                            #   fname=f"{self.output_dir}/pred_{timestamp}.ply"
                            pcl_type="pred"
                            ) 
                    
                    cloud_array = self.merge_xyz_rgb(pred_pcd.points, pred_pcd.colors)
                    pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                        cloud_array, rospy.Time.now(), "cam_right_color_optical_frame"
                    )
                    self.pub_pred.publish(pointcloud_msg)  
                    print(f"Published pred pcl at {timestamp}")
    
    def inference_worker(self):
        """推理线程，处理队列中的图像"""
        while not rospy.is_shutdown():
            try:
                # # 从队列获取图像
                # rgb_frame, depth_frame = self.frame_queue.get(timeout=1.0)
                # time1 = time.time()
                # 执行推理
                # pred_depth = self.droma.infer_with_rgb_raw(self.rgb_frame, self.depth_frame)
                if self.rgb_frame is not None and self.depth_frame is not None:
                    rgb_image = cv2.cvtColor(self.rgb_frame, cv2.COLOR_BGR2RGB)
                    # 确保深度图像的类型为float32
                    depth_image = self.depth_frame.astype(np.float32)
                    pred_depth = self.droma.infer_with_rgb_raw(rgb_image, depth_image)
                    
                    points, colors = self.create_point_cloud(self.rgb_frame, pred_depth, self.camera_config.K.arr, voxel_size = 0.005, fname=f"{self.output_dir}/pred_{timestamp}.ply")
                    # 发布点云数据到ROS话题
                    cloud_array = self.merge_xyz_rgb(points, colors)
                    pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                        cloud_array, rospy.Time.now(), "/cam_right_color_optical_frame"
                    )
                    self.pub.publish(pointcloud_msg)
                
                
                timestamp = time.time()
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
                
                # 打印处理时间
                rospy.loginfo(f"Processed frame at {rospy.Time.now().to_sec()}")
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error in inference: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.rgb_sub.unregister()
        self.depth_sub.unregister()
        self.process_thread.join()
        self.inference_thread.join()

if __name__ == "__main__":
    rospy.init_node('d3roma_processor', anonymous=True)
    try:
        processor = D3RoMaProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        processor.cleanup()
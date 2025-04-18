'''
RealSense Camera.
'''

import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2


class RealSenseRGBDCamera:
    '''
    RealSense RGB-D Camera.
    '''
    def __init__(
        self, 
        serial, 
        frame_rate = 30, 
        resolution = (640, 360),# width, height
        # resolution = (180, 320),
        align = True,
        **kwargs
    ):
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 15, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - align: bool, optional, default: True, whether align the frameset with the RGB image.
        '''
        super(RealSenseRGBDCamera, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        # =============== Support L515 Camera ============== #
        self.is_radar = str.isalpha(serial[0])
        print(f"self.is_radar: {self.is_radar}") # False
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000
        else:
            self.depth_scale = 1000
        # ================================================== #
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)
        self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align

    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        return color_image

    def get_depth_image(self):
        '''
        Get the depth image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / self.depth_scale
        return depth_image

    def get_rgbd_image(self):
        '''
        Get the RGB image along with the depth image from the camera.
        '''
        frameset = self.pipeline.wait_for_frames()
        if self.with_align:
            frameset = self.align.process(frameset)
        color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.float32) / self.depth_scale
        return color_image, depth_image
    
    def rgbd_to_pointcloud(
        self, 
        color:np.ndarray, 
        depth:np.ndarray, 
        intrinsic:np.ndarray, 
        extrinsic:np.ndarray=np.eye(4), 
        downsample_factor:float=1,
        fname:str=None,
        pcl_type:str="pred",
        voxel_size:float=0.005
    ):
        depth = depth.astype(np.float32)
        # downsample image
        color = cv2.resize(color, (int(640 / downsample_factor), int(480 / downsample_factor))).astype(np.uint8)
        depth = cv2.resize(depth, (int(640 / downsample_factor), int(480 / downsample_factor)))

        # if pcl_type == "raw":
        #     depth /= 1000.0  # from millimeters to meters
        # depth[depth < self.min_depth_m] = 0
        # depth[depth > self.max_depth_m] = 0
        

        rgbd_image = o3d.geometry.RGBDImage()
        rgbd_image = rgbd_image.create_from_color_and_depth(o3d.geometry.Image(color),
            o3d.geometry.Image(depth), depth_scale=1.0, convert_rgb_to_intensity=False)

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(640 / downsample_factor), int(480 / downsample_factor), 
            intrinsic[0, 0] / downsample_factor, intrinsic[1, 1] / downsample_factor, 
            intrinsic[0, 2] / downsample_factor, intrinsic[1, 2] / downsample_factor)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d)
        pcd = pcd.voxel_down_sample(voxel_size) # 下采样
        # 在保存点云前添加坐标系变换
        pcd.transform(extrinsic)
        # pcd.transform(np.linalg.inv(extrinsic))
        # pcd = self.transform_pointcloud(pcd)
        # 可视化变换后的点云
        # o3d.visualization.draw_geometries([pcd])
        # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd)
        
        return pcd
    

    def transform_pointcloud(self,pcd):
        # 平移向量
        translation = np.array([-0.769, 0.436, 0.265])
        # 四元数
        quaternion = np.array([0.031, 0.833, -0.553, -0.002])

        # 使用 scipy 库将四元数转换为旋转矩阵
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # 构建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        # 应用变换矩阵到点云
        transformed_pcd = pcd.transform(transform_matrix)

        return transformed_pcd
    
          
    def cam_extrinsic(self):
        # return np.array([[ 0.00076424,  0.99948454, -0.03150254, -0.769],
        #             [-0.03094391,  0.03154152,  0.99902737,  0.436     ],
        #             [-0.99951999,  0.00057717, -0.03093903,  0.265     ],
        #             [ 0.,          0.,          0.,          1.        ]])
        return np.array([[ -0.99807128,  0.04940125, -0.03759308, -0.769],
                    [0.05382232,  0.38686651,  -0.92056367,  0.436     ],
                    [-0.03093349,  -0.9208115,  -0.38877924,  0.265     ],
                    [ 0.,          0.,          0.,          1.        ]])


    def create_point_cloud(self, colors, depths, cam_intrinsics, voxel_size = 0.005, fname = None):
        """
        color, depth => point cloud
        """
        h, w = depths.shape
        print(f"h: {h}, w: {w}")
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        
        # colors = o3d.geometry.Image(np.ascontiguousarray(depths).astype(np.float32))
        # depths = o3d.geometry.Image(np.ascontiguousarray(colors).astype(np.uint8))
        
        # xyz_base_cam = [-0.783746, 0.437297, 0.246427]
        # rpy_base_cam = [-1.96669, 0.0312856, 3.09404]
        xyz_base_cam = [-0.783746, 0.437297, 0.246427+0.018]
        rpy_base_cam = [-1.96669+1.5+1.5, 0.0312856+23, 3.09404-90-5]
        
        # 转换为弧度
        rpy_base_cam = np.radians(rpy_base_cam)
        
        xyz_cam_op = [0.000, 0.015, 0.000]
        rpy_cam_op = [-1.568, 0.018, -1.577]


        # 转换为齐次变换矩阵
        base_to_camera = self.xyz_rpy_to_homogeneous_matrix(xyz_base_cam, rpy_base_cam)
        camera_to_opti = self.xyz_rpy_to_homogeneous_matrix(xyz_cam_op, rpy_cam_op)
        
        base_to_opti = np.dot(base_to_camera, camera_to_opti)
        opti_to_base = np.linalg.inv(base_to_opti)
        

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics, opti_to_base)
        # cloud = cloud.voxel_down_sample(voxel_size) # 下采样
        
        # 在保存点云前添加坐标系变换
        cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        if fname is not None:
            o3d.io.write_point_cloud(fname, cloud)
        
        # # 将点云转换到base_link坐标系
        # xyz = [-0.783746, 0.437297, 0.246427+0.018]
        # rpy = [-1.96669+1.5+1.5, 0.0312856+23, 3.09404-90-5]
        
        

        
        # cloud.transform(base_to_opti)
        
        # if show:
        #     o3d.visualization.draw_geometries([cloud])
        if fname is not None:
            o3d.io.write_point_cloud(fname, cloud)
        
        points = np.array(cloud.points).astype(np.float32)
        colors = np.array(cloud.colors).astype(np.float32)

        WORKSPACE_MIN = np.array([-1.2, -0.5, 0])
        WORKSPACE_MAX = np.array([-0.5, 0.5, 0.3])
        IMG_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_STD = np.array([0.229, 0.224, 0.225])

        x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
        # imagenet normalization
        colors = (colors - IMG_MEAN) / IMG_STD
        
        # final cloud
        cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
        # if show:
        #     o3d.visualization.draw_geometries([cloud_final])
        # if fname is not None:
        #     o3d.io.write_point_cloud(fname, cloud_final)
        return cloud_final, points, colors
    
    
    def xyz_rpy_to_homogeneous_matrix(self, xyz, rpy):
        """
        将 xyz 平移向量和 rpy 旋转角度转换为 4x4 的齐次变换矩阵
        :param xyz: 包含 x, y, z 坐标的列表或数组
        :param rpy: 包含滚转、俯仰、偏航角度的列表或数组，单位为弧度
        :return: 4x4 的齐次变换矩阵
        """
        # 创建旋转对象
        rotation = R.from_euler('xyz', rpy)
        # 获取旋转矩阵
        rotation_matrix = rotation.as_matrix()
        # 创建 4x4 的齐次变换矩阵
        homogeneous_matrix = np.eye(4)
        # 将旋转矩阵赋值给齐次变换矩阵的左上角 3x3 子矩阵
        homogeneous_matrix[:3, :3] = rotation_matrix
        # 将平移向量赋值给齐次变换矩阵的最后一列的前三个元素
        homogeneous_matrix[:3, 3] = xyz
        return homogeneous_matrix
    
    
    def merge_xyz_rgb(self, xyz, rgb):
        # 将点云的空间坐标(xyz)和颜色信息(rgb)合并成一个结构化数组
        # 将RGB颜色值打包成一个32位的浮点数
        # 用于创建ROS点云消息
        
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb)

        # 将颜色信息转换为 32 位无符号整数
        colors_uint32 = (rgb * 255).astype(np.uint32)
        r = (colors_uint32[:, 0] << 16)
        g = (colors_uint32[:, 1] << 8)
        b = colors_uint32[:, 2]
        rgb_packed = r | g | b

        structured_array = np.zeros(
            xyz.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32), #注意这里的rgb是32位无符号整数，负责颜色会错乱
            ],
        )
        structured_array["x"] = xyz[:, 0]
        structured_array["y"] = xyz[:, 1]
        structured_array["z"] = xyz[:, 2]
        structured_array["rgb"] = rgb_packed

        return structured_array
    


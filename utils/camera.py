import cv2 
import numpy as np
from PIL import Image
import math
import open3d as o3d

import matplotlib.pyplot as plt
cmap_jet = plt.get_cmap('jet')
cmap_magma = plt.get_cmap('magma')
cmap_spectral = plt.get_cmap('Spectral')

class K(object):
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def vec(self) -> np.array:
        """ return 1d vector """
        return np.array([self.fx, self.fy, self.cx, self.cy])

    @property
    def arr(self) -> np.array:
        """ return 3x3 matrix """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def inv(self) -> np.array:
        return np.linalg.inv(self.arr)

    def __mul__(self, scale):
        return K(self.fx * scale, self.fy * scale, self.cx * scale, self.cy * scale)
    
class DepthCamera(object):
    def __init__(self, resolution, scale = 1):
        self.scale = scale
        self.resolution_str = resolution # "HxW"
        self.intrinsics = {}  # keys: "color", "depth"
        self.distortions = {} # keys: "color", "depth"
        self.extrinsics = {}  # keys: "color_to_depth", "left_to_right if applicable
        self._T_fc = np.eye(4) # no frame conversion here
        self.config = {} # for viz, training etc
        self.device = "unknown"

        self._changed_res = False

    @staticmethod
    def from_device(device="sim"):
        if device == "clearpose":
            return RGBDCamera.default_clearpose()
        elif device == "syntodd":
            return RGBDCamera.default_syntodd()
        elif device == "sim":
            return Realsense.default_sim(min_depth=0.2, max_depth=5.0)
        else:
            return Realsense.default_real(device)

    def change_resolution(self, new_res_str):
        if not self._changed_res:
            if type(new_res_str) == str:
                w, h = [int(x) for x in new_res_str.split('x')]
            elif type(new_res_str) == tuple:
                w, h = new_res_str
            # if h / self.H != w / self.W:
            #     raise ValueError("aspect ratio must be the same")
            self.scale = h / self.H
            self.resolution_str = f"{w}x{h}"
            self._changed_res = True
        else:
            """ THIS FUNCTION SHOULD ONLY BE CALLED ONCE BY DESIGN """
            raise RuntimeError("resolution already changed")

    @property
    def resolution(self):
        """ return H,W """
        W, H =self.resolution_str.split('x')
        return int(H), int(W)
    
    @property
    def H(self):
        return self.resolution[0]
    
    @property
    def W(self):
        return self.resolution[1]

    @property
    def K_color(self):
        return K(*self.intrinsics["color"]) * self.scale
    
    @property
    def K_depth(self):
        return K(*self.intrinsics["depth"]) * self.scale
    
    @property
    def min_depth(self):
        return self.config["min_depth"]

    @property
    def max_depth(self):
        return self.config["max_depth"]
    
    @property
    def fxb_color(self):
        """ focal x baseline """
        return self.baseline * self.K_color.fx
    
    @property
    def fxb_depth(self):
        """ focal x baseline """
        return self.baseline * self.K_depth.fx
    
    @property
    def min_disp(self):
        return self.K_depth.fx * self.baseline / self.config["max_depth"]
    
    @property
    def max_disp(self):
        return self.K_depth.fx * self.baseline / self.config["min_depth"]
    
    def unnormalize_disp(self, disp):
        undisp = unnormalize_disp(disp, self.min_disp, self.max_disp, self.config["shift"])
        # for realsense stereo, there shouldn't be negative disparity
        return undisp.clip(self.min_disp, self.max_disp)

    def normalize_disp(self, disp):
        ndisp = normalize_disp(disp, self.min_disp, self.max_disp, self.config["shift"])
        return ndisp
        # for realsense stereo, there shouldn't be negative disparity
        # return ndisp.clip(self.min_disp, self.max_disp)

    @property
    def T_cl(self):
        return self._T_fc @ self.extrinsics["color_to_depth"] @ self._T_fc.T

    def viz_pointcloud(self, rgb, depth, show=False, fname = None):
        """ 
        rgb: HxWx3
        depth: HXW (meter)
        """
        assert type(rgb) == np.ndarray and type(depth) == np.ndarray, "rgb and depth must be numpy array"

        H, W = self.H, self.W
        if "auto_scale" in self.config and self.config["auto_scale"]:
            if not (H == rgb.shape[0] and W == rgb.shape[1]):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            
            if not (H == depth.shape[0] and W == depth.shape[1]):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        
        assert H == rgb.shape[0] and W == rgb.shape[1], "rgb resolution mismatch"
        assert H == depth.shape[0] and W == depth.shape[1], "depth resolution mismatch"
        assert depth.mean() > self.min_depth and depth.mean() < self.max_depth, "are you sure the depth is in meter?"

        if not np.allclose(self.T_cl, np.eye(4)):
            print("warning: did you forget to call transform_depth_to_rgb_frame()? if you did,")
            
        # if "clip_minmax" in self.config and self.config["clip_minmax"]:
        #     # depth = np.clip(depth, self.min_depth, self.max_depth), 
        #     depth[depth < self.min_depth] = 0.0
        #     depth[depth > self.max_depth] = 0.0

        depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        color_raw = o3d.geometry.Image(np.ascontiguousarray(rgb).astype(np.uint8))
        rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
            depth_scale=1., depth_trunc=10, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.W, self.H, *self.K.vec)
        pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw, intrinsic)
        if show:
            o3d.visualization.draw_geometries([pcd_rgbd])
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd_rgbd)

class RGBDCamera(DepthCamera):

    @staticmethod
    def default_nyu():
        """ copy from toolkit v2 """
        return RGBDCamera("640x480", {
            "intrinsic": [5.1885790117450188e+02, 5.1946961112127485e+02, 3.2558244941119034e+02, 2.5373616633400465e+02], # fx, fy, cx, cy
            "distortion": [2.0796615318809061e-01, -5.8613825163911781e-01, 7.2231363135888329e-04, 1.0479627195765181e-03, 4.9856986684705107e-01], # k1, k2, p1, p2, k3
        },{
            "intrinsic": [5.8262448167737955e+02, 5.8269103270988637e+02, 3.1304475870804731e+02, 2.3844389626620386e+02],
            "distortion": [-9.9897236553084481e-02, 3.9065324602765344e-01, 1.9290592870229277e-03, -1.9422022475975055e-03, -5.1031725053400578e-01]
        }, [
            [9.9997798940829263e-01, 5.0518419386157446e-03, 4.3011152014118693e-03, 2.5031875059141302e-02],
            [-5.0359919480810989e-03, 9.9998051861143999e-01, -3.6879781309514218e-03, 6.6238747008330102e-04],
            [-4.3196624923060242e-03, 3.6662365748484798e-03, 9.9998394948385538e-01, -2.9342312935846411e-04],
            [0, 0, 0, 1],
        ]
        )

    @staticmethod
    def default_clearpose():
        cam = RGBDCamera("640x480", 
        {
            # "intrinsic": [601.46000163, 601.5933431, 334.89998372, 248.15334066],  # fx, fy, cx, cy
            "intrinsic": [601.33333333, 601.33333333, 334.66666667, 248.],
            "distortion": [] # k1, k2, p1, p2, k3
        },
        {
            # "intrinsic": [601.46000163, 601.5933431, 334.89998372, 248.15334066],
            "intrinsic": [601.33333333, 601.33333333, 334.66666667, 248.],
            "distortion": []
        },
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        )
        cam.device = "clearpose"
        cam._baseline = 24.54705 / 601.33333333 # hack: so fxb is same as realsense
        return cam

    @staticmethod
    def default_syntodd():
        cam = RGBDCamera("640x480", 
        {
            "intrinsic": [613.9624633789062, 613.75634765625, 324.4471435546875, 239.1712188720703], # fx, fy, cx, cy
            "distortion": [] # k1, k2, p1, p2, k3
        },
        {
            "intrinsic": [613.9624633789062, 613.75634765625, 324.4471435546875, 239.1712188720703],
            "distortion": []
        },
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        )
        cam.device = "syntodd"
        cam._baseline = 24.54705 / 613.9624633789062 # hack: so fxb is same as realsense
        return cam

    def __init__(self, resolution, rgb_cam_params, depth_cam_params, extrinsics):
        """ resolution: hxw
            rgb_cam_params, depth_cam_params: intrinsic: kx, ky, cx, cy , distortion: k1, k2, p1, p2, k3
            extrinsics: nparray 4x4, T_{c->d}  color to depth transformation
        """
        super().__init__(resolution)
        self.intrinsics['color'] = rgb_cam_params['intrinsic']
        self.distortions['color'] =  rgb_cam_params['distortion']
        self.intrinsics['depth'] = depth_cam_params['intrinsic']
        self.distortions['depth'] =  depth_cam_params['distortion']
        self.extrinsics['color_to_depth'] = np.array(extrinsics)
        self.config['min_depth'] = 0.2
        self.config['max_depth'] = 5.0
        self.config['shift'] = 0.
        self.config['margin_left'] = 0
        self.config["aggressive_fill"] = True

    def transform_depth_to_rgb_frame(self, depth):
        H, W = self.H, self.W
        if not (H == depth.shape[0] and W == depth.shape[1]):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # project depth to color frame
        depth_1d = depth.reshape(1, H * W)
        meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        ones = np.ones((1, H * W), dtype=np.float32)
        pix_coords = np.concatenate((id_coords[0].reshape(1, -1), id_coords[1].reshape(1, -1), ones), axis=0)
        cam_points_color = (self.K_depth.inv @ pix_coords) * depth_1d # back project in color frame
        valid_mask = cam_points_color[2] > 0. # filter out invalid points
        cam_points_color = cam_points_color[:, valid_mask]
        # cam_points_color = self.T_cl[:3,:3] @ cam_points_color + self.T_cl[:3,3:] # convert to color frame

        pix_coords_color = (self.K_color.arr @ cam_points_color) # project to color frame 
        pix_coords_color[:2] /= pix_coords_color[2:3] # normalize

        proj_depth_color = np.zeros((H, W), dtype=np.float32)
        u, v = pix_coords_color[:2]
        u_left, u_right = np.floor(u).astype(np.uint32), np.ceil(u).astype(np.uint32)
        v_up, v_bottom = np.floor(v).astype(np.uint32), np.ceil(v).astype(np.uint32)

        def fill(depth_map, pred_depth, u, v):
            u, v = np.round(u).astype(np.uint32), np.round(v).astype(np.uint32)
            uv = np.vstack([u,v])
            valid_color = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
            u, v = uv[:, valid_color]
            depth_map[v, u] = pred_depth[0, valid_mask][valid_color]

        if self.config["aggressive_fill"]:
            """ fill all the nearby pixels """
            fill(proj_depth_color, depth_1d, u_left, v_up)
            fill(proj_depth_color, depth_1d, u_left, v_bottom)
            fill(proj_depth_color, depth_1d, u_right, v_up)
            fill(proj_depth_color, depth_1d, u_right, v_bottom)
        else:
            fill(proj_depth_color, depth_1d, u, v)

        return proj_depth_color
    

    # @property ambiguous 
    # def K(self):
    #     return K(*self.intrinsics["color"])

    @property 
    def K(self):
        """ caution """
        return self.K_depth

    @property
    def baseline(self):
        """ pseudo baseline for inverse depth """
        if hasattr(self, "_baseline"):
            return self._baseline
        return 1.0 / self.K_depth.fx
    
    @property
    def fxb(self):
        return self.fxb_depth
    
    # @property
    # def max_disp(self):
    #     return self.K.fx * self.baseline / self.config["min_depth"]

class Realsense(DepthCamera):
    """ Coordinate system:
        https://support.intelrealsense.com/hc/en-us/community/posts/15464823208851-Extrinsic-Camera-Calibration

        Check realsense intrinsics (Ubuntu):
        rs-enumerate-devices -c > intrisis.txt
    """

    def __init__(self, resolution="640x360", **kwargs):
        if resolution == "640x360":
            scale = 1
        elif resolution == "1280x720":
            scale = 1280 / 640
        elif resolution == "224x126":
            scale = 224 / 640 
        elif resolution == "320x256":
            scale = 320 / 640 
        elif resolution == "960x540":
            scale = 960 / 640
        elif resolution == "480x270":
            scale = 480 / 640
        else:
            raise RuntimeError("no supported resolution")
        
        super().__init__(resolution, scale)
        
        min_depth = kwargs["min_depth"] if "min_depth" in kwargs else 0.2
        max_depth = kwargs["max_depth"] if "max_depth" in kwargs else 2
        shift = kwargs["shift"] if "shift" in kwargs else 0.0
        margin_left = kwargs["margin_left"] if "margin_left" in kwargs else 0

        self.config = {
            "clip_minmax": True,
            "auto_scale": True,
            "aggressive_fill": True,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "shift": shift,
            "margin_left": margin_left
        }

        # convert to open gl frame: +Z forward, +Y downward (+X right)
        self._T_fc = np.array([
            [-1, 0, 0, 0],
            [ 0, -1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1],
        ])

    @staticmethod
    def default_sim(res="640x360", fov=71.28, t_cl=0.0, t_lr=0.055, **kwargs):
        cam = Realsense.create_sim(res, fov, t_cl, t_lr, **kwargs)
        cam.device = "sim"
        return cam

    @staticmethod
    def default_real(device="wsl"):
        if device == "wsl":
            conf = {
                "intrinsics": {
                    # Intrinsic of "Color" / 640x360 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
                    "color": np.array([455.209289550781, 455.209289550781, 317.77197265625,  179.728973388672]),
                    # Intrinsic of "Depth" / 640x360 / {Z16}
                    "depth": np.array([447.721832275391, 447.721832275391, 322.147064208984, 172.095764160156]),
                },
                "extrinsics": {
                    # Extrinsic from "Color" To "Depth"
                    "color_to_depth": np.array([
                        [0.999998,    -0.000968804, -0.0016287, -0.0149246659129858],
                        [0.000967551, 0.999999,     -0.000770266, -3.58414604306745e-06],
                        [0.00162944,  0.000768688,  0.999998, 4.43683347839396e-05],
                        [0, 0, 0, 1]
                    ]),
                    # Extrinsic from "Infrared 1" To "Infrared 2":
                    "left_to_right": np.array([
                        [1, 0, 0, -0.0551159121096134],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                }
            }
        elif device == "cwb":
            conf = {
                "intrinsics": {
                    "color": np.array([455.966003417969, 455.966003417969, 323.101226806641, 172.794128417969]),
                    "depth": np.array([450.814300537109, 450.814300537109, 318.694610595703,183.3427734375])
                },
                "extrinsics": {
                    "color_to_depth": np.array([
                        [0.999939, -0.00144325, -0.0109409, -0.0150572098791599],
                        [0.00139053, 0.999987, -0.00482466, 0.000150697553181089],
                        [0.0109478 , 0.00480915, 0.999929, -0.000134243455249816],
                        [0, 0, 0, 1]
                    ]),
                    "left_to_right": np.array([
                        [1, 0, 0, -0.0551359392702579],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                }
            }
        elif device == "fxm":
            conf = {
                "intrinsics": {
                    "color": np.array([454.445556640625, 453.388824462891, 312.513153076172, 173.245956420898]),
                    "depth": np.array([443.914642333984, 443.914642333984, 315.938354492188, 186.570434570312])
                },
                "extrinsics": {
                    "color_to_depth": np.array([
                        [0.999998  ,       0.00216086 ,      1.59334e-05, -0.0150564182549715 -0.01], 
                        [-0.00216081 ,      0.999995 ,       -0.00245083, -3.06203619402368e-05 -0.01],
                        [-2.12292e-05,      0.00245079 ,     0.999997,   0.000343983672792092],
                        [0, 0, 0, 1]
                    ]),
                    "left_to_right": np.array([
                        [1, 0, 0, -0.0547803528606892],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                }
            }
        elif device == "jav":
            conf = {
                "intrinsics": {
                    "color": np.array([453.822387695312, 453.152984619141, 317.806304931641, 174.375793457031]),
                    "depth": np.array([445.645629882812, 445.645629882812, 319.318328857422, 185.445999145508])
                },
                "extrinsics": {
                    "color_to_depth": np.array([
                        [0.999994   ,       0.000278227 ,      0.00333567, -0.0150431916117668], 
                        [-0.000274474 ,     0.999999,       -0.00112542, -2.82119053736096e-05],
                        [-0.00333598 ,      0.0011245 ,     0.999994,   0.000287492759525776],
                        [0, 0, 0, 1]
                    ]),
                    "left_to_right": np.array([
                        [1, 0, 0, -0.0547580868005753],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                }
            }
        elif device == "d435":
            # 640x360
            conf = {
                "intrinsics": {
                    "color": np.array([455.379180908203, 455.442810058594, 319.510498046875, 190.218185424805]),
                    # "depth": np.array([323.665466308594, 323.665466308594, 324.088043212891, 175.596954345703])
                    "depth": np.array([455.379180908203, 455.442810058594, 319.510498046875, 190.218185424805]),
                },
                "extrinsics": {
                    "color_to_depth": np.array([
                        [0.999823   ,       -0.0178564,     -0.00593102, -0.0150311784818769], 
                        [0.0178789 ,     0.999833,          0.0037614, 3.65326523024123e-05],
                        [0.00586286 ,     -0.00386678,     0.999975,   -0.00051211315440014],
                        [0, 0, 0, 1]
                    ]),
                    "left_to_right": np.array([
                        [1, 0, 0, -0.0547580868005753],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                }
            }
        else:
            raise RuntimeError("unknown real device (no intrinsics/extrinsics), please call Realsense.create_real() instead.")

        cam = Realsense.create_real("640x360", conf)
        cam.device = device

        if device == "d435": # hack
            cam.config["min_depth"] = 0.2
            cam.config["max_depth"] = 5.0
        return cam

    @staticmethod
    def create_sim(res, fov, t_cl, t_lr, **kwargs):
        camera = Realsense(res, **kwargs)
        H, W = camera.resolution
        H, W = int(H / camera.scale), int(W / camera.scale) # ! convert to standard resolution (640x360)
        fx = W / (2 * math.tan(math.radians(fov) / 2))
        K = np.array([fx, fx, W/2-0.5, H/2-0.5])
        T_cl, T_lr = np.eye(4), np.eye(4)
        T_cl[0,3] = -t_cl
        T_lr[0,3] = -t_lr
        conf = {
            "intrinsics": {
                "color": K,
                "depth": K
            },
            "extrinsics": {
                "color_to_depth": T_cl,
                "left_to_right": T_lr
            }
        }
        camera._import_conf(conf)
        return camera
    
    @staticmethod
    def create_real(res, conf):
        camera = Realsense(res)
        camera._import_conf(conf)
        return camera
        
    def _import_conf(self, conf = None):
        #  Intrinsic of "Color" / 640x360 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y8/Y16}
        self.intrinsics["color"]= conf["intrinsics"]["color"]# 
        #  Intrinsic of "Depth" / 640x360 / {Z16}, same as "Infrared 1" 
        self.intrinsics["depth"] = conf["intrinsics"]["depth"] #
        
        # Extrinsic from "Color" to "Depth"
        self.extrinsics["color_to_depth"] = conf["extrinsics"]["color_to_depth"]
        # Extrinsic from "Infrared 1" to "Infrared 2"
        self.extrinsics["left_to_right"] = conf["extrinsics"]["left_to_right"]

    @property 
    def K(self):
        """ caution """
        return self.K_color
    
    @property
    def T_cr(self):
        return self.T_cl @ self.T_lr
    
    @property
    def T_lr(self):
        return self._T_fc @ self.extrinsics["left_to_right"] @ self._T_fc.T
    
    @property
    def focal(self):
        return self.K_color.fx

    @property
    def fxb(self):
        return self.fxb_color
    
    @property
    def baseline(self):
        return (self._T_fc @ self.extrinsics["left_to_right"] @ self._T_fc.T )[0,3]
    
    def unnormalize_disp(self, disp):
        ndisp = unnormalize_disp(disp, self.min_disp, self.max_disp, self.config["shift"])
        # for realsense stereo, there shouldn't be negative disparity
        return ndisp.clip(self.min_disp, self.max_disp)

    def normalize_disp(self, disp):
        ndisp = normalize_disp(disp, self.min_disp, self.max_disp, self.config["shift"])
        return ndisp
        # for realsense stereo, there shouldn't be negative disparity
        # return ndisp.clip(self.min_disp, self.max_disp)

    def transform_cropped_depth_to_rgb_frame(self, depth, cropped_K):
        if np.allclose(self.T_cl, np.eye(4)):
            # print("warning: no depth to color frame transformation needed!")
            return depth

        H, W = depth.shape[:2]
        off_x = int(self.K.cx - cropped_K[0,2]) # see also, stereo_datasets.py
        off_y = int(self.K.cy - cropped_K[1,2])
        depth_1d = depth.reshape(1, H * W)
        meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        ones = np.ones((1, H * W), dtype=np.float32)
        pix_coords = np.concatenate((id_coords[0].reshape(1, -1), id_coords[1].reshape(1, -1), ones), axis=0)
        cam_points_color = (np.linalg.inv(cropped_K) @ pix_coords) * depth_1d # back project in ir frame
        valid_mask = cam_points_color[2] > 0. # filter out invalid points
        cam_points_color = cam_points_color[:, valid_mask]
        cam_points_color = self.T_cl[:3,:3] @ cam_points_color + self.T_cl[:3,3:] # convert to color frame

        pix_coords_color = (self.K_color.arr @ cam_points_color) # project to color frame 
        pix_coords_color[:2] /= pix_coords_color[2:3] # normalize

        proj_depth_color = np.zeros((self.H, self.W), dtype=np.float32)
        u, v = pix_coords_color[:2]
        u_left, u_right = np.floor(u).astype(np.uint32), np.ceil(u).astype(np.uint32)
        v_up, v_bottom = np.floor(v).astype(np.uint32), np.ceil(v).astype(np.uint32)

        def fill(depth_map, pred_depth, u, v):
            u, v = np.round(u).astype(np.uint32), np.round(v).astype(np.uint32)
            uv = np.vstack([u,v])
            valid_color = (uv[0] >= 0) & (uv[0] < self.W) & (uv[1] >= 0) & (uv[1] < self.H)
            u, v = uv[:, valid_color]
            depth_map[v, u] = pred_depth[0, valid_mask][valid_color]

        if False and self.config["aggressive_fill"]:
            """ fill all the nearby pixels """
            fill(proj_depth_color, depth_1d, u_left, v_up)
            fill(proj_depth_color, depth_1d, u_left, v_bottom)
            fill(proj_depth_color, depth_1d, u_right, v_up)
            fill(proj_depth_color, depth_1d, u_right, v_bottom)
        else:
            fill(proj_depth_color, depth_1d, u, v)

        return proj_depth_color[off_y:off_y+H, off_x:off_x+W]

    def transform_depth_to_rgb_frame(self, depth):
        if np.allclose(self.T_cl, np.eye(4)) or self.device == "d435": # d435 is already aligned
            # print("warning: no depth to color frame transformation needed")
            return depth

        H, W = self.H, self.W
        if self.config["auto_scale"]:
            if not (H == depth.shape[0] and W == depth.shape[1]):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        
        assert H == depth.shape[0] and W == depth.shape[1], "depth resolution mismatch"

        # project depth to color frame
        depth_1d = depth.reshape(1, H * W)
        meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        ones = np.ones((1, H * W), dtype=np.float32)
        pix_coords = np.concatenate((id_coords[0].reshape(1, -1), id_coords[1].reshape(1, -1), ones), axis=0)
        cam_points_color = (self.K_depth.inv @ pix_coords) * depth_1d # back project in color frame
        valid_mask = cam_points_color[2] > 0. # filter out invalid points
        cam_points_color = cam_points_color[:, valid_mask]
        cam_points_color = self.T_cl[:3,:3] @ cam_points_color + self.T_cl[:3,3:] # convert to color frame

        pix_coords_color = (self.K_color.arr @ cam_points_color) # project to color frame 
        pix_coords_color[:2] /= pix_coords_color[2:3] # normalize

        proj_depth_color = np.zeros((H, W), dtype=np.float32)
        u, v = pix_coords_color[:2]
        u_left, u_right = np.floor(u).astype(np.uint32), np.ceil(u).astype(np.uint32)
        v_up, v_bottom = np.floor(v).astype(np.uint32), np.ceil(v).astype(np.uint32)

        def fill(depth_map, pred_depth, u, v):
            u, v = np.round(u).astype(np.uint32), np.round(v).astype(np.uint32)
            uv = np.vstack([u,v])
            valid_color = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
            u, v = uv[:, valid_color]
            depth_map[v, u] = pred_depth[0, valid_mask][valid_color]

        if self.config["aggressive_fill"]:
            """ fill all the nearby pixels """
            fill(proj_depth_color, depth_1d, u_left, v_up)
            fill(proj_depth_color, depth_1d, u_left, v_bottom)
            fill(proj_depth_color, depth_1d, u_right, v_up)
            fill(proj_depth_color, depth_1d, u_right, v_bottom)
        else:
            fill(proj_depth_color, depth_1d, u, v)

        return proj_depth_color
    
    def viz_colormap_grid(self, depths, consistent = True, cmap = "turbo", fname = None):
        """ visualize multiple depths in a grid 
            depth: [HxW]*B or BxHxW
            consistent: if set True, use the same max,min for all depths after clip
        """
        if type(depths) == list:
            depths = np.vstack([np.expand_dims(d,axis=0) for d in depths])
        
        elif type(depths) == np.ndarray:
            if len(depths.shape) == 3:
                depths = np.vstack(depths)
            elif len(depths.shape) == 2:
                depths = np.expand_dims(depths, axis=0)
            else:
                raise RuntimeError("depths must be 2d or 3d array")
        
        B = depths.shape[0]

        if self.config["clip_minmax"]:
            # depth = np.clip(depth, self.min_depth, self.max_depth), 
            depths[depths < self.min_depth] = 0.0 # mark invalid depth as 0
            depths[depths > self.max_depth] = 0.0
            
        if consistent:
            maxmin = [[np.max(depths), np.min(depths)]] * B
        else:
            maxmin = [[np.max(depths[i]), np.min(depths[i])] for i in range(B)]

        nrows = math.ceil(math.sqrt(B)) #max(1, math.floor(math.sqrt(B)))
        ncols = int(B / nrows) + (B % nrows > 0)
        
        cmap = plt.get_cmap(cmap)
        # grid_size = max(nrows, ncols)
        # fig = plt.figure(figsize=(grid_size * 5, grid_size * 5), frameon=False)

        for i in range(B):
            plt.subplot(nrows, ncols, i+1)
            img = cmap((depths[i] - maxmin[i][1]) / (maxmin[i][0] - maxmin[i][1]))
            im = plt.imshow(img)
            plt.colorbar(im) #, shrink=0.5
            
        if fname is not None:
            plt.savefig(fname, facecolor="white", transparent=False)
            plt.close()
        else:
            plt.show()

    def viz_cropped_pointcloud(self, K, rgb, depth, show=False, fname=None):
        """ visualize point cloud which is random cropped from the original image
            rgb: HxWx3
            depth: HXW (meter)
        """
        assert type(rgb) == np.ndarray and type(depth) == np.ndarray, "rgb and depth must be numpy array"
        assert rgb.shape[:2] == depth.shape, "rgb & depth do not match"

        H, W = rgb.shape[:2]
        depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        color_raw = o3d.geometry.Image(np.ascontiguousarray(rgb).astype(np.uint8))
        rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, 
            depth_scale=1., depth_trunc=10, convert_rgb_to_intensity=False)
        K_vec = [K[0,0], K[1,1], K[0,2], K[1,2]]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, *K_vec)
        pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw, intrinsic)
        if show:
            o3d.visualization.draw_geometries([pcd_rgbd])
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd_rgbd)

def normalize_disp(disp, min_disp=5, max_disp=120, shift=0):
    """ normalize disparity to [0,1], 
        ideally, with mean value '''shifted''' around 0.5
    """
    disp = (disp - min_disp) / (max_disp - min_disp) + shift  # shift mean after normalize
    return disp 

def unnormalize_disp(disp, min_disp=5, max_disp=120, shift=0):
    """ unnormalize disparity from [0,1] to [min_disp, max_disp] """
    return min_disp + (disp - shift) * (max_disp - min_disp) # shift mean before unnormalize

def plot_error_map(error_map):
    """ error_map: BxHxW """
    B, H, W = error_map.shape
    
    nrows = math.ceil(math.sqrt(B))
    ncols = int(B / nrows) + (B % nrows > 0)
    
    dummy_dpi = 80
    fig = plt.figure(figsize=(ncols * (W+20) / dummy_dpi, nrows * H / dummy_dpi), dpi=dummy_dpi)
    for i in range(B):
        ax1 = fig.add_subplot(nrows, ncols, i+1)
        im = ax1.imshow(error_map[i], cmap="turbo")
        plt.colorbar(im, ax=ax1)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return Image.fromarray(img)

def plot_loss_terms(loss_terms, fname):
    T = len(loss_terms)
    x = np.linspace(0, T, T)
    y = loss_terms
    plt.figure()
    plt.plot(x, y, label="reweighted loss terms")
    plt.legend(loc="upper right")
    plt.xlabel("diffusion step")
    plt.ylabel("weights")
    plt.savefig(fname)
    plt.close()
    
def plot_uncertainties(uncertainties):
    """ uncertainties: BxHxW 
        plot matplotlib into PIL image
    """
    B, H, W = uncertainties.shape
    
    nrows = math.ceil(math.sqrt(B))
    ncols = int(B / nrows) + (B % nrows > 0)
    
    dummy_dpi = 80
    fig = plt.figure(figsize=(ncols * (W+20) / dummy_dpi, nrows * H / dummy_dpi), dpi=dummy_dpi)
    for i in range(B):
        ax1 = fig.add_subplot(nrows, ncols, i+1)
        im = ax1.imshow(uncertainties[i], cmap="plasma")
        plt.colorbar(im, ax=ax1)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(img)

def plot_denoised_images(config, intermediates, pred_disps_ss, normalized_rgb=None, raw_disp=None, left_image = None, right_image = None, mask = None, **kwargs):
    """ denoised_images: BxTxHxW intermediate steps
        pred_disps_ss: BxHxW   final prediction, scale and shift aligned
        cond_image: Bx3xHxW
        gt_image: Bx1xHxW
        left_image: Bx3xHx(W+Margin)
        right_image: Bx3xHx(W+Margin)
    """
    to_numpy = lambda x: x.cpu().numpy().copy().transpose(0,2,3,1) if x is not None else x # [B,H,W,C]
    normalized_rgb, raw_disp, left_image, right_image = map(to_numpy, [normalized_rgb, raw_disp, left_image, right_image])

    B, T, H, W = intermediates.images_sampled_prev.shape
    num_inter = 5 if config.plot_intermediate_images else 1
    nrows, ncols = B * num_inter, T + 2 # T : sample_pred, pred_orig, perturbed_orig, pred_prev, perturbed_prev,  2: pred_ss, gt, 
    if normalized_rgb is not None:
        ncols += 1

    if left_image is not None and right_image is not None:
        ncols += 2
        margin_left = left_image.shape[-2] - W
        crop_left = lambda x: x[:, margin_left:, :]

    if 'raw_depth' in kwargs and config.prediction_space == "depth":
        raw_depth = to_numpy(kwargs['raw_depth'])
        plot_raw_depth = True
        ncols += 1
    elif 'sim_disp' in kwargs:
        ncols += 1
        from utils.utils import Normalizer
        norm = Normalizer.from_config(config)
        if config.ssi:
            sim_disp = kwargs['sim_disp']
            sim_valid = sim_disp != 0 # no exactly correct but ok for visualization
            sim_disp[sim_valid] =  sim_disp[sim_valid] / 2.0 + 0.5
        else:
            sim_disp = norm.denormalize(kwargs['sim_disp'])
            # norm_sim[sim_valid] = 
        raw_depth = to_numpy(sim_disp)
        plot_raw_depth = True
    else:
        plot_raw_depth = False
        
    if config.plot_mask and mask is not None:
        ncols += 1

    grid = Image.new("RGB", size=(ncols * W, nrows * H))

    def viz_normalizer(x, i, apply_mask=True, low_p=0, high_p=100):
        # assert x.min() >= 0.0, "bug"
        valid =  mask[i].cpu().numpy().astype(bool)[0] if apply_mask and mask[i] is not None else x > 0.0
        low, high = np.percentile(x[valid], (low_p, high_p))
        x[valid] = (x[valid] - low) / (high - low + 1e-10) 
        x[~valid] = 0.0
        return x
        
    depth_to_grayscale = lambda x, i: (viz_normalizer((x+1)/2, i, apply_mask=False, low_p=2, high_p=98).clip(0, 1) * 255.0).astype(np.uint8) # from ~[-1,1] -> [0,255]
    depth_to_grayscale_after_mask = lambda x, i: (viz_normalizer(x, i, apply_mask=True, low_p=0, high_p=100) * 255.0).astype(np.uint8).clip(0, 255) # from ~[-1,1] -> [0,255]

    gray_to_jet = lambda x: (cmap_spectral(x/255.0)*255.)[...,:3].astype(np.uint8) #x
    # diff_to_rgb = lambda x: ((x + 1.0) / 2 *255.0).astype(np.uint8).clip(0, 255) # from [-1,1] -> [0,255]
    diff_to_rgb = lambda x, m: np.clip(((x - x[m].min()) / (x[m].max() - x[m].min() + 1e-6)) * 255.0, 0, 255).astype(np.uint8) # from [-r,r] -> [0,255]
    clip = lambda x: np.clip(x, -100, 100) #-config.clip_sample_range, config.clip_sample_range
    rgb_to_rgb = lambda x: ((x + 1)*127.5).astype(np.uint8).clip(0, 255) # [-1,1] to [0,255] 
    
    for b in range(B):
        # if b > 3: break # too many !
        for i in range(num_inter):
            if i == 0:
                denoised_images = intermediates.images_sampled_prev.cpu().numpy().copy()
            elif i == 1:
                denoised_images = intermediates.images_pred_orig.cpu().numpy().copy()
            elif i == 2:
                denoised_images = intermediates.images_perturbed_orig.cpu().numpy().copy()
            elif i == 3:
                denoised_images = intermediates.images_pred_prev.cpu().numpy().copy()
            elif i == 4:
                denoised_images = intermediates.images_purturbed_pred_prev.cpu().numpy().copy()

            for j in range(T):
                img = Image.fromarray(gray_to_jet(diff_to_rgb(clip(denoised_images[b, j].copy()), mask[b,0].cpu().numpy().astype(bool))))
                grid.paste(img, box=(j * W, (b*num_inter+i) * H))

            pred_ss = Image.fromarray(gray_to_jet(depth_to_grayscale_after_mask(pred_disps_ss[b].copy(), b)))
            grid.paste(pred_ss, box=(T * W, (b*num_inter+i) * H))

            gt = Image.fromarray(gray_to_jet(depth_to_grayscale(raw_disp[b, ..., 0].copy(), b)))
            grid.paste(gt, box=((T+1) * W, (b*num_inter+i) * H))

            if plot_raw_depth:
                raw = Image.fromarray(gray_to_jet(depth_to_grayscale_after_mask(raw_depth[b, ..., 0].copy(), b)))
                grid.paste(raw, box=((T+2) * W, (b*num_inter+i) * H))

            offset = 1 if plot_raw_depth else 0

            if normalized_rgb is not None:
                rgb = Image.fromarray(rgb_to_rgb(normalized_rgb[b].copy()))
                grid.paste(rgb, box=((T+2+offset) * W, (b*num_inter+i) * H))
                offset += 1

            if left_image is not None and right_image is not None:
                left = Image.fromarray(rgb_to_rgb(crop_left(left_image[b].copy())))
                grid.paste(left, box=((T+2+offset) * W, (b*num_inter+i) * H))

                right = Image.fromarray(rgb_to_rgb(crop_left(right_image[b].copy())))
                grid.paste(right, box=((T+3+offset) * W, (b*num_inter+i) * H))
                offset += 2
            
            if config.plot_mask and mask is not None:
                mask_img = Image.fromarray((mask[b,0]*255).cpu().numpy().astype(np.uint8))
                # mask_img = Image.fromarray(np.ones_like(mask[b,0]))
                grid.paste(mask_img, box=((T+2+offset) * W, (b*num_inter+i) * H))

    return grid

def make_image_grid__(images: np.array, cond_images: np.array = None, gt_images: np.array = None,
                    rows: int=0, cols: int=0, resize: int = None) -> Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """

    # FIXME!!!
    N = images.shape[0]
    if rows == 0:
        rows = np.ceil(np.sqrt(N)).astype(np.int32)
    if cols == 0:
        cols = N // rows

    repeat = 1
    if cond_images is not None:
        assert images.shape[0] == cond_images.shape[0]
        repeat += 1
    
    if gt_images is not None:
        assert gt_images.shape[0] == cond_images.shape[0]
        repeat += 1

    if resize is not None: ## TODO refactory
        images = [img.resize((resize, resize)) for img in images]
        if cond_images is not None:
            cond_images = [img.resize((resize, resize)) for img in cond_images]
        if gt_images is not None:
            gt_images = [img.resize((resize, resize)) for img in gt_images]
    

    h, w = images[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w * repeat, rows * h))

    # hack zip
    if cond_images is None:
        cond_images = [None] * N
        gt_images = [None] * N

    consistent_maxmin = []
    for i, (img,cond,gt) in enumerate(zip(images,cond_images,gt_images)):
        # if img.shape[2] == 1:
        #     img = np.tile(img, (1,1,3))
        # if gt.shape[2] == 1:
        #     gt = np.tile(gt, (1,1,3))

        assert gt is not None
        consistent_min = np.min([np.min(img), np.min(gt)])
        consistent_max = np.max([np.max(img), np.max(gt)])
        img = np.clip(img, consistent_min, consistent_max)
        gt = np.clip(gt, consistent_min, consistent_max)
        consistent_maxmin.append([consistent_min, consistent_max])

        img = Image.fromarray((cmap_jet(img[...,0] / consistent_max)[...,:3]*255).astype(np.uint8)) 
      

        # img = Image.fromarray((img*255.0).astype(np.uint8))
        # gt = Image.fromarray((gt*255.0).astype(np.uint8))
        grid.paste(img, box=((i % cols) * repeat * w , i // cols * h))

        if gt is not None:
            gt = Image.fromarray((cmap_jet(gt[...,0] / consistent_max)[...,:3]*255).astype(np.uint8))
            # gt = Image.fromarray((gt*255.0).astype(np.uint8))
            # gt = Image.fromarray(((gt+1.)/2*255.0).astype(np.uint8))
            grid.paste(gt, box=(((i % cols) * repeat + 1) * w, i // cols  * h ))

        if cond is not None:
            cond = Image.fromarray(((cond+1.)/2.*255.0).astype(np.uint8))
            grid.paste(cond, box=(((i % cols) * repeat +2) * w, i // cols  * h ))
    return grid, consistent_maxmin


if __name__ == "__main__":
    realsense = Realsense.default_real()
    print(realsense.K.vec)

    min_disp = 20
    max_disp = 120
    shift = 0.2
    assert normalize_disp(20, min_disp=min_disp, max_disp=max_disp) == 0
    assert normalize_disp(120, min_disp=min_disp, max_disp=max_disp) == 1
    assert normalize_disp(30, min_disp=min_disp, max_disp=max_disp) == 0.1
    assert normalize_disp(70, min_disp=min_disp, max_disp=max_disp) == 0.5

    assert normalize_disp(0, min_disp=min_disp, max_disp=max_disp) == -0.2

    assert normalize_disp(20, min_disp=min_disp, max_disp=max_disp, shift=shift) == shift
    assert normalize_disp(0, min_disp=min_disp, max_disp=max_disp, shift=shift) == -0.2+shift
    assert normalize_disp(120, min_disp=min_disp, max_disp=max_disp,  shift=shift) == 1 + shift

    assert unnormalize_disp(0, min_disp=min_disp, max_disp=max_disp) == 20
    assert unnormalize_disp(1, min_disp=min_disp, max_disp=max_disp) == 120
    assert unnormalize_disp(0.5, min_disp=min_disp, max_disp=max_disp) == 70

    assert unnormalize_disp(shift, min_disp=min_disp, max_disp=max_disp, shift=shift) == 20
    assert unnormalize_disp(shift+1, min_disp=min_disp, max_disp=max_disp, shift=shift) == 120
    assert unnormalize_disp(shift+0.5, min_disp=min_disp, max_disp=max_disp, shift=shift) == 70

    
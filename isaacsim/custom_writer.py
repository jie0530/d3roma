import os
import io
import json, math, copy
import numpy as np
# import cv2
import warp as wp
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# import open3d as o3d
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, BasicWriter, WriterRegistry

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def colorize_normals(data):
    colored_data = ((data * 0.5 + 0.5) * 255).astype(np.uint8)
    return colored_data

class ColorWriter(BasicWriter): 
    def __init__(
        self,
        **kwargs
    ):  
        self.version = "0.0.1"
        if "semantic_segmentation" in kwargs:
            del kwargs["semantic_segmentation"]
        if "distance_to_image_plane" in kwargs:
            del kwargs["distance_to_image_plane"]
        if "pointcloud" in kwargs:
            del kwargs["pointcloud"]

        if "disparity" in kwargs:
            del kwargs["disparity"]

        interval = kwargs.pop("interval", 1)
        ticker = kwargs.pop("ticker", None)

        if "start_sequence_id" in kwargs: # keep it simple here
            start_sequence_id = kwargs["start_sequence_id"]
            assert start_sequence_id >= 0, "start_sequence_id must be >= 0"
            del kwargs["start_sequence_id"]

        super().__init__(**kwargs)

        self._frame_id = 0
        self._sequence_id = start_sequence_id
        self._start_sequence_id = start_sequence_id
        self._interval = interval
        self._ticker = ticker 
        if self._ticker is None:
            self._ticker = lambda: self._frame_id

    def write(self, data: dict):
        if self._ticker()[0] == "rgb":
            for annotator, val in data["annotators"].items():
                if annotator.startswith("rgb"):
                    file_path = f"{self._output_dir}/{self._sequence_id:04d}_color.png"
                    self._backend.write_image(file_path, val["RenderProduct_CameraRGB"]["data"])
            # print(f"rendered color {self._sequence_id:04d}")
            self._sequence_id += 1
        self._frame_id += 1
    
    def _write_rgb(self, data: dict, render_product_path: str, annotator: str):
        file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        self._backend.write_image(file_path, data[annotator])
    
    def on_final_frame(self): # reset
        self._frame_id = 0
        self._sequence_id = self._start_sequence_id

class GtWriter(BasicWriter):
    """ not only render depth, but also render semantic / masks / pointcloud, etc. """

    def __init__(self, interval=1, depth_sensor_cfg=dict(), **kwargs):
        self.version = "0.0.1"
        ticker = kwargs.pop("ticker")

        conifg = copy.copy(kwargs)
        # kwargs = dict(conifg["writer_config"])

        if "rgb" in kwargs:
            del kwargs["rgb"]

        if "disparity" in kwargs: # hack
            self.render_disparity = kwargs["disparity"]
            self.depth_sensor_cfg = depth_sensor_cfg
            self.set_render_disparity()
            del kwargs["disparity"]
        else:
            self.render_disparity = False
        # kwargs["pointcloud_include_unlabelled"] = True

        if "start_sequence_id" in kwargs:
            start_sequence_id = kwargs["start_sequence_id"] 
            assert start_sequence_id >= 0, "start_sequence_id must be >= 0"
            del kwargs["start_sequence_id"]

        

        super().__init__(**kwargs)
        self._frame_id = 0
        self._sequence_id = start_sequence_id
        self._start_sequence_id = start_sequence_id
        self._interval = interval
        self._ticker = ticker
        self._last_tick = None

    def set_render_disparity(self):
        FOV = np.deg2rad(self.depth_sensor_cfg["fov"])
        W = self.depth_sensor_cfg["resolution"][0] 
        # H = cfg["depth_sensor"]["resolution"][1]
        focal = W / (2 * math.tan(FOV / 2))
        # assert np.allclose(focal, 446.31), "do you have the correct focal length?"
        
        baseline = self.depth_sensor_cfg["placement"]["rgb_to_right_ir"] - self.depth_sensor_cfg["placement"]["rgb_to_left_ir"]
        assert np.isclose(baseline, 0.055), "wrong baseline"
        self.fxb = focal * baseline

    def write(self, data: dict):

        def write_exr(path, data, exr_flag=None):
            """ fix for isaac-sim 2022.2.1 """
            import imageio
            if isinstance(data, wp.array):
                data = data.numpy()

            # Download freeimage dll, will only download once if not present
            # from https://imageio.readthedocs.io/en/v2.8.0/format_exr-fi.html#exr-fi
            imageio.plugins.freeimage.download()
            if exr_flag == None:
                exr_flag = imageio.plugins.freeimage.IO_FLAGS.EXR_ZIP

            exr_bytes = imageio.imwrite(
                imageio.RETURN_BYTES,
                data,
                format="exr",
                flags=exr_flag,
            )
            self._backend.write_blob(path, exr_bytes)

        if self._ticker()[0] == "gt":
            if self._last_tick is not None and self._ticker()[1] == self._last_tick:
                return  # hack to avoid duplicate frames (only happens for GT writer on isaac-sim 2023 hotfix)
            for annotator, val in data["annotators"].items():
                if annotator.startswith("distance_to_image_plane"):
                    # file_path = f"{self._output_dir}/{self._sequence_id:04d}_depth.png"
                    # self._backend.write_image(file_path, (data[annotator]*1000).astype(np.uint16))
                    # file_path = f"{self._output_dir}/{self._sequence_id:04d}_depth.npy"
                    # self._backend.write_array(file_path, data[annotator])

                    file_path_exr = f"{self._output_dir}/{self._sequence_id:04d}_depth.exr"
                    # self._backend.write_exr(file_path_exr, data[annotator])
                    # cv2.imwrite(file_path_exr, data[annotator])
                    write_exr(file_path_exr, val["RenderProduct_CameraDepth"]["data"])

                    if self.render_disparity:
                        assert self.fxb is not None, "please call set_render_disparity() first"
                        disparity = self.fxb / val["RenderProduct_CameraDepth"]["data"]
                        # file_path = f"{self._output_dir}/{self._sequence_id:04d}_disp.npy"
                        file_path_exr = f"{self._output_dir}/{self._sequence_id:04d}_disp.exr"
                        # self._backend.write_array(file_path, disparity)
                        # self._backend.write_exr(file_path_exr, disparity)
                        # cv2.imwrite(file_path_exr, disparity)
                        write_exr(file_path_exr, disparity)
                        
                if annotator.startswith("semantic_segmentation"):
                    semantic_seg_data = val["RenderProduct_CameraDepth"]["data"]
                    height, width = semantic_seg_data.shape[:2]

                    file_path = (f"{self._output_dir}/{self._sequence_id:04d}_mask.png")
                    if self.colorize_semantic_segmentation:
                        semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
                        self._backend.write_image(file_path, semantic_seg_data)
                    else:
                        semantic_seg_data = semantic_seg_data.view(np.uint32).reshape(height, width)
                        self._backend.write_image(file_path, semantic_seg_data)

                    id_to_labels = val["RenderProduct_CameraDepth"]["idToLabels"]
                    file_path = f"{self._output_dir}/{self._sequence_id:04d}_mask.json"
                    buf = io.BytesIO()
                    buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
                    self._backend.write_blob(file_path, buf.getvalue())

                if annotator.startswith("normals"):
                    normals_data = val["RenderProduct_CameraDepth"]["data"]
                    file_path_normal = f"{self._output_dir}/{self._sequence_id:04d}_normal.png"
                    colorized_normals_data = colorize_normals(normals_data)
                    self._backend.write_image(file_path_normal, colorized_normals_data)

                if annotator.startswith("pointcloud"):
                    pointcloud_data = data[annotator]["data"]
                    file_path = f"{self._output_dir}/{self._sequence_id:04d}_pcd.npy"
                    self._backend.write_array(file_path, pointcloud_data)

                    pointcloud_rgb = data[annotator]["info"]["pointRgb"].reshape(-1, 4)
                    rgb_file_path = f"{self._output_dir}/{self._sequence_id:04d}_pcd_rgb.npy"
                    self._backend.write_array(rgb_file_path, pointcloud_rgb)

                    """ pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pointcloud_data.astype(np.float32).reshape(-1, 3))
                    o3d.io.write_point_cloud(file_path, pcd) """
            self._last_tick = self._ticker()[1]
            # print(f"rendered gt {self._sequence_id:04d}")
            self._sequence_id += 1
        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0
        self._sequence_id = self._start_sequence_id

class IRWriter(Writer):
    def __init__(
        self,
        output_dir,
        start_sequence_id=0,
        interval=1,
        ticker=None,
    ):
        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        self._output_dir = output_dir
        self._interval = interval

        assert start_sequence_id >= 0, "start_sequence_id must be >= 0"
        self._frame_id = 0
        self._sequence_id = start_sequence_id
        self._start_sequence_id = start_sequence_id
        self._ticker = ticker
        if self._ticker is None:
            self._ticker = lambda: self._frame_id

    def write(self, data: dict):
        if self._ticker()[0] == "ir":
            for annotator in data.keys():
                if annotator.startswith("rgb"):
                    # ir_name = 'ir_l' if 'Left' in annotator else 'ir_r'
                    ir_name = 'ir_l' if '01' in annotator else 'ir_r' # HACK
                    filename = f"{self._output_dir}/{self._sequence_id:04d}_{ir_name}.png"
                    self.backend.write_image(filename, rgb2gray(data[annotator]).astype(np.uint8))
            # print(f"rendered ir {self._sequence_id:04d}")
            self._sequence_id += 1    
        self._frame_id += 1
    
    def on_final_frame(self):
        self._frame_id = 0
        self._sequence_id = self._start_sequence_id


    
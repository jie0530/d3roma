from typing import Dict, Tuple
from abc import ABC, abstractmethod 
import os, random, time, json, math, copy
import numpy as np
import omni, carb
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade, UsdGeom
import omni.replicator.core as rep
import transforms3d as t3d
from utils_func import find_next_sequence_id
from omni.isaac.core.utils.prims import get_prim_at_path

def compute_obb(bbox_cache: UsdGeom.BBoxCache, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Oriented Bounding Box (OBB) of a prim

    .. note::

        * The OBB does not guarantee the smallest possible bounding box, it rotates and scales the default AABB.
        * The rotation matrix incorporates any scale factors applied to the object.
        * The `half_extent` values do not include these scaling effects.

    Args:
        bbox_cache (UsdGeom.BBoxCache): USD Bounding Box Cache object to use for computation
        prim_path (str): Prim path to compute OBB for

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following OBB information:
            - The centroid of the OBB as a NumPy array.
            - The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
            - The half extent of the OBB as a NumPy array.

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> # 1 stage unit length cube centered at (0.0, 0.0, 0.0)
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> half_extent
        [0.5 0.5 0.5]
        >>>
        >>> # the same cube rotated 45 degrees around the z-axis
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[ 0.70710678  0.70710678  0.        ]
         [-0.70710678  0.70710678  0.        ]
         [ 0.          0.          1.        ]]
        >>> half_extent
        [0.5 0.5 0.5]
    """
    # Compute the BBox3d for the prim
    prim = get_prim_at_path(prim_path)
    bound = bbox_cache.ComputeWorldBound(prim)

    # Compute the translated centroid of the world bound
    centroid = bound.ComputeCentroid()

    # Compute the axis vectors of the OBB
    # NOTE: The rotation matrix incorporates the scale factors applied to the object
    rotation_matrix = bound.GetMatrix().ExtractRotationMatrix()
    x_axis = rotation_matrix.GetRow(0)
    y_axis = rotation_matrix.GetRow(1)
    z_axis = rotation_matrix.GetRow(2)

    # Compute the half-lengths of the OBB along each axis
    # NOTE the size/extent values do not include any scaling effects
    half_extent = bound.GetRange().GetSize() * 0.5

    return np.array([*centroid]), np.array([[*x_axis], [*y_axis], [*z_axis]]), np.array(half_extent)

def get_obb_corners(centroid: np.ndarray, axes: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    """Computes the corners of the Oriented Bounding Box (OBB) from the given OBB information

    Args:
        centroid (np.ndarray): The centroid of the OBB as a NumPy array.
        axes (np.ndarray): The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
        half_extent (np.ndarray): The half extent of the OBB as a NumPy array.

    Returns:
        np.ndarray: NumPy array of shape (8, 3) containing each corner location of the OBB

        :math:`c_0 = (x_{min}, y_{min}, z_{min})`
        |br| :math:`c_1 = (x_{min}, y_{min}, z_{max})`
        |br| :math:`c_2 = (x_{min}, y_{max}, z_{min})`
        |br| :math:`c_3 = (x_{min}, y_{max}, z_{max})`
        |br| :math:`c_4 = (x_{max}, y_{min}, z_{min})`
        |br| :math:`c_5 = (x_{max}, y_{min}, z_{max})`
        |br| :math:`c_6 = (x_{max}, y_{max}, z_{min})`
        |br| :math:`c_7 = (x_{max}, y_{max}, z_{max})`

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> bounds_utils.get_obb_corners(centroid, axes, half_extent)
        [[-0.5 -0.5 -0.5]
         [-0.5 -0.5  0.5]
         [-0.5  0.5 -0.5]
         [-0.5  0.5  0.5]
         [ 0.5 -0.5 -0.5]
         [ 0.5 -0.5  0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.5  0.5]]
    """
    corners = [
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
    ]
    return np.array(corners)

class Replicator:

    scene_prim_path = "/World/scene" #!!

    @staticmethod
    def factory(world, config): 
        if config['replicator'] == "std_obj":
            from .std_object import STDObjectReplicator
            return STDObjectReplicator(world, config)
        elif config['replicator'] == "glass":
            from .glass import GlassReplicator
            return GlassReplicator(world, config)
        elif config['replicator'] == "graspnet":
            from .graspnet import GraspNetReplicator
            return GraspNetReplicator(world, config)
        else:
            raise Exception("Unknown replicator: {}".format(config['replicator']))

    def __init__(self, world, config) -> None:
        self._world = world
        self._config = config
        self._log = print

        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["writer_config"]["output_dir"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.next_seq_id = self._config["writer_config"]["start_sequence_id"]
        if self.next_seq_id == -1:
            self.next_seq_id = find_next_sequence_id(self.output_dir)
    
        self._log(f"next sequence id {self.next_seq_id}")
        self._log(f"num frames to render {self._config['num_frames_per_surface']}")

        """ show transparent object in RTX raytracing rendering mode
            https://forums.developer.nvidia.com/t/replicator-composer-with-transparent-objects/262774
            # rep.settings.set_render_rtx_realtime()
        """
        carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)

        self._env_light = None
        self._step_tick = 0
        self._writer_tick = None
    
    @abstractmethod
    def setup_domain_randomization()->Dict:
        pass
    
    @abstractmethod
    def render() -> None:
        pass
    
    def calc_mesh_center(self, glass_prim):
        surface_tf = omni.usd.get_world_transform_matrix(glass_prim)
        bb_cache = create_bbox_cache()
        centroid, axes, half_extent = compute_obb(bb_cache, glass_prim.GetPrimPath())
        larger_xy_extent = (half_extent[0], half_extent[1], half_extent[2])
        obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
        return np.mean(obb_corners, axis=0)

    def calc_surface_center(self, surface_prim):
        surface_tf = omni.usd.get_world_transform_matrix(surface_prim)
        # surface_pos = surface_tf.ExtractTranslation()
        
        bb_cache = create_bbox_cache()
        centroid, axes, half_extent = compute_obb(bb_cache, surface_prim.GetPrimPath())
        larger_xy_extent = (half_extent[0], half_extent[1], half_extent[2])
        obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
        # TODO test
        top_corners = [
            # obb_corners[0].tolist(),
            # obb_corners[1].tolist(),
            obb_corners[2].tolist(),
            obb_corners[3].tolist(),
            # obb_corners[4].tolist(),
            # obb_corners[5].tolist(),
            obb_corners[6].tolist(),
            obb_corners[7].tolist(),
        ]

        position = np.mean(top_corners, axis=0)
    
        self._surface_obb = {
            "centroid": centroid,
            "axes": axes,
            "half_extent": half_extent,
            "position": position,
            "orientation": t3d.quaternions.mat2quat(axes),
        }
        return position
    
    def enable_physics(self, prim):
        # Enable physics (gravity), maybe not?
        # rigid_prim = RigidPrim(prim_path=str(prim.GetPrimPath()), name=rep_meta['category'])
        # rigid_prim.enable_rigid_body_physics()

        # Enable collision (rigid body)
        collisonAPI = UsdPhysics.CollisionAPI.Apply(prim)

    def create_omnipbr_material(self, mtl_url, mtl_name, mtl_path):
        stage = omni.usd.get_context().get_stage()
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_path)
        material_prim = stage.GetPrimAtPath(mtl_path)
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, get_prim=True))

        # Add value inputs
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float)

        # Add texture inputs
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("metallic_texture", Sdf.ValueTypeNames.Asset)

        # Add other attributes
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)

        # Add texture scale and rotate
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
        shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)

        # transparency
        shader.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float)
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool)
        shader.CreateInput("cutout_opacity", Sdf.ValueTypeNames.Float)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool)
        shader.CreateInput('glass_ior', Sdf.ValueTypeNames.Float)
        shader.CreateInput('frosting_roughness', Sdf.ValueTypeNames.Float)
        shader.CreateInput('reflection_color', Sdf.ValueTypeNames.Color3f)

        shader.GetInput("opacity_threshold").Set(0.0)
        shader.GetInput("enable_opacity").Set(True)
        
        cutout_opacity = self.dr["std"]["transparent"]["cutout_opacity"]
        shader.GetInput("cutout_opacity").Set(cutout_opacity)
        thin_walled = self.dr["std"]["transparent"]["thin_walled"]
        shader.GetInput("thin_walled").Set(thin_walled)

        glass_ior = self.dr["std"]["transparent"]["glass_ior"]
        shader.GetInput("glass_ior").Set(glass_ior)

        frosting_roughness = self.dr["std"]["transparent"]["frosting_roughness"]
        shader.GetInput("frosting_roughness").Set(frosting_roughness)

        reflection_color = self.dr["std"]["specular"]["reflection_color"]
        shader.GetInput("reflection_color").Set((reflection_color, reflection_color, reflection_color))

        material = UsdShade.Material(material_prim)
        return material

    def rep_randomize_camera(self, surface_config, surface_center, cam_p_list, cam_q_list):
        # surface_center = self.calc_surface_center(surface_prim)
        rgb_pos_list = []
        left_ir_pos_list = []
        right_ir_pos_list = []
        cam_euler_list = []
        projector_pos_list = []
        projector_euler_list = []
        
        placement = self._config["depth_sensor"]["placement"]
        # through observation from isaac sim
        Rpc = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
        assert np.allclose(t3d.euler.mat2euler(Rpc, 'sxyz'), (np.pi/2, 0, np.pi/2)) # from GUI 

        for p, q in zip(cam_p_list, cam_q_list):
            Rwc = t3d.quaternions.quat2mat(q)
            Rwp = Rwc @ Rpc.T
            eulers = t3d.euler.mat2euler(Rwp, 'sxyz') # extrinsic
            proj_eulers = t3d.euler.mat2euler(Rwc, 'sxyz') # extrinsic

            rgb_pos = p + surface_center
            ir_left_pos = rgb_pos + Rwc[:3, 0] * placement["rgb_to_left_ir"]
            ir_right_pos = rgb_pos + Rwc[:3, 0] * placement["rgb_to_right_ir"]
            projector_pos = rgb_pos + Rwc[:3, 0] * placement["rgb_to_projector"]

            for _ in range(2): # repeat for toggle
                rgb_pos_list.append(rgb_pos.astype(np.float32).tolist())
                left_ir_pos_list.append(ir_left_pos.astype(np.float32).tolist())
                right_ir_pos_list.append(ir_right_pos.astype(np.float32).tolist())
                cam_euler_list.append(np.rad2deg(eulers).astype(np.float32).tolist())
                projector_pos_list.append(projector_pos.astype(np.float32).tolist())
                projector_euler_list.append(np.rad2deg(proj_eulers).astype(np.float32).tolist())

        projctor_intensity = self._config["depth_sensor"]["projector"]["intensity"]
        projector_intensity_list = [0, projctor_intensity] * (len(rgb_pos_list)//2)
        intensity_on_off = [self.dr["lighting"]["intensity"][0],
                            self.dr["lighting"]["intensity"][1]]
        light_intensity_list = intensity_on_off * (len(rgb_pos_list)//2)

        env_light_intensity_list = [self._config["lighting"]["Distant_light"]["intensity"], 
                                    10] * (len(rgb_pos_list)//2)

        path_pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pattern.png")
        texture_list = [f"file://{path_pattern}"] *  len(rgb_pos_list)
        isProjector_list = [True, True] * (len(rgb_pos_list)//2)

        with rep.trigger.on_frame(): # interval=2, rgb + ir
            with self.projector:
                rep.modify.attribute("intensity", rep.distribution.sequence(projector_intensity_list))
                rep.modify.attribute("texture:file", rep.distribution.sequence(texture_list))
                rep.modify.attribute("isProjector", rep.distribution.sequence(isProjector_list))
                rep.modify.pose(
                    position=rep.distribution.sequence(projector_pos_list),
                    rotation=rep.distribution.sequence(projector_euler_list)
                )

            with self._light:
                rep.modify.attribute("intensity", rep.distribution.sequence(light_intensity_list))

            with self.cam_rgb:
                rep.modify.pose(
                    position=rep.distribution.sequence(rgb_pos_list),
                    rotation=rep.distribution.sequence(cam_euler_list)
                )
            with self.cam_ir_left:
                rep.modify.pose(
                    position=rep.distribution.sequence(left_ir_pos_list),
                    rotation=rep.distribution.sequence(cam_euler_list)
                )
            with self.cam_ir_right:
                rep.modify.pose(
                    position=rep.distribution.sequence(right_ir_pos_list),
                    rotation=rep.distribution.sequence(cam_euler_list)
                )
            if self._env_light is not None:
                with self._env_light:
                    rep.modify.attribute("intensity", rep.distribution.sequence(env_light_intensity_list))

    def setup_depth_sensor(self, suffix=""):
        clipping_range = self._config["depth_sensor"]["clipping_range"]
        clipping_range = (clipping_range[0], clipping_range[1]) # datatype conversion
        W, H = self._config["depth_sensor"]["resolution"]
        FOV = np.deg2rad(self._config["depth_sensor"]["fov"])
        # @see exts/omni.isaac.sensor/omni/isaac/sensor/scripts/camera.py
        f = self._config["depth_sensor"]["focal_length"] * 10 
        h = 2 * f * math.tan(FOV/2) 
        ir_cam_cfgs = {
            "focal_length": f,
            "clipping_range": clipping_range,
            "horizontal_aperture": h
        }
        # https://www.intel.com/content/dam/support/us/en/documents/
        # emerging-technologies/intel-realsense-technology/Intel-RealSense-D400-Series-Datasheet.pdf
        self.cam_rgb = rep.create.camera(name=f"CameraRGB{suffix}", **ir_cam_cfgs)
        self.cam_ir_left = rep.create.camera(name=f"Camera01{suffix}", **ir_cam_cfgs)
        self.cam_ir_right = rep.create.camera(name=f"Camera02{suffix}", **ir_cam_cfgs)

        # TODO self.third_person_cam = rep.create.camera(name=f"ThirdPersonCamera{suffix}")

        # hack vertical aperture
        v = h * H / W
        prim_cam_rgb = self._world.stage.GetPrimAtPath(f"/Replicator/CameraRGB{suffix}_Xform/CameraRGB{suffix}")
        prim_cam_rgb.GetAttribute("verticalAperture").Set(v)
        prim_cam_ir_left = self._world.stage.GetPrimAtPath(f"/Replicator/Camera01{suffix}_Xform/Camera01{suffix}")
        prim_cam_ir_left.GetAttribute("verticalAperture").Set(v)
        prim_cam_ir_right = self._world.stage.GetPrimAtPath(f"/Replicator/Camera02{suffix}_Xform/Camera02{suffix}")
        prim_cam_ir_right.GetAttribute("verticalAperture").Set(v)
        
        # ir pattern projector
        path_pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pattern.png")
        exposure = self._config["depth_sensor"]["projector"]["exposure"]
        self.projector = rep.create.light(
            light_type = "Rect",
            intensity = 0,
            exposure = exposure,
            color = (1, 0, 0),
            position = (0.0, 0.0, 0.0),
            scale = (1, 1, 1),
            count = 1,
            texture = f"file://{path_pattern}",
            name="RectLight"
        )
        prim_path = f"/Replicator/RectLight{suffix}_Xform/RectLight{suffix}"
        rect_light = self._world.stage.GetPrimAtPath(prim_path)
        # set scale
        if not rect_light.GetAttribute("xformOp:translate"):
            UsdGeom.Xformable(rect_light).AddTranslateOp()
        if not rect_light.GetAttribute("xformOp:scale"):
            UsdGeom.Xformable(rect_light).AddScaleOp()
        rect_light.GetAttribute("xformOp:scale").Set((1.532075471*1.1, 1.1*1.1, 1)) # 812/583 * 1.1 * 1.1
        rect_light.GetAttribute("xformOp:translate").Set((0, 0, 0))

        is_projector_attr = rect_light.GetAttribute("isProjector")
        if not is_projector_attr:
            self._log("is_projector_attr is None, creating a new one..")
            omni.kit.commands.execute(
                "CreateUsdAttribute",
                prim=rect_light,
                attr_name="isProjector",
                attr_type=Sdf.ValueTypeNames.Bool,
                attr_value=True
            )
        omni.kit.commands.execute("ChangeProperty",
            prop_path=Sdf.Path(prim_path + ".isProjector"),
            value=True,
            prev=False)

    def ticker(self):
        return self._writer_tick, self._step_tick
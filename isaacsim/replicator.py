
import os, sys
import csv, copy, math
import time, json
import numpy as np
import random
import transforms3d as t3d
# from scipy.spatial.transform import Rotation
from typing import Union, Type, List
from functools import partial
from PIL import Image

import carb
import omni.replicator.core as rep
import omni.usd
from omni.isaac.kit import SimulationApp

from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.replicator.core import Writer, AnnotatorRegistry
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.objects import DynamicCuboid
from pxr import Gf, Sdf, Usd, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

# import offline_generation_utils
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from custom_writer import ColorWriter, GtWriter, IRWriter
from omni.replicator.core import WriterRegistry
from replicate import Replicator

scene_prim_path = "/World/scene" #!!

class IRReplicator:
    def __init__(self, app: SimulationApp, world: World, config:DictConfig) -> None:
        self._app = app
        self._world = world
        self._config = config
        self._log = self._app.app.print_and_log

        # Get server path
        # self.assets_root_path = get_assets_root_path()
        # if self.assets_root_path is None:
        #     carb.log_error("Could not get nucleus server path, closing application..")
        #     app.close()

        # load different scene replicator according to configuration
        self.replicator = Replicator.factory(world, config)

        # self._light: Usd.Prim = self.setup_lighting()

        self._scene: Usd.Prim = self.load_scene()
        # self._world.scene.add_default_ground_plane()
        """ self.scene = UsdPhysics.Scene.Define(self._world.stage, Sdf.Path("/physicsScene"))
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(9.81)
        omni.kit.commands.execute(
            "AddGroundPlaneCommand",
            stage=self._world.stage,
            planePath="/groundPlane",
            axis="Z",
            size=10.000,
            position=Gf.Vec3f(0, 0, -0.01), # hack to hide ground mesh
            color=Gf.Vec3f(0.5),
        ) """

        # self._mats = self.load_materials()
        
        # Disable capture on play and async rendering
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        carb.settings.get_settings().set("/app/asyncRendering", False)

        # https://forums.developer.nvidia.com/t/replicator-images-contain-artifacts-from-other-frames/220837
        # carb.settings.get_settings().set("/rtx/ambientOcclusion/enabled", False)
        # rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

        # start replicator
        if self._config["rt_subframes"] > 1:
            rep.settings.carb_settings("/omni/replicator/RTSubframes", self._config["rt_subframes"])
        else:
            carb.log_warn("RTSubframes is set to 1, consider increasing it if materials are not loaded on time")

        self.clear_previous_semantics()

        self.output_dir = os.path.join(os.path.dirname(__file__), config["writer_config"]["output_dir"])
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.replicator.setup_depth_sensor()

        WriterRegistry.register(ColorWriter)
        WriterRegistry.register(GtWriter)
        WriterRegistry.register(IRWriter)

        self.dr = self.replicator.setup_domain_randomization()
        self._log(json.dumps(self.dr))

    def clear_previous_semantics(self):
        return
        if self._config["clear_previous_semantics"]:
            offline_generation_utils.remove_previous_semantics(self._world.stage)
    
    
    def setup_lighting(self):
        # prim_path = "/World/DiskLight"
        # diskLight = UsdLux.DiskLight.Define(self._world.stage, Sdf.Path(prim_path))
        # diskLight.CreateIntensityAttr(15000)
        
        # light = self._world.stage.GetPrimAtPath(prim_path)
        # if not light.GetAttribute("xformOp:translate"):
        #     UsdGeom.Xformable(light).AddTranslateOp()
        # return light
        pass

    # def setup_projector_lighting(self):
    #     prim_path = "/World/RectLight"
    #     rectLight = UsdLux.RectLight.Define(self._world.stage, Sdf.Path(prim_path))
    #     rectLight.CreateIntensityAttr(500)
    #     rectLight.Create
    
    def load_scene(self):
        scene_name = self._config["hssd"]["name"]
        data_dir = os.path.abspath(self._config.hssd["data_dir"])
        env_url = f"{data_dir}/{scene_name}/{scene_name}.usd"
        assert os.path.exists(env_url), f"Scene file {env_url} does not exist"
        add_reference_to_stage(usd_path=env_url, prim_path=scene_prim_path) 

        hssd_env = self._world.stage.GetPrimAtPath(scene_prim_path)
        if not hssd_env.GetAttribute("xformOp:translate"):
            UsdGeom.Xformable(hssd_env).AddTranslateOp()
        if not hssd_env.GetAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(hssd_env).AddRotateXYZOp()
        if not hssd_env.GetAttribute("xformOp:scale"):
            UsdGeom.Xformable(hssd_env).AddScaleOp() 

        hssd_env.GetAttribute("xformOp:rotateXYZ").Set((90, 0, 0))
        scale = self._config["hssd"]["scale"]
        hssd_env.GetAttribute("xformOp:scale").Set((scale, scale, scale))

        if self._config["hssd"]["hide_ceilings"]:
            ceiling = hssd_env.GetPrimAtPath(f"{scene_prim_path}/ceilings")
            ceiling.GetAttribute("visibility").Set("invisible")

        if self._config["hssd"]["hide_walls"]: # an ugly hack
            walls = hssd_env.GetPrimAtPath(f"{scene_prim_path}/walls")
            walls.GetAttribute("visibility").Set("invisible")

        return hssd_env
    
    # deprecated
    def load_materials(self): 
        #https://forums.developer.nvidia.com/t/how-can-i-change-material-of-the-existing-object-in-runtime/161253
        # path_mat_glass_clear = assets_root_path + "/NVIDIA/Materials/vMaterials_2/Glass/Glass_Clear.mdl"
        path_mat_glass_clear = "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Glass/Glass_Clear.mdl"
        # load more
        success, result = omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
            mtl_url=path_mat_glass_clear, # This can be path to local or remote MDL
            mtl_name='Glass_Clear', # sourceAsset:subIdentifier (i.e. the name of the material within the MDL)
            mtl_path="/World/Looks/Glass_Clear" # Prim path for the Material to create.
        )
        t = UsdShade.Material(self._world.stage.GetPrimAtPath("/World/Looks/Glass_Clear"))

        path_mat_metal_aluminum = "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Metal/Aluminum.mdl"
        success, result = omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
            mtl_url=path_mat_glass_clear, # This can be path to local or remote MDL
            mtl_name='Aluminum',
            mtl_path="/World/Looks/Aluminum" # Prim path for the Material to create.
        )
        s = UsdShade.Material(self._world.stage.GetPrimAtPath("/World/Looks/Aluminum"))

        return {
            'transparent': [t], # TODO add more
            'specular': [s] # TODO add more
        }
    
    # deprecated
    def create_rep_object(self, surface_center_pos):
        test_model = rep.create.from_usd(f"file:///home/songlin/Projects/DREDS/DepthSensorSimulator/cad_model/02691156/1c93b0eb9c313f5d9a6e43b878d5b335_converted/model_obj.usd", 
            semantics=[("class", "test")])
            
        test_ball = rep.create.sphere(name="test_ball", position=surface_center_pos, scale=(0.1, 0.1, 0.1))
        with test_model:
            rep.physics.collider()
            rep.physics.rigid_body(
                # velocity=rep.distribution.uniform((-0,0,-0),(0,0,1)),
                # angular_velocity=rep.distribution.uniform((-0,0,-100),(0,0,0))
            )

    

    def start(self):
        # self.debug = 0
        # Find the desired surface
        # for surface_config in self._config["hssd"]['surfaces']:
            # surface = self._config["hssd"]['surface']
            self.replicator.render()
    
    """ def randomize_texture(self, dred_models):
        materials = create_materials(self._world.stage, len(dred_models))
        assets_root_path = get_assets_root_path()
        textures = [
            assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
            assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
            assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
            assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
        ]

        delay=0.2
        initial_materials = {} 
        for i, shape in dred_models.items(): #enumerate():
            cur_mat, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial()
            initial_materials[shape] = cur_mat
            UsdShade.MaterialBindingAPI(shape).Bind(materials[i-1], UsdShade.Tokens.strongerThanDescendants)

        for mat in materials:
            shader = UsdShade.Shader(omni.usd.get_shader_from_material(mat, get_prim=True))
            # diffuse_texture = np.random.choice(textures)
            # shader.GetInput("diffuse_texture").Set(diffuse_texture)

            # project_uvw = np.random.choice([True, False], p=[0.9, 0.1])
            # shader.GetInput("project_uvw").Set(bool(project_uvw))

            # texture_scale = np.random.uniform(0.1, 1)
            # shader.GetInput("texture_scale").Set((texture_scale, texture_scale))

            # texture_rotate = np.random.uniform(0, 45)
            # shader.GetInput("texture_rotate").Set(texture_rotate)

            shader.GetInput("metallic_constant").Set(1.0)
            shader.GetInput("reflection_roughness_constant").Set(0.0) """

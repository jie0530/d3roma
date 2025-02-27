import os, random, time, json, math, copy
import numpy as np

import omni
import omni.replicator.core as rep
from omni.isaac.core.utils import prims
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage, open_stage, create_new_stage
from pxr import Gf, Sdf, Usd, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

from replicate.scene_replicator import Replicator
from dreds_renderer import DredsRenderer, generate_material_type, g_synset_name_scale_pairs
from utils_func import get_all_child_mesh, get_visibility_attribute

scene_prim_path = "/World/scene" #!!

class STDObjectReplicator(Replicator):

    def __init__(self, world, config) -> None:
        super().__init__(world, config)

        self.dr = {}

    def setup_domain_randomization(self):
        self.domain_randomization = self._config["domain_randomization"]
        assert self.domain_randomization, "not implemented yet!"

        # domain randomization of lighting
        light_type_dr = self._config["lighting"]["light_type"]
        self.light_type = light_type_dr[random.randint(0, len(light_type_dr))-1]
        light_conf_dr = self._config["lighting"][f"{self.light_type}_light"]

        self.dr['lighting'] = {}
        self.dr['lighting']['type'] = self.light_type
        
        light_conf = {
            'radius': random.uniform(*light_conf_dr['radius']),
            'height': random.uniform(*light_conf_dr['height']),
            'intensity': [
                random.uniform(*light_conf_dr['intensity']['on']),
                random.uniform(*light_conf_dr['intensity']['off'])
            ]
        }
        # self.dr['lighting'][f'{self.light_type}_light'] = light_conf
        self.dr['lighting'].update(light_conf)

        # scene disk light
        self._light = rep.create.light(
            light_type = self.light_type, #"Sphere", #"Disk",
            intensity = self.dr['lighting']["intensity"][0],
            color = (1.0, 1.0, 1.0),
            position = (0.0, 0.0, 0.0),
            name= f"{self.light_type}Light"
        )

        # prim_path_disk = "/Replicator/DiskLight_Xform/DiskLight"
        # rect_light = self._world.stage.GetPrimAtPath(prim_path_disk)
        # rect_light.GetAttribute("inputs:radius").Set(self._config["lighting"]["disk_light"]["radius"])

        prim_path_light = f"/Replicator/{self.light_type}Light_Xform/{self.light_type}Light"
        prim_light = self._world.stage.GetPrimAtPath(prim_path_light)
        prim_light.GetAttribute("inputs:radius").Set(self.dr["lighting"]["radius"])

        if self.dr["lighting"]["type"] == "Sphere":
            prim_light.GetAttribute("treatAsPoint").Set(True)

        # domain randomization of materials
        self.dr["std"] = {}
        transparent_dr = self._config["transparent"]
        transparent_conf = {
            "roughness_constant": random.uniform(*transparent_dr["roughness_constant"]),
            "cutout_opacity": random.uniform(*transparent_dr["cutout_opacity"]),
            "thin_walled": transparent_dr["thin_walled"],
            "glass_ior": random.uniform(*transparent_dr["glass_ior"]),
            "frosting_roughness": random.uniform(*transparent_dr["frosting_roughness"])
        }
        self.dr["std"]["transparent"] = transparent_conf

        specular_dr = self._config["specular"]
        specular_conf = {
            "reflection_roughness_constant": random.uniform(*specular_dr["reflection_roughness_constant"]),
            "metallic_constant": random.uniform(*specular_dr["metallic_constant"]),
            "reflection_color": random.uniform(*specular_dr["reflection_color"]),
        }
        self.dr["std"]["specular"] = specular_conf
        return self.dr
    
    def render(self) -> None:
        self._log("start std_obj render on surface")

        surface_config = self._config["hssd"]['surface']
        origin_prim_path = surface_config['prim_path']
        prim_path = origin_prim_path.replace("/World", scene_prim_path)
        surface_prim = self._world.stage.GetPrimAtPath(prim_path)
        self.enable_physics(surface_prim)

        surface_center_pos = self.calc_surface_center(surface_prim)
        # move disk light 1m above the surface center
        # self._light.GetAttribute("xformOp:translate").Set((surface_center_pos[0], surface_center_pos[1], surface_center_pos[2] + 1.0))
        with self._light:
            rep.modify.pose(position=(surface_center_pos[0], 
                                        surface_center_pos[1], 
                                        surface_center_pos[2] + self.dr["lighting"]["height"]))
            
        # domain randomization
        root_dir = os.path.abspath(self._config.dreds.cad_model_dir)
        renderer = DredsRenderer(root_dir)
        select_model_list, cam_q_list, cam_p_list = renderer.domain_randomize(self._config["num_frames_per_surface"])
        surface_center_pos = self.calc_surface_center(surface_prim)
        
        # load object
        all_rigid_objects = []
        # last_object_name = None
        # model_prims = {}
        # material_prims = []
        initial_materials = {}
        for model in select_model_list:
            prim_name = f"model_{model['instance_id']}_{model['class_name']}"
            self._log(f"{model['material_type']}, {model['class_name']}, {model['instance_path']}")

            model_prim = prims.create_prim(
                prim_path=f"/World/{model['class_name']}_{model['instance_id']}",
                usd_path=f"file://{model['instance_path']}",
                semantic_label=prim_name,
                scale=[model['scale']]*3
            )
            # Wrap the prim into a rigid prim to be able to simulate it
            box_rigid_prim = RigidPrim(
                prim_path=str(model_prim.GetPrimPath()),
                name=model['instance_name'],
                position=surface_center_pos + Gf.Vec3d(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),  model['instance_id'] * 0.05),
                orientation=euler_angles_to_quat([random.uniform(0, math.pi/2), random.uniform(0, math.pi/2), random.uniform(0, math.pi)]),
            )
            # set object as rigid body
            box_rigid_prim.enable_rigid_body_physics()
            # Enable collision
            UsdPhysics.CollisionAPI.Apply(model_prim)
            # Register rigid prim with the scene
            self._world.scene.add(box_rigid_prim)
            # last_object_name = model['instance_name']
            all_rigid_objects.append(model['instance_name'])
            # model_prims[model['instance_id']] = model_prim

            # disable opacity for ground truth depth rendering, tested in PathRendering mode.
            for prim in get_all_child_mesh(model_prim):
                cur_mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                shader = UsdShade.Shader(omni.usd.get_shader_from_material(cur_mat, get_prim=True))
                
                shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool)
                shader.GetInput("enable_opacity").Set(False)

            # change material
            
            if model["material_type"] == "transparent" or model['class_name'] in ["cup", "bottle"]: # hack transparent cup and bottle
                mat_type = model["material_type"]
                MDL = "OmniGlass.mdl"
                mtl_name, _ = os.path.splitext(MDL)
                MAT_PATH = "/World/Looks"
                
                prim_path = omni.usd.get_stage_next_free_path(self._world.stage, f"{MAT_PATH}/{mtl_name}", False)
                mat = self.create_omnipbr_material(mtl_url=MDL, mtl_name=mtl_name, mtl_path=prim_path)
                
                initial_materials[model_prim] = mat
                # material_prims.append(prim_path)

            elif model["material_type"] == "specular":
                mat_type = model["material_type"]
                for prim in get_all_child_mesh(model_prim):
                    
                    if len(prim.GetChildren()) >1 :
                        # hot fix: multi-materials
                        self._log(f"multi-materials: {prim.GetPath()}")
                        for subp in prim.GetChildren():
                            mat, _ = UsdShade.MaterialBindingAPI(subp).ComputeBoundMaterial()
                            shader = UsdShade.Shader(omni.usd.get_shader_from_material(mat, get_prim=False))

                            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
                            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)

                            shader.GetInput("metallic").Set(self.dr["std"]["specular"]["metallic_constant"])
                            shader.GetInput("roughness").Set(self.dr["std"]["specular"]["reflection_roughness_constant"])
                        continue

                    cur_mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    shader = UsdShade.Shader(omni.usd.get_shader_from_material(cur_mat, get_prim=True))

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
                    
                    shader.GetInput("metallic_constant").Set(self.dr["std"]["specular"]["metallic_constant"])
                    shader.GetInput("reflection_roughness_constant").Set(self.dr["std"]["specular"]["reflection_roughness_constant"])

                    UsdShade.MaterialBindingAPI(prim).Bind(cur_mat, UsdShade.Tokens.strongerThanDescendants)

            elif model["material_type"] == "diffuse":
                mat_type = model["material_type"]
                pass

        # randomize camera
        surface_center = self.calc_surface_center(surface_prim)
        self.rep_randomize_camera(None, surface_center, cam_p_list, cam_q_list)

        # output_dir = f"{self.output_dir}/{self._config["hssd"]['name']}/{surface_config['category']}"
        # os.makedirs(output_dir, exist_ok=True)
        # self.writer._output_dir = output_dir
        # output_dir = self._config.writer_config.output_dir
        with open(f"{self.output_dir}/meta_{self.next_seq_id}.json", 'w') as f:
            meta = {
                "models": select_model_list,
                "domain_randomization":  self.dr
            }
            f.write(json.dumps(meta, indent=4, sort_keys=True))

        # replicate texture
        # self.randomize_texture(model_prims)
        # Setup the writer

        cfg = copy.deepcopy(self._config["writer_config"])
        cfg["output_dir"] = self.output_dir
        cfg["start_sequence_id"] = self.next_seq_id

        _config = copy.copy(self._config)
        _config["writer_config"]["output_dir"] = self.output_dir
        _config["writer_config"]["start_sequence_id"] = self.next_seq_id
        
        # self._config["writer_config"]["output_dir"]

        resolution = np.array(self._config["depth_sensor"]["resolution"]).astype(np.uint32).tolist()
        dep_res = (resolution[0], resolution[1])
        self.writer_gt = rep.WriterRegistry.get("GtWriter")
        self.writer_gt.initialize(ticker=self.ticker, depth_sensor_cfg=_config["depth_sensor"], **_config["writer_config"])
        cam_gt_rp = rep.create.render_product(self.cam_rgb, dep_res, name="CameraDepth")
        self.writer_gt.attach([cam_gt_rp])

        # start simulation
        self._world.reset()

        if self._config["render_after_quiet"]:
            # wait for objects to fall
            # last_box = self._world.scene.get_object(last_object_name)
            max_tried = 0
            while True and max_tried < 10:
                max_sim_steps = 250
                for i in range(max_sim_steps):
                    self._world.step(render=False)
                quited = True
                for rigid_object in all_rigid_objects:
                    obj = self._world.scene.get_object(rigid_object)
                    if obj is None:
                        self._log(f"{rigid_object} is not found!")
                        continue
                    if np.linalg.norm(obj.get_linear_velocity()) > 0.001:
                        quited = False
                        break
                if quited:
                    self._log("all objects quited")
                    break # stop physics simulation, start rendering
                max_tried += 1
                self._log("still waiting for objects to fall")
        
        rep.settings.set_render_rtx_realtime()
        start_time = time.time()
        # rep.orchestrator.run_until_complete(num_frames=2*self._config['num_frames_per_surface'])
        for _ in range(2*self._config['num_frames_per_surface']):
            self._writer_tick = "gt"
            if _ % 2 == 0:
                self._step_tick += 1
            rep.orchestrator.step(rt_subframes=self._config['rt_subframes'], pause_timeline=True)
        
        end_time = time.time()

        # log running time
        runtime = end_time - start_time
        fps = runtime / self._config['num_frames_per_surface']
        self._log(f"Replicator finished in {round(runtime, 2)} seconds, FPS={round(fps, 2)}")

        # change materials
        for model_prim_, mat_ in initial_materials.items():
            UsdShade.MaterialBindingAPI(model_prim_).Bind(mat_, UsdShade.Tokens.strongerThanDescendants)

        self.writer_gt.detach()

        self.writer_rgb = rep.WriterRegistry.get("ColorWriter")
        self.writer_rgb.initialize(ticker=self.ticker, **cfg)
        cam_rgb_rp = rep.create.render_product(self.cam_rgb, resolution, name="CameraRGB")
        self.writer_rgb.attach([cam_rgb_rp])

        self.writer_ir = rep.WriterRegistry.get("IRWriter")
        self.writer_ir.initialize(output_dir = self.output_dir, start_sequence_id = self.next_seq_id, ticker=self.ticker)
        cam_left_ir_rp = rep.create.render_product(self.cam_ir_left, resolution, name="Camera01")
        cam_right_ir_rp = rep.create.render_product(self.cam_ir_right, resolution, name="Camera02")
        self.writer_ir.attach([cam_left_ir_rp, cam_right_ir_rp])

        if self._config["launch_config"]["renderer"] == "PathTracing": # hack
            rep.settings.set_render_pathtraced()
        start_time = time.time()
        # rep.orchestrator.run_until_complete(num_frames=2*self._config['num_frames_per_surface'])
        for _ in range(2*self._config['num_frames_per_surface']):
            if _ % 2 == 0:
                self._writer_tick = "rgb"
            else:
                self._writer_tick = "ir"
                self._step_tick += 1
            rep.orchestrator.step(rt_subframes=self._config['rt_subframes'], pause_timeline=True)
        end_time = time.time()
        runtime = end_time - start_time
        fps = runtime / self._config['num_frames_per_surface']
        self._log(f"Replicator finished in {round(runtime, 2)} seconds, FPS={round(fps, 2)}")
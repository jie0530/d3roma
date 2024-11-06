from typing import List 
from .mono_datasets import *
from .stereo_datasets import *
from config import TrainingConfig
from omegaconf import OmegaConf
from torch.utils.data.dataset import ConcatDataset
from utils.camera import Realsense, RGBDCamera

def create_dataset(config: TrainingConfig, dataset_name, split = "train"):
    mono_lst = ['NYUv2', 'ScanNet', 'HyperSim', 'SceneNet', 'ScanNetpp', 'VK2', 'KITTI', "Middlebury", "InStereo2K", "Tartenair", "HRWSI", "SynTODD"]
    stereo_lst = ["Dreds",  "Middlebury", "SceneFlow", "Real", "HssdIsaacStd", "ClearPose", "SynTODDRgbd", "Gapartnet2"]
    image_size = tuple(config.image_size)

    if len(dataset_name.split("_")) > 1: # Real_split_device
        dataset_name, split, device = dataset_name.split("_")

    from utils.utils import Normalizer
    normalizer = Normalizer.from_config(config)
    
    if dataset_name in stereo_lst:
        cam_res = [int(x) for x in config.camera_resolution.split("x")[::-1]] 
        
        if split == "train":
            # dataset = eval(dataset_name)(f"datasets/{dataset_name}", split="train", image_size=config.image_size, augment=config.augment, camera = config.camera)
            aug_params = {"crop_size": image_size, 
                          "min_scale": config.augment["min_scale"], 
                          "max_scale": config.augment["max_scale"],
                          "yjitter": config.augment["yjitter"]}
            aug_params["saturation_range"] = tuple(config.augment["saturation_range"])
            aug_params["gamma"] = config.augment["gamma"]
            aug_params["do_flip"] = config.augment["hflip"] #config.augment["hflip"]["prob"] > 0.0
            # aug_params["camera_resolution"] = cam_res
            if dataset_name == 'SceneFlow': # BUG? min disp=0.5, max disp=192.0?
                disp_reader = partial(frame_utils.read_sceneflow, cam_res)
                clean_dataset = SceneFlow(aug_params=aug_params, root="datasets/sceneflow", dstype='frames_cleanpass', 
                                                reader=disp_reader, normalizer=normalizer)
                final_dataset = SceneFlow(aug_params=aug_params, root="datasets/sceneflow", dstype='frames_finalpass', 
                                                reader=disp_reader, normalizer=normalizer)
                dataset = clean_dataset + final_dataset
            elif dataset_name == 'HssdIsaacStd':
                sim_camera = DepthCamera.from_device("sim") # BUG? max depth=5.
                # sim_camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
                sim_camera.change_resolution(config.camera_resolution)
                disp_reader = partial(frame_utils.readDispReal, sim_camera)
                dataset = HssdIsaacStd(sim_camera, normalizer, image_size, split, config.prediction_space, aug_params, reader=disp_reader)
            elif dataset_name == "Dreds":
                sim_camera = Realsense.default_sim() # BUG? max depth=2.
                # sim_camera.change_resolution(f"{image_size[1]}x{image_size[0]}")
                sim_camera.change_resolution(config.camera_resolution)
                # assert image_size == (126, 224)
                # disp_reader = partial(frame_utils.readDispDreds_exr, sim_camera)
                dataset = Dreds(sim_camera, normalizer, image_size, split, config.prediction_space, aug_params)
            elif dataset_name == "ClearPose":
                camera = RGBDCamera.default_clearpose() # BUG? max depth=5.
                camera.change_resolution(config.camera_resolution)
                disp_reader = partial(frame_utils.readDispReal, camera)
                dataset = ClearPose(camera, normalizer, image_size, split, config.prediction_space, reader=disp_reader)
            elif dataset_name == "SynTODDRgbd":
                camera = RGBDCamera.default_syntodd()
                camera.change_resolution(config.camera_resolution)
                disp_reader = partial(frame_utils.readDispReal, camera)
                dataset = SynTODDRgbd(config.dataset_variant, camera, normalizer, image_size, split, config.prediction_space, reader=disp_reader)
            elif dataset_name == "Gapartnet2":    
                sim_camera = Realsense.from_device("sim")
                sim_camera.change_resolution(config.camera_resolution)
                disp_reader = partial(frame_utils.readDispReal, sim_camera)
                dataset = Gapartnet2(sim_camera, normalizer, image_size, split, config.prediction_space, aug_params, reader=disp_reader)
            else:
                raise NotImplementedError
            
        else:
            if dataset_name == 'SceneFlow':
                disp_reader = partial(frame_utils.read_sceneflow, cam_res)
                dataset = SceneFlow(root="datasets/sceneflow", dstype='frames_cleanpass', things_test=True, 
                                            reader=disp_reader, normalizer=normalizer)
            elif dataset_name == "HssdIsaacStd":
                sim_camera = Realsense.from_device("sim")
                sim_camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
                disp_reader = partial(frame_utils.readDispReal, sim_camera)
                dataset = HssdIsaacStd(sim_camera, normalizer, image_size, split, space=config.prediction_space, reader=disp_reader)
            elif dataset_name == "Dreds": 
                sim_camera = Realsense.default_sim()
                sim_camera.change_resolution(f"{image_size[1]}x{image_size[0]}")
                # assert image_size == (126, 224) # reprod dreds-1.0
                # disp_reader = partial(frame_utils.readDispDreds_exr, sim_camera)
                dataset = Dreds(sim_camera, normalizer, image_size, split, space=config.prediction_space)
            elif dataset_name == "Real":
                real_cam = Realsense.default_real("fxm")
                real_cam.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
                dataset = Real(camera=real_cam, normalizer=normalizer,
                               image_size=image_size, scene=split, space=config.prediction_space)
            elif dataset_name == "ClearPose":
                camera = RGBDCamera.default_clearpose()
                camera.change_resolution(f"{image_size[1]}x{image_size[0]}")
                disp_reader = partial(frame_utils.readDispReal, camera)
                dataset = ClearPose(camera, normalizer, image_size, split, config.prediction_space, reader=disp_reader)
            elif dataset_name == "SynTODDRgbd":
                camera = RGBDCamera.default_syntodd()
                camera.change_resolution(f"{image_size[1]}x{image_size[0]}")
                disp_reader = partial(frame_utils.readDispReal, camera)
                dataset = SynTODDRgbd(config.dataset_variant, camera, normalizer, image_size, split, config.prediction_space, reader=disp_reader)
            elif dataset_name == "Gapartnet2":
                sim_camera = Realsense.from_device("sim")
                sim_camera.change_resolution(f"{config.image_size[1]}x{config.image_size[0]}")
                disp_reader = partial(frame_utils.readDispReal, sim_camera)
                dataset = Gapartnet2(sim_camera, normalizer, image_size, split, space=config.prediction_space, reader=disp_reader)

            else:
                raise NotImplementedError
            
    elif dataset_name in mono_lst:
        if split == "train":
            dataset= eval(dataset_name)(f"datasets/{dataset_name}", split="train", image_size=image_size, augment=config.augment)
        else:
            dataset = eval(dataset_name)(f"datasets/{dataset_name}", split=split, image_size=image_size, augment=None)
    else:
        raise NotImplementedError
    return dataset

def fetch_dataloader(config: TrainingConfig):
    """ Create the data loader for the corresponding trainign set """
        
    """ if not isinstance(config.dataset, List):
        dataset_lst = [config.dataset]
    else:
        dataset_lst = config.dataset
    
    if not isinstance(config.dataset_weight, List):
        weight_lst = [config.dataset_weight]
    else:
        weight_lst = config.dataset_weight """
        
    assert len(config.train_dataset) == len(config.dataset_weight)
    
    val_loader_lst = []
    train_dataset = None 
    for i, dataset_name in enumerate(config.train_dataset):
        new_dataset = create_dataset(config, dataset_name, split = "train")

        # multiple dataset weights
        if type(new_dataset) == ConcatDataset:
            # hack: unsupported operand type(s) for *: 'ConcatDataset' and 'int'
            for i in range(max(0, int(config.dataset_weight[i])-1)):
                new_dataset += new_dataset
        else:
            new_dataset = new_dataset * config.dataset_weight[i]
        
        # add train dataset together
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    
    for i, dataset_name in enumerate(config.eval_dataset):
        # saperately evaluate each dataset
        val_dataset = create_dataset(config, dataset_name, split = "val")
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config.eval_batch_size,
                                                shuffle=True,
                                                pin_memory=False, 
                                                drop_last=False)
        val_loader_lst.append(val_dataloader)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.train_batch_size, 
                                                shuffle=True,
                                                pin_memory=False,
                                                num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, 
                                                drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_dataloader, val_loader_lst


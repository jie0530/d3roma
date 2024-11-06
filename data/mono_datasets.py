import os
import numpy as np
import torch
import random 
from PIL import Image
import cv2 
from glob import glob
import os.path as osp
import h5py 
from .dataset import WarpDataset
from utils import frame_utils
import torchvision.transforms.functional as F

class MonoDataset(WarpDataset):
    def __init__(self, image_size, max_depth, augment):
        self.init_seed = False
        self.max_depth = max_depth
        self.is_test = False
        super().__init__(image_size, augment)
                
    def read_data(self, index):
        raise NotImplementedError
    
    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % self.__len__()
        rgb, depth, mask = self.read_data(index)

        # TODO filling the holes

        rgb = torch.tensor(rgb).to(torch.float32).permute(2,0,1)
        depth = torch.tensor(depth).to(torch.float32).unsqueeze(0)
        mask = torch.tensor(mask).to(torch.float32).unsqueeze(0)
        assert rgb.shape[1:] == depth.shape[1:]
        
        # data augmentation
        if not self.is_test:
            rgb, depth, mask = self.data_aug(rgb, depth, mask)

        # skip bad data
        masked_depth = depth[mask.bool()]
        if mask.int().max() == 0 or masked_depth.max() == masked_depth.min():
            print('skip bad data ', index, mask.int().max())            
            return self.__getitem__(index+1)
        
        # # visualization debug:
        # import matplotlib.pyplot as plt
        # cmap_gray = plt.get_cmap('gray')
        # depth[~mask.bool()] = 0
        # Image.fromarray((cmap_gray((mask[0].int()==1).int()) * 255).astype(np.uint8)).save(f"test_mask{index}.png")
        # Image.fromarray((cmap_gray(depth[0]/depth[0].max()) * 255).astype(np.uint8)).save(f"test_depth{index}.png")
        # Image.fromarray(rgb.permute(1,2,0).byte().cpu().numpy()).save(f"test_rgb{index}.png")
        # exit(1)
        
        # normalize disp and rgb to [-1, 1]
        if self.__class__.__name__ == "HyperSim" and not self.is_test:
            normalized_depth = self.normalize_depth(depth, mask, 0.02, 0.98)
            normalized_depth = normalized_depth.clamp(-1, 1) # more friendly for VAE if in range [-1,1]
        else:
            normalized_depth = self.normalize_depth(depth, mask)
            # normalized_depth = normalized_depth.clamp(-1, 1)

        # mask = mask.bool() & (normalized_depth >= 0) & (normalized_depth <= 1)
        # mask = mask.float()
            
        fxb = 24
        raw_disp = fxb / depth
        if self.__class__.__name__ == "HyperSim" and not self.is_test:
            normalized_disp = self.normalize_depth(raw_disp, mask, 0.02, 0.98)
            normalized_disp = normalized_disp.clamp(-1, 1)
        else: 
            normalized_disp = self.normalize_depth(raw_disp, mask)
            # normalized_disp = normalized_disp.clamp(-1, 1)
             
        normalized_rgb = self.normalize_rgb(rgb)
        
        # remove inf and nan
        normalized_depth[~mask.bool()] = 0
        normalized_disp[~mask.bool()] = 0
        raw_disp[~mask.bool()] = 0

        model_pred_space = "depth" # TODO config this
        return {
            "raw_disp": depth if model_pred_space == "depth" else raw_disp,
            "normalized_disp": normalized_depth if model_pred_space == "depth" else normalized_disp, 

            # for checking depth only
            "sim_mask": torch.zeros_like(mask),
            "sim_disp_unnorm": torch.zeros_like(normalized_disp),

            "normalized_rgb": normalized_rgb,
            # "left_image": self.normalize_rgb(rgb),
            # "right_image": self.normalize_rgb(rgb),
            "path": self.rgb_list[index],
            "raw_depth": depth, # raw_depth is also ground truth depth
            "mask": mask,
            "depth": depth, # ground truth depth
            "index": index,
            "fxb": fxb,
        }

    def __len__(self):
        return len(self.rgb_list)


class Tartenair(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(Tartenair, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)
        
        scene_lst = ['office2']
        for scene in scene_lst:
            depth = sorted(glob(osp.join(data_dir, f'{scene}/Easy/{scene}/{scene}/Easy/**/depth_left/**.npy')))
            rgb = sorted(glob(osp.join(data_dir, f'{scene}/Easy/{scene}/{scene}/Easy/**/image_left/**.png')))
        
            self.rgb_list.extend(rgb) 
            self.depth_list.extend(depth)
            
        assert len(self.rgb_list) == len(self.depth_list)
        print(f'Tartenair {split} data {len(self.rgb_list)}')
        
    def read_data(self, index):
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]

        depth = np.load(depth_path, allow_pickle=True)
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        return rgb, depth, mask


class HRWSI(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(HRWSI, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)
        if split != 'train':
            split = 'val'
        split = 'train'
        self.rgb_list = sorted(glob(osp.join(data_dir, f'{split}/imgs/**.jpg')))
        self.depth_list = sorted(glob(osp.join(data_dir, f'{split}/gts/**.png')))
        
        assert len(self.rgb_list) == len(self.depth_list)
        print(f'HRWSI {split} data {len(self.rgb_list)}')
        
    def read_data(self, index):
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]

        disp = np.array(Image.open(depth_path)).astype(np.float64) 
        print(disp.min(), disp.max())
        depth = 10 / disp
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        print(depth.shape, rgb.shape, depth.max(), depth.min())
        return rgb, depth, mask


class ScanNet(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(ScanNet, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)

        split_file = f'{data_dir}/splits/scannetv2_{split}.txt'
        with open(split_file, 'r') as f:
            traj_ids = f.readlines()
            
        for traj_id in traj_ids:
            traj_id = traj_id.split('\n')[0]
            root = f"{data_dir}/processed/{traj_id}/"
            rgb_list = sorted(glob(osp.join(root, f'color/**.jpg')))
            depth_list = sorted(glob(osp.join(root, f'depth/**.png')))
            self.rgb_list.extend(rgb_list)
            self.depth_list.extend(depth_list)
            
        assert len(self.rgb_list) == len(self.depth_list)
        # print(f'ScanNet {split} traj {len(traj_ids)}')
        print(f'ScanNet {split} data {len(self.rgb_list)}')
        
    def read_data(self, index):
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]

        depth = cv2.imread(depth_path, -1).astype(np.float32)
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = depth / 1000    # around 0~10
        return rgb, depth, mask

class HyperSim(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(HyperSim, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)

        split_file = f'{data_dir}/splits/{split}_v1.txt'
        self.is_test = not (split == "train")
        
        bad_paths = []
        if os.path.exists(f"{data_dir}/bad_hypersim_{split}.txt"):
            with open(f"{data_dir}/bad_hypersim_{split}.txt", 'r') as f:
                bad_paths = f.readlines()
            bad_paths = [path.split('\n')[0] for path in bad_paths]

        with open(split_file, 'r') as f:
            traj_ids = f.readlines()
        for traj_id in traj_ids:
            traj_id = traj_id.split('\n')[0]
            rgb_lst = sorted(glob(osp.join(data_dir, f'raw/{traj_id}/images/scene_cam_**_final_preview/frame.**.tonemap.jpg'), recursive=True))
            depth_lst = sorted(glob(osp.join(data_dir, f'raw/{traj_id}/images/scene_cam_**_geometry_hdf5/frame.**.depth_meters.hdf5'), recursive=True))

            if len(bad_paths) > 0:
                for rgb, depth in zip(rgb_lst, depth_lst):
                    if rgb in bad_paths:
                        continue
                    self.rgb_list.append(rgb)
                    self.depth_list.append(depth)
            else:
                self.rgb_list.extend(rgb_lst)
                self.depth_list.extend(depth_lst)

        assert len(self.rgb_list) == len(self.depth_list)
        print(f'HyperSim {split} data {len(self.rgb_list)}')
        
        intWidth = 1024
        intHeight = 768
        self.fltFocal = 886.81
        npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
        npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
        npyImageplaneZ = np.full([intHeight, intWidth, 1], self.fltFocal, np.float32)
        self.npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
        
    def read_data(self, index):
        '''
        depth range is quite large: 0. ~ 20+
        '''
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]
        fd = h5py.File(depth_path, 'r')
        # NOTE hypersim raw data stores the distance from the camera center. Convert it to traditional planar depth. 
        depth = fd['dataset'] / np.linalg.norm(self.npyImageplane, 2, 2) * self.fltFocal
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        return rgb, depth, mask # 768 x 1024

class SynTODD(MonoDataset):
    
    def __init__(self, root, split='train', image_size=-1, augment=None, sparse=False, reader=None, normalizer=None):
        assert split in ['train', 'val', 'test'], "Invalid split!"
        super(SynTODD, self).__init__(image_size, 3.0, augment)
        # super(SynTODD, self).__init__(augment, reader=reader, normalizer=normalizer)

        self.root = root 
        self.split = split
        self.dataset_path = self.root + '/' +  self.split + '_png'
        assert os.path.exists(self.dataset_path)
        self.camera_params = self.LoadCameraParams()
        self._add_things()

    def read_data(self, index):
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]

        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        mask = (depth> 0) & (depth<self.max_depth)
        return rgb, depth, mask

    def LoadCameraParams(self):
        camera_params = {'focal_length': 613.9624633789062, 
                            'camera_intrinsic': [[613.9624633789062, 0, 324.4471435546875], [0, 613.75634765625, 239.1712188720703], [0, 0, 1]], 
                            'resolution_y':480, 
                            'resolution_x':640, 
                            'baseline': 0.06499999761581421}
        return camera_params

    def _add_things(self):
        rgb_list = sorted(glob(osp.join(self.root, f"{self.split}_png/*_ir_l.png")))
        count = 0
        for rgb in rgb_list:
            depth = rgb.replace("_ir_l.png", "_depth.exr")
            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            if self.split == "val" and count > 1000: break
            count += 1

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index):
        ret =  super().__getitem__(index)
        ret["K"] = np.array(self.camera_params["camera_intrinsic"], dtype=np.float32)
        ret["device"] = "syntodd"
        return ret

class VK2(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(VK2, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)

        self.depth_list = sorted(glob(osp.join(data_dir, f'Scene**/**/frames/depth/Camera_*/depth_**.png'), recursive=True))
        self.rgb_list = sorted(glob(osp.join(data_dir, f'Scene**/**/frames/depth/Camera_*/rgb_**.jpg'), recursive=True))
            
        assert len(self.rgb_list) == len(self.depth_list)
        print(f'Virtual KITTI 2 {split} data {len(self.rgb_list)}')
        
    def read_data(self, index):
        '''
        depth range is quite large: 0. ~ 20+
        '''
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]
        depth = cv2.imread(depth_path, -1).astype(np.float32)   # raw resolution: (192, 256)
        # depth = cv2.resize(depth, dsize=(4*depth.shape[1], 4*depth.shape[0]), interpolation=cv2.INTER_NEAREST)  #  (768, 1024)
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]   # raw resolution: (1440, 1920)
        
        print(depth.shape, rgb.shape)
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(depth.max(), depth.min())
        raise NotImplementedError
        depth = depth / 500   # make it to around 0~10
        return rgb, depth, mask


class SceneNet(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(SceneNet, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)

        self.rgb_list = sorted(glob(osp.join(data_dir, f'{split}/**/**/photo/**.jpg')))
        self.depth_list = sorted(glob(osp.join(data_dir, f'{split}/**/**/depth/**.png')))
        
        assert len(self.rgb_list) == len(self.depth_list)
        print(f'SceneNet {split} data {len(self.rgb_list)}')
    
    def read_data(self, index):
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]
        depth = cv2.imread(depth_path, -1).astype(np.float32)   # (240, 320)
        depth = cv2.resize(depth, dsize=(2*depth.shape[1], 2*depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = depth / 1000   # make it to around 0~10
        return rgb, depth, mask

class NYUv2(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100, augment=None):
        super(NYUv2, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)

        self.is_test = not (split == "train")
        self.rgb_list = sorted(glob(osp.join(data_dir, f'{split}/**/**.h5')))
        self.depth_list = self.rgb_list
        print(f'NYUv2 {split} data {len(self.rgb_list)}')

        # abnormals = [42, 297, 343, 607, 36, 102, 497, 153, 445, 91, 186, 146, 322, 46, 14,
        #              311, 48, 355, 276, 47, 383, 309, 368]
        abnormals = []
        self.ab_rgb_list = []
        self.ab_dep_list = []
        if len(abnormals) > 0:
            for i in abnormals :
                self.ab_rgb_list.append(self.rgb_list[i])
                self.depth_list.append(self.depth_list[i])

            print(f'NYUv2 {split} abnormal data {len(self.ab_rgb_list)}')   
            self.rgb_list = self.ab_rgb_list
            self.depth_list = self.ab_dep_list
    
    def read_data(self, index):
        # rgb and depth are in the same file, so only need to read once
        rgb_path = self.rgb_list[index]
        h5f = h5py.File(rgb_path, "r")
        rgb = np.array(h5f["rgb"])  # (3, 480, 640)
        depth = np.array(h5f["depth"])   # (480, 640)

        # https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/evaluate.py#L127
        mask = np.zeros_like(depth)
        mask[45:471, 41:601] = 1 # eigen crop
        mask = mask.astype(bool) & (depth > 1e-3) & (depth < 10.0) & ~np.isinf(depth) & ~np.isnan(depth)
        
        # nearest neighbor interpolation for missing values
        if (~mask).sum() > 0 and not self.is_test:
            depth = frame_utils.interpolate_missing_pixels(depth, (~mask), "nearest")
            new_mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth) 
            assert new_mask.sum() >= mask.sum()

        # if not self.is_test:
        depth[mask] = np.clip(depth[mask], 1e-3, 10.0)
        
        # w = 10
        # rgb = torch.from_numpy(rgb[:, w:-w, w:-w])
        # rgb = F.pad(rgb, (w, w, w, w), fill=0.5, padding_mode='symmetric').numpy()

        # rgb = rgb[:, 45:471, 41:601]
        # depth = depth[45:471, 41:601]
        # mask = mask[45:471, 41:601]

        # depth is in around 0~10
        rgb = np.transpose(rgb, (1, 2, 0))
        return rgb, depth, mask

class ScanNetpp(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(ScanNetpp, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)
        split_file = osp.join(data_dir, f'splits/nvs_sem_{split}.txt')
        with open(split_file, 'r') as f:
            traj_ids = f.readlines()
        
        for traj_id in traj_ids:
            traj_id = traj_id.split('\n')[0]
            rgb_lst = sorted(glob(osp.join(data_dir, f'data/{traj_id}/iphone/rgb/**.jpg')))
            depth_lst = sorted(glob(osp.join(data_dir, f'data/{traj_id}/iphone/depth/**.png')))
            mask_lst = sorted(glob(osp.join(data_dir, f'data/{traj_id}/iphone/rgb_masks/**.png')))
            
            # assert len(rgb_lst) == len(depth_lst) and len(depth_lst) == len(mask_lst), f"{traj_id} {len(rgb_lst)}  {len(depth_lst)}  {len(mask_lst)}"
            
            self.rgb_list.extend(rgb_lst)
            self.depth_list.extend(depth_lst)
            self.mask_list.extend(mask_lst)
            
        assert len(self.rgb_list) == len(self.depth_list) and len(self.rgb_list) == len(self.mask_list)

        print(f'ScanNet++ {split} data {len(self.rgb_list)}')
    
    def read_data(self, index):
        '''
        Use NEAREST interpolation to resize rgb. Because there is some anonymization mask with color  (255,0,255).
        '''
        depth_path = self.depth_list[index]
        rgb_path = self.rgb_list[index]
        # mask_path = self.mask_list[index]
        
        depth = cv2.imread(depth_path, -1).astype(np.float32)   # raw resolution: (192, 256)
        depth = cv2.resize(depth, dsize=(4*depth.shape[1], 4*depth.shape[0]), interpolation=cv2.INTER_NEAREST)  #  (768, 1024)
        
        # mask = cv2.imread(mask_path, -1)    # raw resolution: (1440, 1920)
        # mask = cv2.resize(mask, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)  #  (768, 1024)
        # mask = (mask == 255) & (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]   # raw resolution: (1440, 1920)
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        depth = depth / 500   # make it to around 0~10
        return rgb, depth, mask


class InStereo2K(MonoDataset):
    def __init__(self, data_dir, split, image_size=-1, max_depth=100000, augment=None):
        super(InStereo2K, self).__init__(image_size=image_size, max_depth=max_depth, augment=augment if split == 'train' else None)
        self.disparity_list = []
        
        image1_list = sorted( glob(osp.join(data_dir, f'{split}/part*/**/left.png')) )
        disp_list = sorted( glob(osp.join(data_dir, f'{split}/part*/**/left_disp.png')) )
        
        for img1, disp in zip(image1_list, disp_list):
            self.rgb_list += [img1]
            self.disparity_list += [ disp ]

        assert len(self.rgb_list) == len(self.disparity_list) 

    def read_data(self, index):
        disp_path = self.disparity_list[index]
        rgb_path = self.rgb_list[index]
        
        disp = np.array(Image.open(disp_path)).astype(np.float64)   # raw resolution: (1440, 1920)
        depth = 10000 / disp        # [860, 1080], depth range is around 0~5
        mask = (depth > 0) & (depth < self.max_depth) & ~np.isinf(depth) & ~np.isnan(depth)
        
        rgb = np.array(Image.open(rgb_path)).astype(np.uint8)[...,:3]   # raw resolution: (1440, 1920)
        rgb = cv2.resize(rgb, dsize=(depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return rgb, depth, mask
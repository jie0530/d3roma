# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T 
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2

from utils import frame_utils
# from utils.utils import normalize_disp, unnormalize_disp
from data.augmentor import FlowAugmentor, SparseFlowAugmentor
from functools import partial
import cv2
from PIL import Image
from utils.utils import Normalizer
from typing import Optional
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from utils.camera import DepthCamera, Realsense

normalize_rgb = lambda x: (x / 255. - 0.5) * 2

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, normalizer=None):
        self.augmentor = None
        self.sparse = sparse
        self.normalizer: Optional[Normalizer] = normalizer
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.sim_disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        index = index % len(self.image_list)
        # bugs = [20231, 22081, 19675, 13033, 12782,  6265,   280,  8574]
        # index = index % len(bugs)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid, min_disp, max_disp = disp
        else:
            min_disp = 0
            max_disp = 512
            valid = (disp < max_disp) and (disp > min_disp)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        img1 = cv2.resize(img1, dsize=disp.shape[::-1], interpolation=cv2.INTER_NEAREST)
        img2 = cv2.resize(img2, dsize=disp.shape[::-1], interpolation=cv2.INTER_NEAREST)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        if index < len(self.sim_disparity_list):
            # np.array(Image.open(self.sim_disparity_list[index]))
            sim_disp, sim_valid, *_ = self.disparity_reader(self.sim_disparity_list[index])
            sim_disp = np.array(sim_disp).astype(np.float32)
            # sim_disp = cv2.resize(sim_disp, dsize=disp.shape[::-1], interpolation=cv2.INTER_NEAREST)
            assert sim_disp.shape[:2] == disp.shape[:2]
            
            epe = np.abs(sim_disp[sim_valid] - disp[sim_valid]).mean()
            # if epe > 20:
            #     print(f"very bad quality sim disp, {self.disparity_list[index]} {epe}")
            sim_flow = np.stack([-sim_disp, np.zeros_like(sim_disp)], axis=-1)
        else:
            sim_flow = None
            sim_valid = None

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid, sim_flow, sim_valid = self.augmentor(img1, img2, flow, valid, sim_flow, sim_valid)
            else:
                img1, img2, flow, sim_flow, sim_valid = self.augmentor(img1, img2, flow, sim_flow, sim_valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        sim_flow = torch.from_numpy(sim_flow).permute(2, 0, 1).float()
        sim_valid = torch.from_numpy(sim_valid).unsqueeze(0)

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0:1].abs() < max_disp) & (flow[1:].abs() < max_disp)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        disp = torch.clamp(-flow[0:1], 0.25, max_disp)
        sim_disp = torch.clamp(-sim_flow[0:1], 0.25, max_disp)
        valid = valid & (disp > min_disp)
        #print(np.abs(sim_disp[valid>0]-disp[valid>0]).mean())
        
        result = {
            "raw_disp": disp, # ground truth
            "normalized_disp": self.normalizer.normalize(disp, valid)[0],

            # for checking depth only
            "sim_mask": sim_valid.float() if self.space == "disp" else torch.zeros_like(valid),
            "sim_disp_unnorm": sim_disp if self.space == "disp" else torch.zeros_like(disp),

            # "sim_mask": sim_valid.float(),
            # "sim_disp_unnorm": sim_disp,
            "sim_disp": self.normalizer.normalize(sim_disp, valid & sim_valid)[0],
            "left_image": normalize_rgb(img1),
            "right_image": normalize_rgb(img2),
            "path": self.disparity_list[index],
            "mask": valid.float(),
            "index": index,
            "fxb": 1.0,
            "depth": 1 / disp,
            "normalized_rgb": normalize_rgb(img1)
        }

        result["K"] = np.eye(3)
        result["device"] = "unknown"

        return result

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)
        
class SceneFlow(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False, reader=None, normalizer=None):
        super(SceneFlow, self).__init__(aug_params, reader=reader, normalizer=normalizer)
        self.root = root
        self.dstype = dstype
        self.space = "disp"

        self.bad_paths = []
        if os.path.exists(f"{root}/bad_sceneflow_train.txt"):
            with open(f"{root}/bad_sceneflow_train.txt", 'r') as f:
                bad_paths = f.readlines()
            self.bad_paths = [path.split(' ')[0] for path in bad_paths]

        if things_test:
            self.is_test = True
            self._add_things("TEST")
        else:
            self.is_test = False
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        sim_disp_images = [ im.replace('frames', 'raw').replace(self.dstype, 'cleanpass') for im in left_images ] # use sim disparity from cleanpass

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp, sim) in enumerate(zip(left_images, right_images, disparity_images, sim_disp_images)):
            if disp in self.bad_paths: continue
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                self.sim_disparity_list += [ sim ]

        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        sim_disp_images = [ im.replace('frames', 'raw').replace('finalpass', 'cleanpass') for im in left_images ]

        for img1, img2, disp, sim in zip(left_images, right_images, disparity_images, sim_disp_images):
            if disp in self.bad_paths: continue
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            self.sim_disparity_list += [ sim ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]
        sim_disp_images = [ im.replace('frames', 'raw').replace(self.dstype, 'cleanpass') for im in left_images ]

        for img1, img2, disp, sim in zip(left_images, right_images, disparity_images, sim_disp_images):
            if disp in self.bad_paths: continue
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            self.sim_disparity_list += [ sim ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            extra_info_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/calib.txt') for name in lines])
            
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                self.rgb_list += [ img1 ]

            for ex in extra_info_list:
                self.extra_info += [ex]

class ActiveStereoDataset(StereoDataset):
    def __init__(self, camera, normalizer, image_size, split="train", space="disp", aug_params=None, reader=None):
        super(ActiveStereoDataset, self).__init__(aug_params, reader=reader, sparse=True, normalizer=normalizer)
        
        self.rgb_list = []
        self.depth_list = []
        self.raw_depth_list = []
        self.space = space

        self.camera: DepthCamera = camera
        self.image_size = image_size
        self.split = split

        if type(image_size) == int:
            self.image_size = (image_size, image_size) # H x W
        elif type(image_size) == tuple:
            self.image_size = image_size
        else:
            raise ValueError("image_size must be int or tuple")
        
        self.is_test = not (split == "train")

    def __getitem__(self, index):
        # result = super().__getitem__(index)
        # assert tuple(result["normalized_disp"].shape[-2:]) == tuple(self.image_size)
        # index = 22949 # DEBUG
        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid, min_disp, max_disp = disp
        else:
            min_disp = 0
            max_disp = 512
            valid = (disp < max_disp) and (disp > min_disp)
        
        rgb = np.array(Image.open(self.rgb_list[index])).astype(np.uint8)[...,:3]

        if self.image_list is not None and len(self.image_list) > 0:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
        else:
            img1 = np.zeros_like(rgb)
            img2 = np.zeros_like(rgb)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if img1.shape[:2] != self.camera.resolution:
            # be very carefull here !!!
            img1 = cv2.resize(img1, dsize=self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, dsize=self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)
            rgb = cv2.resize(rgb, dsize=self.camera.resolution[::-1], interpolation=cv2.INTER_LINEAR)

        disp = np.array(disp).astype(np.float32)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()

        # nearest neighbor interpolation for missing values
        if (~valid).sum() > 0 and not self.is_test and False:
            disp = frame_utils.interpolate_missing_pixels(disp, (~valid), "nearest")

        def read_depth(filename):
            """ return h,w,1 """
            depth = np.array(frame_utils.read_gen(filename))
            depth_unit = 1.0
            if self.camera.device == "fxm" or self.camera.device == "jav" or self.camera.device == "d435" or self.camera.device == "clearpose":
                depth_unit = 1e-3
                depth = depth.astype(np.int32)

            depth = cv2.resize(depth, dsize=self.camera.resolution[::-1], interpolation=cv2.INTER_NEAREST)
            depth = np.array(depth).astype(np.float32) * depth_unit
            if len(depth.shape) == 3 and depth.shape[-1] == 3:
                depth = depth [...,0]
            if len(depth.shape) == 2:
                depth = depth[...,None]
            return depth

        depth = read_depth(self.depth_list[index])
        if self.__class__.__name__ in (["HISS", "Gapartnet2"]) and self.space == "depth" and False:
            depth[...,0] = frame_utils.interpolate_missing_pixels(depth[...,0], (~valid), "nearest")

        depth = torch.from_numpy(depth).permute(2, 0, 1).float() # [1,h,w]

        raw_depth = read_depth(self.raw_depth_list[index])
        raw_depth = torch.from_numpy(raw_depth).permute(2, 0, 1).float()
        raw_depth[raw_depth<0] = 0 # for dreds

        disp = torch.from_numpy(disp).float().unsqueeze(0)  # [1,h,w]
        valid = torch.from_numpy(valid).float().unsqueeze(0)  # [1,h,w]
        
        def random_crop_with_margin(x, margin=16): # horizontal margin left
            H, W = self.image_size # crop size
            H1, W1 = x.shape[-2:] # image size
            H2, W2 = H, W + margin # *2 # margined image size
            if not (H < H1 and W + margin < W1):
                # "invalid margin, do nothing"
                return x, 0, 0, 0
            
            margin = min(margin, (W1 - W) )# // 2
            off_y = random.randint(0, H1 - H)
            off_x = random.randint(0, W1 - W - margin)# * 2
            x = x[..., off_y:off_y+H, off_x:off_x+W+margin]# *2
            if x.shape[-1] < W2:
                # padding with border
                to_pad = (W2 - x.shape[-1])
                x = F.pad(x, (to_pad, 0), mode="replicate")
            return x, off_x, off_y, margin

        dr12 = torch.vstack([disp, rgb, img1, img2, valid, depth])
        if raw_depth is not None:
            dr12 = torch.vstack([dr12, raw_depth])

        if self.split == "train":
            transform = T.RandomHorizontalFlip(p=0.5) 
            dr12 = transform(dr12)

        margin_left = self.camera.config["margin_left"]
        assert margin_left == 0, "not implemented yet"
        dr12, off_x, off_y, margin_left = random_crop_with_margin(dr12, margin_left)
        if raw_depth is not None:
            disp, rgb, img1, img2, valid, depth, raw_depth = torch.split(dr12, [1, 3, 3, 3, 1, 1, 1], dim=0)
        else:
            disp, rgb, img1, img2, valid, depth = torch.split(dr12, [1, 3, 3, 3, 1, 1], dim=0)

        # center_crop = transforms.CenterCrop(self.image_size)
        def left_crop(x, margin=16):
            # H, W = self.image_size
            return x[..., margin:]

        # center crop without margin
        disp = left_crop(disp, margin_left)
        rgb = left_crop(rgb, margin_left)
        valid = left_crop(valid, margin_left)
        depth = left_crop(depth, margin_left)
        if raw_depth is not None:
            raw_depth = left_crop(raw_depth, margin_left)

        if self.space == "depth":
            gt_data = depth
            normalized_data = self.normalizer.normalize(depth, valid)[0]
            # if depth.max() > depth[valid>0].max():
            #     print("hello world")
            normalized_sim = self.normalizer.normalize(raw_depth, raw_depth > 0)[0]
            
        elif self.space == "disp":
            gt_data = disp
            normalized_data, low, up = self.normalizer.normalize(disp, valid)
            sim_disp = torch.zeros_like(raw_depth)
            sim_valid = valid.to(torch.bool) & (raw_depth > 0)
            if sim_valid.sum() == 0:
                # raise ValueError(f"no valid sim data at {index}: {self.raw_depth_list[index]}")
                print(f"warning: no valid sim data at {index}: {self.raw_depth_list[index]}")
            sim_disp[sim_valid] = self.camera.fxb_depth / raw_depth[sim_valid]
            # raw_depth = sim_disp # ugly hack, for guidance?
            normalized_sim = self.normalizer.normalize(sim_disp, sim_valid, low=low, up=up)[0]
        else:
            raise NotImplementedError
        
        result = {
            "raw_disp": gt_data, # bad key naming
            "normalized_disp": normalized_data, # bad key naming
            "sim_disp": normalized_sim, # bad key naming

            # for checking depth only
            "sim_mask": sim_valid.float() if self.space == "disp" else torch.zeros_like(valid),
            "sim_disp_unnorm": sim_disp if self.space == "disp" else torch.zeros_like(disp),

            "normalized_rgb": normalize_rgb(rgb),
            "left_image": normalize_rgb(img1), # c,h,w
            "right_image": normalize_rgb(img2),
            "path": self.raw_depth_list[index],
            "mask": valid.float(),
            "depth": depth, # gt
            # "disp": disp, # gt
            "index": index,
            "fxb": self.camera.fxb_depth
        }
        # self.camera.change_resolution(f"{self.image_size[1]}x{self.image_size[0]}")
        if self.split != "train":
            result["raw_depth"] = raw_depth

        result["K"] = self.camera.K_depth.arr
        result["device"] = self.camera.device
        return result

class Dreds(ActiveStereoDataset):
    
    def __init__(self, camera, normalizer, image_size, split="train", space="depth", aug_params=None):
        super().__init__(camera, normalizer, image_size, split, space, aug_params, 
                         reader=partial(frame_utils.readDispDreds_exr, camera))

        self.is_test = not (split == "train") # ?
        self.split = split

        root = f"datasets/DREDS/{split}"
        rgb_list = sorted(glob(osp.join(root, f'**/*color.png'), recursive=True))

        gt_depth_ext = "_gt_depth.exr" if split.startswith("test_std") else "depth_120.exr"
        depth_list = sorted(glob(osp.join(root, f'**/*{gt_depth_ext}'), recursive=True))

        raw_depth_ext = "_depth_415.exr" if split.startswith("test_std") else "_simDepthImage.exr"
        raw_depth_list = sorted(glob(osp.join(root, f'**/*{raw_depth_ext}'), recursive=True))
        raw_disparity_list = sorted(glob(osp.join(root, f'**/*{raw_depth_ext}'), recursive=True))

        for rgb, depth, raw_depth, sim_disp in zip(rgb_list, depth_list, raw_depth_list, raw_disparity_list): 
            self.rgb_list += [rgb]
            self.depth_list += [ depth ]
            self.disparity_list += [ depth ] # ~~~ to convert depth to disparity
            self.raw_depth_list += [raw_depth]
            self.sim_disparity_list += [sim_disp]

        if not split.startswith("test_std"):
            image1_list = sorted(glob(osp.join(root, f'**/*ir_l.png'), recursive=True))
            image2_list = sorted(glob(osp.join(root, f'**/*ir_r.png'), recursive=True))

            for img1, img2 in zip(image1_list, image2_list):
                self.image_list += [ [img1, img2] ]

            assert len(self.rgb_list) == len(self.image_list)
        
        assert len(self.rgb_list) == len(self.depth_list) == len(self.sim_disparity_list) > 0, "no data found"
        

class HISS(ActiveStereoDataset):
    def __init__(self, camera, normalizer, image_size, split="train", space="disp", aug_params=None, reader=None):
        super().__init__(camera, normalizer, image_size, split, space, aug_params, reader)

        root = f"datasets/HISS/{split}"

        # rgb_list = sorted(glob(osp.join(root, f'**/*color.png'), recursive=True))
        # image1_list = sorted(glob(osp.join(root, f'**/*ir_l*'), recursive=True))
        # image2_list = sorted(glob(osp.join(root, f'**/*ir_r*'), recursive=True))
        # disp_list = sorted(glob(osp.join(root, f'**/*depth.exr'), recursive=True))
        depth_list = sorted(glob(osp.join(root, f'**/*depth.exr'), recursive=True))
        # raw_depth_list = sorted(glob(osp.join(root, f'**/*simDepthImage.exr'), recursive=True))
        # sim_disp_list = sorted(glob(osp.join(root, f'**/*raw_disparity*.png'), recursive=True))
        # sim_disp_list = sorted(glob(osp.join(root, f"**/*simDepthImage.exr"), recursive=True))

        self.bad_paths = []
        if os.path.exists(f"{root}/bad_his.txt"):
            with open(f"{root}/bad_his.txt", 'r') as f:
                bad_paths = f.readlines()
            self.bad_paths = [path.split(' ')[0] for path in bad_paths]

        for depth in depth_list:
            if "glass" in depth or "mirror" in depth or "bed" in depth: 
                continue
            if depth in self.bad_paths: # .replace("depth", "disp")
                continue

            rgb = depth.replace("_depth.exr", "_color.png")
            img1 = depth.replace("_depth.exr", "_ir_l.png")
            img2 = depth.replace("_depth.exr", "_ir_r.png")
            # disp = depth.replace("_depth.exr", "_disp.exr")
            # sim_disp = depth.replace("_depth.exr", "_raw_disparity.exr")
            raw_depth = depth.replace("_depth.exr", "_simDepthImage.exr")

            self.rgb_list += [rgb]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [depth]
            self.depth_list += [ depth ]
            self.sim_disparity_list += [raw_depth]
            self.raw_depth_list += [raw_depth]

        assert len(self.rgb_list) == len(self.image_list) == len(self.sim_disparity_list) > 0

class ClearPose(ActiveStereoDataset):
    def __init__(self, camera, normalizer, image_size, split="train", space="depth", aug_params=None, reader=None):
        super().__init__(camera, normalizer, image_size, split, space, aug_params, reader)

        self.is_test = not (split == "train")
        self.root = f"datasets/clearpose"
        self.train_split = {1:(1,4), 4:(1,5), 5:(1,5), 6:(1,5), 7:(1,5)}
        self.test_split = {1:(5,5),2:(0,0),3:(0,0),4:(6,6),5:(6,6),6:(6,6),7:(6,6),8:(0,0),9:(0,0)}
        self.split = split

        if split == "train":
            self.add_train()
        elif split == "val":
            self.add_test(total = 300)
        elif split == "plot":
            pics = [
                'datasets/clearpose/set9/scene10/000000-color.png',
                'datasets/clearpose/set9/scene12/000800-color.png',
                'datasets/clearpose/set8/scene4/000800-color.png',
                'datasets/clearpose/set8/scene1/001200-color.png',
                'datasets/clearpose/set1/scene5/000000-color.png',
                'datasets/clearpose/set2/scene3/000000-color.png'
            ]
            for rgb in pics:
                depth = rgb.replace("color.png", "depth_true.png")
                raw_depth = depth.replace("color.png", "depth.png")

                self.rgb_list += [rgb]
                self.image_list += [ [rgb, rgb] ]
                self.depth_list += [ depth ]
                self.disparity_list += [depth]
                self.sim_disparity_list += [raw_depth]
                self.raw_depth_list += [raw_depth]
        else:
            """ test split according to clearpose paper, Table 2 and Figure 6
                heavy occlusion: set2, set3: all scenes
                new background: set1: scene5, set4-7: scene6
                with opaque objects: set8: scene1,2,3
                translucent cover: set8: scene4, set9: scene7,8
                non planar: set8: scene5, set9: scene11,12
                filled with liquid: set8: scene6, set9: scene9,10
            """
            if split == "heavy_occ":
                self.test_split = {2:(0,0),3:(0,0)}
            elif split == "new_bg":
                self.test_split = {1:(5,5), 4:(6,6),5:(6,6),6:(6,6),7:(6,6)}
            elif split == "opaque_obj":
                self.test_split = {8:(1,3)}
            elif split == "translucent_cover":
                self.test_split = {8:(4,4), 9:(7,8)}
            elif split == "non_planar":
                self.test_split = {8:(5,5), 9:(11,12)}
            elif split == "liquid":
                self.test_split = {8:(6,6), 9:(9,10)}
            else:
                raise ValueError(f"unknown split {split}")
            self.add_test()

    def __len__(self):
        return len(self.depth_list)

    def add_train(self):
        for i in self.train_split.keys():
            cover = self.train_split[i]
            for scene in range(cover[0], cover[1]+1):
                depth_list = sorted(glob(osp.join(self.root, f"set{i}/scene{scene}/*depth_true.png")))
                for depth in depth_list:
                    rgb = depth.replace("depth_true.png", "color.png")
                    raw_depth = depth.replace("depth_true.png", "depth.png")

                    self.rgb_list += [rgb]
                    self.image_list += [ [rgb, rgb] ]
                    self.depth_list += [ depth ]
                    self.disparity_list += [depth]
                    self.sim_disparity_list += [raw_depth]
                    self.raw_depth_list += [raw_depth]

        assert len(self.rgb_list) == len(self.depth_list)  > 0

    def add_test(self, total = np.inf):
        for i in self.test_split.keys():
            cover = self.test_split[i]
            if cover[0]:
                for scene in range(cover[0], cover[1]+1):
                    depth_list = sorted(glob(osp.join(self.root, f"set{i}/scene{scene}/*depth_true.png")))
                    inc_count = 0
                    for depth in depth_list:
                        rgb = depth.replace("depth_true.png", "color.png")
                        raw_depth = depth.replace("depth_true.png", "depth.png")

                        self.rgb_list += [rgb]
                        self.image_list += [ [rgb, rgb] ]
                        self.depth_list += [ depth ]
                        self.disparity_list += [depth]
                        self.sim_disparity_list += [raw_depth]
                        self.raw_depth_list += [raw_depth]
                        if inc_count > total: break
                        inc_count +=1
            else:
                depth_list = sorted(glob(osp.join(self.root, f"set{i}/**/*depth_true.png"), recursive=True))
                inc_count = 0
                for depth in depth_list:
                    rgb = depth.replace("depth_true.png", "color.png")
                    raw_depth = depth.replace("depth_true.png", "depth.png")

                    self.rgb_list += [rgb]
                    self.image_list += [ [rgb, rgb] ]
                    self.depth_list += [ depth ]
                    self.disparity_list += [depth]
                    self.sim_disparity_list += [raw_depth]
                    self.raw_depth_list += [raw_depth]
                    if inc_count > total: break
                    inc_count +=1

        assert len(self.rgb_list) == len(self.depth_list)  > 0

class SynTODDRgbd(ActiveStereoDataset): # RGB-D version (Because of invalid stlaereo data)
    def __init__(self, variant, camera, normalizer, image_size, split="train", space="depth", aug_params=None, reader=None):
        super().__init__(camera, normalizer, image_size, split, space, aug_params, reader)
        assert variant in ["simdepth", "erodedepth", "dilatedepth"]
        self.variant = variant
        self.split = "test" #split
        self.dataset_path = "datasets/SynTODD" + '/' +  self.split + '_png'
        assert os.path.exists(self.dataset_path)
        self._add_data()

    def _add_data(self):
        """ disp_list = sorted(glob(osp.join(self.dataset_path, "*disp.exr")))
        sim_disp_list = sorted(glob(osp.join(self.dataset_path, "*simDispImage.png")))
        image1_list = sorted(glob(osp.join(self.dataset_path, "*_ir_l.png")))
        image2_list = sorted(glob(osp.join(self.dataset_path, "*_ir_r.png")))

        for img1, img2, sim_disp, disp in zip(image1_list, image2_list, sim_disp_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.sim_disparity_list += [ sim_disp ]
            self.disparity_list += [ disp ] """
        
        depth_list = sorted(glob(osp.join(self.dataset_path, "*_depth.exr")))
        bads = ['datasets/SynTODD/train_png/133800_depth.exr',
            'datasets/SynTODD/train_png/131150_depth.exr',
            'datasets/SynTODD/train_png/16346_depth.exr',
            'datasets/SynTODD/train_png/146643_depth.exr',
        ]
        total = 0
        for depth in depth_list:
            if depth in bads: continue
            left = depth.replace("_depth.exr", "_ir_l.png")
            raw_depth = depth.replace("_depth.exr", f"_{self.variant}.exr")

            self.rgb_list += [ left ]
            self.image_list += [ [left, left]]
            self.depth_list += [ depth ]
            self.disparity_list += [depth]
            self.sim_disparity_list += [raw_depth]
            self.raw_depth_list += [raw_depth]
            # if self.split == "val" and total > 100: break
            total +=1

        assert len(self.rgb_list) == len(self.depth_list) == len(self.sim_disparity_list) > 0

    def __len__(self):
        return len(self.image_list)

class Gapartnet2(ActiveStereoDataset):
    def __init__(self, camera, normalizer, image_size, split="train", space="disp", aug_params=None, reader=None):
        super().__init__(camera, normalizer, image_size, split, space, aug_params, reader)
        root = f"datasets/Gapartnet2/{split}"

        depth_list = sorted(glob(osp.join(root, f'**/depth/*_depth.exr'), recursive=True))
        self.bad_paths = []
        if os.path.exists(f"{root}/bads.txt"):
            with open(f"{root}/bads.txt", 'r') as f:
                bad_paths = f.readlines()
            self.bad_paths = [path.split(' ')[0] for path in bad_paths]

        for depth in depth_list:
            # TODO filter some files out
            # if depth != "datasets/Gapartnet2/train/Dishwasher/12484_9/depth/0006_depth.exr": continue

            if depth in self.bad_paths: continue

            rgb = depth.replace("_depth.exr", "_color.png").replace("depth", "rgb")
            img1 = depth.replace("_depth.exr", "_ir_l.png").replace("depth", "ir")
            img2 = depth.replace("_depth.exr", "_ir_r.png").replace("depth", "ir")
            # disp = depth.replace("_depth.exr", "_disp.exr")
            # sim_disp = depth.replace("_depth.exr", "_raw_disparity.exr")
            raw_depth = depth.replace("depth", "raw").replace("_raw.exr", "_raw_depth.exr")

            self.rgb_list += [rgb]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [depth]
            self.depth_list += [ depth ]
            self.sim_disparity_list += [raw_depth]
            self.raw_depth_list += [raw_depth]
        

class Real(ActiveStereoDataset):
    def __init__(self, camera, normalizer, image_size, scene, space="depth", data_dir="datasets/Real"):
        super(Real, self).__init__(camera=camera, normalizer=normalizer, image_size=image_size, split="val", space=space, 
                                aug_params=None, reader=partial(frame_utils.readDispReal, camera))

        # an ugly hack:
        if scene == "val":
            scene = "xiaomeng"
        root = f"{data_dir}/" + (scene if scene is not None else "")
        rgb_list = sorted(glob(osp.join(root, f'**/*rgb*'), recursive=True))
        image1_list = sorted(glob(osp.join(root, f'**/*ir_l*'), recursive=True))
        image2_list = sorted(glob(osp.join(root, f'**/*ir_r*'), recursive=True))
        # disp_list = sorted(glob(osp.join(root, f'**/*disp.exr'), recursive=True))
        # depth_list = sorted(glob(osp.join(root, f'**/*rs.npy'), recursive=True))
        depth_list = sorted(glob(osp.join(root, f'**/*depth*'), recursive=True))
        # depth_list = sorted(glob(osp.join(root, f'**/*depth*.npy'), recursive=True))
        # sim_disp_list = sorted(glob(osp.join(root, f'**/*raw_disparity*.png'), recursive=True))

        # i = 0
        for rgb, img1, img2, depth in zip(rgb_list, image1_list, image2_list, depth_list): # ,sim_disp, sim_disp_list
            self.rgb_list += [rgb]
            self.image_list += [ [img1, img2] ]
            self.depth_list += [ depth ]
            self.raw_depth_list += [depth]
            self.disparity_list += [ depth ]
            # self.sim_disparity_list += [sim_disp]

        assert len(self.rgb_list) == len(self.image_list)> 0 #  == len(self.sim_disparity_list) 

    def __getitem__(self, index):
        result = super().__getitem__(index)
        # valid = (disp < self.camera.max_disp) & (disp > self.camera.min_disp)
        # assert torch.abs(sim_disp[valid>0].mean() - disp[valid>0].mean()) < 2.0, "sim & raw should be close"
        return result


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''))
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlow(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlow(aug_params, dstype='frames_finalpass')
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI(aug_params, split=dataset_name)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

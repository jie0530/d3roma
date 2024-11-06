from torchvision.transforms import RandomResizedCrop, InterpolationMode
import torchvision.transforms.functional as TF
import torch 
import functools

class WarpDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, augment):
        self.augment = augment
        self.rgb_list = []
        self.depth_list = []
        self.lr_list = []
        self.mask_list = []
        
        if self.augment is None:
            self.augment = dict()
        if type(image_size) == int:
            self.image_size = (image_size, image_size) # H x W
        elif type(image_size) == tuple:
            self.image_size = image_size
        else:
            raise ValueError("image_size must be int or tuple")
        return 
    
    def data_aug(self, rgb, depth, mask, img1=None, img2=None, raw_depth=None):
        # random crop and resize. 
        safe_apply = lambda func, x: func(x) if x is not None else None
        if 'resizedcrop' in self.augment.keys():
            param = self.augment['resizedcrop']
            i, j, h, w = RandomResizedCrop.get_params(rgb, scale=param['scale'], ratio=param['ratio'])
            resized_crop = lambda i, j, h, w, size, interp, x: TF.resized_crop(x, i, j, h, w, size=size, interpolation=interp)
            resized_crop_fn = functools.partial(resized_crop, i,j,h,w,self.image_size, InterpolationMode.NEAREST)
            rgb, mask, depth, img1, img2 = map(lambda x: safe_apply(resized_crop_fn, x), [rgb, mask, depth, img1, img2])

            """ rgb =  TF.resized_crop(rgb, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            mask = TF.resized_crop(mask, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            depth = TF.resized_crop(depth, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            if img1 is not None:
                img1 =  TF.resized_crop(img1, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST)
                img2 = TF.resized_crop(img2, i, j, h, w, size=self.image_size, interpolation=InterpolationMode.NEAREST) """
        else:   # only resize when eval and test
            resize = lambda size, interp, x: TF.resize(x, size=size, interpolation=interp)
            resize_fn = functools.partial(resize, self.image_size, InterpolationMode.NEAREST)
            rgb, mask, depth, img1, img2 = map(lambda x: safe_apply(resize_fn, x), [rgb, mask, depth, img1, img2])

            # rgb = TF.resize(rgb, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            # mask = TF.resize(mask, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            # depth = TF.resize(depth, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            # if img1 is not None:
            #     img1 = TF.resize(img1, size=self.image_size, interpolation=InterpolationMode.NEAREST)
            #     img2 = TF.resize(img2, size=self.image_size, interpolation=InterpolationMode.NEAREST)

        # Random hflip
        if 'hflip' in self.augment.keys():
            param = self.augment['hflip']
            if torch.rand(1) < 0.5: #param['prob']:
                rgb, mask, depth, img1, img2 = map(lambda x: safe_apply(TF.hflip, x), [rgb, mask, depth, img1, img2])
                """ rgb = TF.hflip(rgb)
                mask = TF.hflip(mask)
                depth = TF.hflip(depth)
                if img1 is not None:
                    img1 = TF.hflip(img1)
                    img2 = TF.hflip(img2) """
                    
        # TODO add color augmentation such as changing the lighting 
             
        if img1 is None:   
            return rgb, depth, mask
        else:
            return rgb, depth, mask, img1, img2
    
      
    def normalize_depth(self, depth, mask, low_p=0.00, high_p=1.00):
        """ low_p, high_p: low and high percentile to normalize the depth"""
        mask = mask.bool()
        masked_depth = depth[mask]
        low, high = torch.quantile(masked_depth, torch.tensor((low_p, high_p)))

        depth = (depth - low) / (high - low)
        depth = (depth - 0.5) * 2   # [0,1] -> [-1, 1]
        return depth

    def normalize_rgb(self, rgb):
        return (rgb / 255 - 0.5) * 2 # [0,1] -> [-1, 1]
    
    def __mul__(self, v):
        self.rgb_list = v * self.rgb_list
        self.depth_list = v * self.depth_list
        self.lr_list = v * self.lr_list
        self.mask_list = v * self.mask_list
        return self
    
    def __len__(self):
        return len(self.rgb_list)
            
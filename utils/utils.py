import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from PIL import Image
import math
import open3d as o3d
import random
import json
import matplotlib.pyplot as plt
cmap_jet = plt.get_cmap('jet')
from scipy.optimize import minimize


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.cat(distances, dim=0)
    return dist


def ensemble_depths(
    input_images: torch.Tensor,
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    reduction: str = "median",
    max_res: int = None,
):
    """
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the scale and shift
    """
    device = input_images.device
    dtype = input_images.dtype
    np_dtype = np.float32

    original_input = input_images.clone()
    n_img = input_images.shape[0]
    ori_shape = input_images.shape

    if max_res is not None:
        scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            input_images = downscaler(torch.from_numpy(input_images)).numpy()

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype)

    input_images = input_images.to(device)

    # objective function
    def closure(x):
        l = len(x)
        s = x[: int(l / 2)]
        t = x[int(l / 2) :]
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)

        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = torch.sqrt(torch.mean(dists**2))

        if "mean" == reduction:
            pred = torch.mean(transformed_arrays, dim=0)
        elif "median" == reduction:
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError

        near_err = torch.sqrt((0 - torch.min(pred)) ** 2)
        far_err = torch.sqrt((1 - torch.max(pred)) ** 2)

        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy().astype(np_dtype)
        return err

    res = minimize(
        closure, x, method="BFGS", tol=tol, options={"maxiter": max_iter, "disp": False}
    )
    x = res.x
    l = len(x)
    s = x[: int(l / 2)]
    t = x[int(l / 2) :]

    # Prediction
    s = torch.from_numpy(s).to(dtype=dtype).to(device)
    t = torch.from_numpy(t).to(dtype=dtype).to(device)
    transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1)
    if "mean" == reduction:
        aligned_images = torch.mean(transformed_arrays, dim=0)
        std = torch.std(transformed_arrays, dim=0)
        uncertainty = std
    elif "median" == reduction:
        aligned_images = torch.median(transformed_arrays, dim=0).values
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = torch.abs(transformed_arrays - aligned_images)
        mad = torch.median(abs_dev, dim=0).values
        uncertainty = mad
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

    # Scale and shift to [0, 1]
    _min = torch.min(aligned_images)
    _max = torch.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= _max - _min

    return aligned_images, uncertainty

def pyramid_noise_like(x, discount=0.9):
    b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = random.random()*2+2 # Rather than always going 2x,
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise_i = torch.randn(b, c, w, h).to(x)
        noise += u(noise_i) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance

def debug_stat(x):
    return f"max={x.max()}, mean={x.mean()}, min={x.min()}, std={x.std()}"

class Normalizer():
    def __init__(self, ssi=False, mode="piecewise", num_chs=3, ch_bounds = [64, 32, 32], ch_gammas = [1.,1.,1.], t = 0.5, s = 2.0,
            safe_ssi = True, ransac_error_threshold=0.6, ):
        """ num_chs: number of channels
            ch_bounds: [bound0, bound1, bound2]
            ch_gammas: [gamma0, gamma1, gamma2]
            mode: piecewise or average
        """
        self.ssi = ssi
        self.ch_bounds  = ch_bounds
        self.ch_gammas = ch_gammas
        self.num_chs = num_chs
        self.mode = mode
        self.t = t
        self.s = s
        # safe ssi
        self.safe_ssi = safe_ssi
        self.ransac_error_threshold = ransac_error_threshold
        self.low_p = 0.0
        self.high_p = 1.0


    @staticmethod
    def from_config(config):
        return Normalizer(config.ssi,
                          config.normalize_mode, 
                          config.num_chs,
                          config.ch_bounds,
                          config.ch_gammas,
                          config.norm_t,
                          config.norm_s,
                          config.safe_ssi,
                          config.ransac_error_threshold)
    
    def normalize(self, x, mask=None, low=None, up=None):
        if self.ssi:
            if mask is None:
                mask = torch.ones_like(x).to(torch.bool)
            else:
                mask = mask.to(torch.bool)
            if low is None and up is None:
                low, up = torch.quantile(x[mask.to(torch.bool)], torch.tensor([self.low_p, self.high_p], dtype=x.dtype))
            y = torch.zeros_like(x)
            y[mask] = (torch.clamp((x[mask] - low) / (up - low), 0, 1) - self.t) * self.s
            return y, low, up
        else: 
            y = self.__normalize(x)
            return (y - self.t) * self.s, None, None

    def __normalize(self, x):
        x = torch.clamp(x, max=torch.sum(torch.tensor(self.ch_bounds, dtype=torch.float32)))
        assert x.max() <= np.sum(self.ch_bounds), "out of bound"
        assert len(x.shape) == 3 # 1,H,W

        if self.mode == "average": #Hack
            assert len(self.ch_bounds) == len(self.ch_gammas) == 1, "inconsistent params"
            gamma = self.ch_gammas[0]
            bound = self.ch_bounds[0]
            image_ch = x / bound
            return torch.cat([image_ch ** gamma] * self.num_chs, dim=0)
        else:
            assert len(self.ch_bounds) == len(self.ch_gammas) == self.num_chs, "inconsistent params"
            if self.num_chs >= 1:
                gamma = self.ch_gammas[0]
                bound = self.ch_bounds[0]
            if self.num_chs >= 2:
                gamma1 = self.ch_gammas[1]
                bound1 = self.ch_bounds[1]
            if self.num_chs >= 3:
                gamma2 = self.ch_gammas[2]
                bound2 = self.ch_bounds[2]

            # gamma, gamma1, gamma2 = self.ch_gammas
            # bound, bound1, bound2 = self.ch_bounds
            image_ch0 = torch.minimum(x, torch.ones_like(x)*bound) / bound
            residual0 = torch.where(image_ch0 < 1.0, torch.zeros_like(x), x - bound)
            if self.num_chs == 1:
                assert torch.max(residual0) == 0, "bug"
                y = image_ch0 ** gamma
                return y

            image_ch1 = torch.minimum(residual0, torch.ones_like(x)*bound1) / bound1
            residual1 = torch.where(image_ch1 < 1.0, torch.zeros_like(residual0), residual0 - bound1)
            if self.num_chs == 2:
                assert torch.max(residual1) == 0, "bug"
                y = torch.cat([image_ch0 ** gamma, image_ch1 ** gamma1], dim=0)
                return y
            
            image_ch2 = torch.minimum(residual1, torch.ones_like(residual1)*bound2) / bound2
            residual2 = torch.where(image_ch2 < 1.0, torch.zeros_like(residual1), residual1 - bound2)
            assert torch.max(residual2) == 0, "bug"
            y = torch.cat([image_ch0 ** gamma, image_ch1 ** gamma1, image_ch2 ** gamma2], dim=0)
            return y

    def denormalize(self, y, raw_disp = None, mask = None):
        if self.ssi:
            assert raw_disp is not None and mask is not None
            # assert config.depth_channels == 1, "fixme"
            B, R, H, W = y.shape
            # scale-shift invariant evaluation, consider using config.safe_ssi if the ssi computation is not stable
            batch_pred = y.reshape(-1, H*W) # BR, HW
            batch_gt = raw_disp.repeat(1, R, 1, 1).reshape(-1, H*W) # BR, HW
            batch_mask = mask.repeat(1, R, 1, 1).reshape(-1, H*W)
            if self.safe_ssi:
                from utils.ransac import RANSAC
                regressor = RANSAC(n=0.1, k=10, d=0.2, t=self.ransac_error_threshold)
                regressor.fit(batch_pred, batch_gt, batch_mask)
                st = regressor.best_fit
                # logger.info(f"safe ssi in on: n=0.1, k=10, d=0.2, t={self..ransac_error_threshold}")
            else:
                # logger.info("directly compute ssi")
                st = compute_scale_and_shift(batch_pred, batch_gt, batch_mask) # BR, HW

            s, t = torch.split(st.view(B, R, 1, 2), 1, dim=-1)
            return y * s + t
        else:
            assert len(y.shape) == 4, "should be of shape B,C,H,W"
            B, C, H, W = y.shape
            R = C // self.num_chs
            y = y.view(B * R, self.num_chs, H, W)
            z = self.__denormalize(y / self.s + self.t)
            return z.view(B, R, H, W)

    def __denormalize(self, y):
        assert len(y.shape) == 4 and y.shape[1] == self.num_chs, "B,C,H,W"
        # gamma, gamma1, gamma2 = self.ch_gammas
        # bound, bound1, bound2 = self.ch_bounds

        if self.mode == "average": #Hack
            assert len(self.ch_bounds) == len(self.ch_gammas) == 1, "inconsistent params"
            gamma = self.ch_gammas[0]
            bound = self.ch_bounds[0]
            image_chs = torch.split(y, [1]*self.num_chs, dim=1)
            z = 0
            for i in range(self.num_chs):
                z += image_chs[i] ** (1/gamma) * (bound / self.num_chs)
            return z
        else:
            if self.num_chs >= 1:
                gamma = self.ch_gammas[0]
                bound = self.ch_bounds[0]
            if self.num_chs >= 2:
                gamma1 = self.ch_gammas[1]
                bound1 = self.ch_bounds[1]
            if self.num_chs >= 3:
                gamma2 = self.ch_gammas[2]
                bound2 = self.ch_bounds[2]

            image_chs = torch.split(y, [1]*self.num_chs, dim=1)
            if self.num_chs == 1:
                return image_chs[0] ** (1/gamma) * bound
            elif self.num_chs == 2:
                return image_chs[0] ** (1/gamma) * bound + image_chs[1] ** (1/gamma1) * bound1
            elif self.num_chs == 3:
                return image_chs[0] ** (1/gamma) * bound + image_chs[1] ** (1/gamma1) * bound1 + image_chs[2] ** (1/gamma2) * bound2
            else:
                raise RuntimeError("not implemented")



class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht] # left, right, top, bottom
    
    @property
    def padded_size(self):
        # hxw
        return self._pad[2] + self._pad[3] + self.ht, self._pad[0] + self._pad[1] + self.wd

    def pad(self,  *inputs):
        assert all((x.ndim == 4) for x in inputs if x is not None)
        return [F.pad(x, self._pad, mode='replicate') if x is not None else x for x in inputs ] # B,C,H,W
    
    def pad_zero(self,  *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='constant', value=0) if x is not None else x for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4, "B,C,H,W"
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    # def unpad_batch(self, x):
    #     for t in range(x.shape[-1]):
    #         out_unpad.append(self.unpad(np.expand_dims(x[...,t], axis=1)).squeeze(1))
    #     out_unpad = np.stack(out_unpad, axis=-1)
    #     return out_unpad

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)



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
        assert gt is not None
        consistent_min = np.min([np.min(img), np.min(gt)])
        consistent_max = np.max([np.max(img), np.max(gt)])
        img = np.clip(img, consistent_min, consistent_max)
        gt = np.clip(gt, consistent_min, consistent_max)
        consistent_maxmin.append([consistent_min, consistent_max])

        img = Image.fromarray((cmap_jet(img[...,0] / consistent_max)[...,:3]*255).astype(np.uint8)) 
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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def flatten(dictionary, parent_key='', separator='_'):
    """ https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys """
    from collections.abc import MutableMapping

    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            if callable(getattr(value, 'item', None)):
                value = value.item()
            items.append((new_key, value))
    return dict(items)

# https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0#file-midas_loss-py-L5
def compute_scale_and_shift(prediction, target, mask = None):
    """ B,HxW """
    prediction = prediction.clone() # make sure inputs dose not change
    target = target.clone()
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = mask.clone()

    prediction[~mask.bool()] = 0.0
    target[~mask.bool()] = 0.0

    assert prediction.shape == target.shape == mask.shape, "shape mismatch"
    B = prediction.shape[0]
    prediction = prediction.view(B, -1)
    target = target.view(B, -1)
    mask = mask.view(B, -1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1))
    a_01 = torch.sum(mask * prediction, (1))
    a_11 = torch.sum(mask, (1))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1))
    b_1 = torch.sum(mask * target, (1))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if not torch.all(det > 0.):
        print("det is zero ?!")
        det += 1e-4
    # assert torch.all(det > 0.), "det is zero ?!"

    x_0 = (a_11 * b_0 - a_01 * b_1) / det
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det

    return torch.stack([x_0, x_1]).transpose(0,1) # B,2

# https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/evaluate.py#L17
def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_errors(gt_disps, pred_disps, space="disp", masks=None, fxb=None):
    """ gt_disp: BxHxW, pred_disp: BxHxW, units should be pixels
        mask: BxHxW, valid pixels
        space: disp or depth
    """
    assert len(pred_disps.shape) == len(gt_disps.shape) and len(gt_disps.shape) == 3, "pred_disps should be BxHxW"

    valid = lambda x : ~ (np.isnan(x) | np.isinf(x) | (x == 0)) # valid number mask
    if masks is None:
        masks = valid(gt_disps)

    assert valid(gt_disps[masks]).sum() == gt_disps[masks].size, "WANING: invalid gt found"
    
    pred_disps = pred_disps.copy()
    pred_disps[~masks] = 0 # fair comparison

    # assert np.min(pred_disps[masks]) > 0.0, "you predicted zero disp values ?!"
    
    metrics_disp = []
    metrics_depth = []
    for b in range(gt_disps.shape[0]):
        gt_disp = gt_disps[b]
        pred_disp = pred_disps[b]
        mask = masks[b]

        if space == "depth":
            gt_depth = gt_disp[mask] # 1D array
            pred_depth = pred_disp[mask] # 1D array

            if np.min(pred_depth) <= 0:
                print("warning: you predicted zero depth values ?! ")
                pred_depth = np.clip(pred_depth, 1e-3, None) # pred_depth = np.clip(pred_depth, 1e-3, 10)

            gt_disp= fxb[b] / gt_depth
            pred_disp= fxb[b] / pred_depth

        elif space == "disp":
            # hack: ldm of clearpose
            # mask = mask & (pred_disp>gt_disp[mask].min()) & (pred_disp<gt_disp[mask].max())
            gt_disp = gt_disp[mask]
            pred_disp = pred_disp[mask]

            # failed_mask = None
            if np.min(pred_disp) <= 0:
                print("warning: you predicted zero disp values ?! ")
                min_disp = gt_disp.min() # prevent large mse error
                failed_mask = pred_disp <= 0
                print(f"gt_disp.min()={gt_disp.min()}, num={np.sum(failed_mask)}")
                pred_disp = np.clip(pred_disp, min_disp, None)
            
            if fxb is not None:
                gt_depth = fxb[b] / gt_disp
                pred_depth = fxb[b] / pred_disp
            else:
                gt_depth = 1 / gt_disp
                pred_depth = 1 / pred_disp

            """ # FIXME hack: set failed prediction to min_depth
            if failed_mask is not None:
                pred_depth[failed_mask] = gt_depth.min() """
            
            """ full_depth_pred = np.zeros(gt_disps.shape[1:], dtype=np.float32)
            full_depth_pred[mask] = pred_depth
            np.save(f"full_depth_pred_{b}.npy", full_depth_pred)

            full_depth_gt = np.zeros(gt_disps.shape[1:], dtype=np.float32)
            full_depth_gt[mask] = gt_depth
            np.save(f"full_depth_gt_{b}.npy", full_depth_gt) """
        else:
            raise NotImplementedError

        # compute disparity metric
        epe_err = np.abs(gt_disp - pred_disp)
        avg_epe = epe_err.mean().item()
        
        scale = gt_disp.shape[-1] / 480 # FIXME!!!!
        avg_d1 = np.mean(epe_err > 1.0 * scale).item() # * 100
        avg_d2 = np.mean(epe_err > 2.0 * scale).item() # * 100
        avg_d3 = np.mean(epe_err > 3.0 * scale).item() # * 100
        avg_d5 = np.mean(epe_err > 5.0 * scale).item() # * 100
        metrics_disp.append([avg_epe, avg_d1, avg_d2, avg_d3, avg_d5])


        # compute depth metric
        thresh = np.maximum(gt_depth / pred_depth, pred_depth / gt_depth)
        avg_a1 = np.mean(thresh < 1.05).item()
        avg_a2 = np.mean(thresh < 1.10).item()
        avg_a3 = np.mean(thresh < 1.25).item()

        rmse = np.sqrt(np.mean((gt_depth - pred_depth)**2)).item()
        rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth).item() # * 100
        mae = np.mean(np.abs(gt_depth - pred_depth)).item()

        # err = compute_depth_errors(gt_depth, pred_depths)
        # metrics_depth.append([err['a1'], err['a2'], err['a3'], 
        #                       err['rmse'], err['abs_rel'], err['log_10']])
        metrics_depth.append([avg_a1, avg_a2, avg_a3, rmse, rel, mae])
        
    
    return np.array(metrics_disp), np.array(metrics_depth)
    # return {'disp': dict(zip(["epe", "d1", "d2", "d3", "d5"], np.array(metrics_disp).mean(0))), 
    #         'depth': dict(zip(["a1", "a2", "a3", "rmse", "rel", "mae"], np.array(metrics_depth).mean(0)))}

def metrics_to_dict(metrics_disp, metrics_depth):
    return {
            'disp': dict(zip(["epe", "d1", "d2", "d3", "d5"], np.array(metrics_disp).mean(0))),
            'depth': dict(zip(["a1", "a2", "a3", "rmse", "rel", "mae"], np.array(metrics_depth).mean(0)))
            }

def normalize_rgb(*images):
    ret = []
    for image in images:
        if image is not None:
            x = (image / 255. - 0.5) * 2 # [0,1] -> [-1,1]
        else:
            x = None
        ret.append(x)
    return ret

class RunningAverager:

    def __init__(self, horizon=5, detect_anomaly=dict()):
        """ horizon: number of samples to average
            detect_anomaly: {key->ratio} if metric[key] > ratio, it is considered as anormal
        """
        self.W = horizon # windows size
        self.N = -1 # total length
        self.detect_anomaly = detect_anomaly
        self.stats_per_partition = {}

    def append(self, metrics:dict, partition="default"):
        """ metrics: {key->val}
            partition: name of partition, e.g. scene1, scene2, ..., default
            hierachical dict will be flatten before inserting into stats 
        """
        assert metrics, "metrics should not be empty"
        metrics_flatten = flatten(metrics)

        if self.N > -1:
            for k, v in metrics_flatten.items():
                if not k in self.detect_anomaly:
                    continue
                if (ratio := v / self.running_avg()[k]) > self.detect_anomaly[k]:
                    print(f"anormal data point {ratio}, abandon it")
                    return {}, ratio

        if partition != "default":
            self.update_dict(partition, metrics_flatten) # stat for each partition

        self.update_dict("default", metrics_flatten) # stat overall
        self.N += 1
        return self.running_avg(), 0

    def running_avg(self):
        assert self.N != -1, "no data"
        avg = {}
        for k, v in self.stats_per_partition["default"].items():
            avg[k] = np.mean(v[-self.W:])
        return avg

    def update_dict(self, partition, metrics:dict):
        # init empty contrainer
        if not partition in self.stats_per_partition:
            self.stats_per_partition[partition] = {}

        container = self.stats_per_partition[partition]
        if not container:
            for k, _ in metrics.items():
                container[k] = []

        assert len(container.keys()) == len(metrics.keys()), "inconsistent metrics"

        for k,v in metrics.items():
            assert k in container, f"key {k} not in container"
            container[k].append(v)

    def dump(self):
        assert self.N != -1, "no data"
        
        results = {"default": {}}
        for k, v in self.stats_per_partition["default"].items():
            results["default"][k] = np.mean(v)

        return results
        for partition in self.stats_per_partition:
            if partition == "default":
                continue

            results[partition] = {}
            for k, v in self.stats_per_partition[partition].items():
                results[partition][k] = np.mean(v)
                # print(f"{k}: {v}")

        # del results["default"]
        return results

def pretty_json(dict):
    return json.dumps(dict, indent=4)

def viz_cropped_pointcloud(K, rgb, depth, show=False, fname=None):
    """ visualize point cloud which is random cropped from the original image
        K: 3x3
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
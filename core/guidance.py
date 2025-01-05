import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from utils.camera import Realsense
from utils.ransac import RANSAC, ScaleShiftEstimator, square_error_loss, mean_absolute_error
from utils.utils import compute_scale_and_shift

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, eps=1e-7):  #height, width,
        super(Project3D, self).__init__()

        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
    
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, depth, inv_K):
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)
        # cache this thing for the second run if neccesary
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points_grad = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points_grad = torch.cat([cam_points_grad, self.ones], 1)

        return cam_points_grad
    
class FlowGuidance(ModelMixin, ConfigMixin):
    # ignore_for_config = ['device'] # should come as last when passing to init

    # @register_to_config
    def __init__(self, 
                #  camera: Realsense,
                 flow_guidance_weight=1., 
                 perturb_start_ratio=0.,
                 flow_guidance_mode="imputation"
                 ):
        super().__init__()

        # self.camera = camera

        self.flow_guidance_weight = flow_guidance_weight
        self.perturb_start_ratio = perturb_start_ratio
        self.flow_guidance_mode = flow_guidance_mode

        self.no_ssim = False #not use_ssim
        self.backproj = BackprojectDepth() #B, H, W
        self.proj = Project3D() #B, H, W
        self.ssim = SSIM()

        # self.stereo_matcher = StereoMatching(maxDisp=110, #math.ceil(camera.max_disp),
        #                                      minDisp=10, #math.floor(camera.min_disp), 
        #                                      blockSize=11, 
        #                                      eps=1e-6, 
        #                                      subPixel=True,
        #                                      bilateralFilter=False)
        
        self.guidance_image = None
        
        # hack for .to("cuda"), can be safely remove when there are trainable parameters
        # self.register_buffer("dummpy", torch.tensor(0.))

    def prepare(self, guidance, valid, image_type="depth", camera = None):
        raise DeprecationWarning
        """ noisy (incomplete) guidance image (disp): [B,1,H,W]
        """
        if type(guidance) == np.ndarray:
            guidance = torch.from_numpy(guidance)
        if type(valid) == np.ndarray:
            valid = torch.from_numpy(valid)

        valid = valid.to(bool)
        if image_type == "depth": # convert depth to disp
            valid_sm = valid # & (guidance > camera.min_depth) & (guidance < camera.max_depth)
            guidance[~valid_sm] = 0.0
            disp_sm = torch.zeros_like(guidance) #np.zeros_like(guidance)
            disp_sm[valid_sm] = guidance[valid_sm] # camera.fxb / guidance[valid_sm]

            # def normalize_disp(image, mask, ssi = True):
            #     mask = mask.to(torch.bool)
            #     """ normalize gt_image """
            #     if ssi:
            #         """ scale-shift invariant """
            #         """ normalize depth with percentitles """
            #         lowers = []
            #         uppers = []
            #         B,C = image.shape[:2]
            #         for b in range(B):
            #             valid_pixels = image[b][mask[b]]
            #             low, up = torch.quantile(valid_pixels, torch.tensor([0.00, 1.00]).to(image.device))
            #             lowers.append(low)
            #             uppers.append(up)
            #         lowers = torch.stack(lowers).reshape(B,C,1,1)
            #         uppers = torch.stack(uppers).reshape(B,C,1,1)
            #         # d2, d98 = torch.quantile(image[mask].reshape(1, -1), torch.tensor([0.00, 1.00]).to(image.device), dim=1) # 0.02, 0.98
            #         image_norm = (image - lowers) / (uppers - lowers)

            #     return (image_norm - 0.5) * 2 # [0,1] -> [-1,1]
            
            # self.disp_sm = normalize_disp(disp_sm, valid_sm, True)
            self.disp_sm = disp_sm
            self.disp_sm[~valid_sm] = 0.0
            self.valid_sm = valid_sm.to(self.device)
            self.disp_sm = self.disp_sm.to(self.device)
            return

        elif image_type == "disp":
            disp_sm = guidance
            valid_sm = valid & (guidance > camera.min_disp) & (guidance < camera.max_disp)
            disp_sm[valid_sm] = camera.normalize_disp(guidance[valid_sm])
        else:
            raise NotImplementedError
        
        self.disp_sm = (disp_sm.to(self.device) - 0.5) * 2
        self.valid_sm = valid_sm.to(torch.float32).to(self.device)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
        
    def perturb_bak(self, x_t, K, inv_K, T_rl, left_images, right_images):
        """ x_t: normalized depth image """
        assert not x_t.requires_grad, "x_t should not require grad in the sampling process"

        # assert x_t.min() >= -1. and x_t.max() <= 1., "invalid x_t"
        depth = ((x_t + 1.) / 2.) * self.max_depth # denormalize
        depth = torch.clip(depth, 0.2, self.max_depth)

        with torch.enable_grad():
            depth.requires_grad_(True)

            cam_points = self.backproj(depth, inv_K) # (B, 4, H*W)
            pix_coords = self.proj(cam_points, K, T_rl)

            warped_to_left = F.grid_sample(right_images, pix_coords, padding_mode="border", align_corners=True)
            loss = self.compute_reprojection_loss(warped_to_left, left_images)
            
            grad = torch.autograd.grad(outputs=loss.sum(), inputs=depth)[0] #loss.sum()
            return grad
        
    def perturb(self, pred_original_sample, t, decoder, denormer, raw_mask=None, left_image=None, right_image=None, raw_depth=None):
        # perturbation, g_raw, g_mask = guidance.grad(sample, None, None, t) 
        # perturbation, g_raw, g_mask = guidance.grad(pred_original_sample, None, None, t, 0.6)  # guide in x0 space
        
        if self.flow_guidance_mode == "imputation":
            raise NotImplementedError
            # perturbed_original_sample = pred_original_sample
            # perturbed_original_sample[g_mask] = g_raw[g_mask]
            perturbed_original_sample = self.disp_sm_lat # has holes in it
            
        elif self.flow_guidance_mode == "gradient":
            pred_latent = pred_original_sample.clone()
            pred_latent.requires_grad_(True)
            raw_mask = raw_depth > 0.0
            optimizer = torch.optim.Adam([pred_latent], lr=0.001)
            with torch.enable_grad():
                for _ in range(10):
                    # goes out of latent space because mask does not work in latent space
                    optimizer.zero_grad()
                    pred_depth = decoder(pred_latent)#.clamp(-1, +1)
                    pred_depth = denormer(pred_depth)
                    loss = F.l1_loss(pred_depth * raw_mask, raw_depth * raw_mask) 
                    print(f"{_}: loss={loss.item()}")
                    loss.backward()
                    optimizer.step()

            perturbed_original_sample = pred_latent.detach()
        else:
            raise NotImplementedError

        return perturbed_original_sample

    def start_stereo_match(self, left_images, right_images):
        return # hacky
        disp_sm, valid_sm = self.stereo_matcher(left_images, right_images)
        valid_sm = valid_sm & (disp_sm > self.camera.min_disp) & (disp_sm < self.camera.max_disp)
        disp_sm[~valid_sm] = 0.0

        depth_sm = self.camera.fxb / disp_sm
        def left_crop(x, margin=16):
            return x[..., margin:]
        
        margin_left = self.camera.config["margin_left"]
        disp_sm = left_crop(disp_sm, margin_left)
        valid_sm = left_crop(valid_sm, margin_left)
        depth_sm = left_crop(depth_sm, margin_left)

        self.disp_sm = (self.camera.normalize_disp(disp_sm).cuda() - 0.5) * 2
        self.valid_sm = valid_sm.cuda()
        
    def grad(self, disp_raw, left_images, right_images, timestep, th):
        if hasattr(self, "disp_sm") and hasattr(self, "valid_sm"): # a little hacky
            B,_,H,W = disp_raw.shape
            
            # regressor = RANSAC(n=0.1, k=10, d=0.2, t=th)
            # regressor.fit(disp_raw, self.disp_sm, self.valid_sm) # B,1,H,W
            # """ if regressor.best_fit is None:
            #     print("SOMETHING TERRIBLE HAPPENS!!! RELAX RANSAC")
            #     regressor = RANSAC(n=1000, d=5000, t=3, model=ScaleShiftEstimator(), loss=square_error_loss, metric=mean_absolute_error)
            #     regressor.fit(disp_raw, self.disp_sm, self.valid_sm) """

            # s, t = torch.split(regressor.best_fit.view(B, 1, 1, 2), 1, dim=-1)
            # self.d1 = (self.disp_sm - t) / s

            # raw_mask_ransac = regressor.best_mask_inlier.view(B,1,H,W)
            # # grad = self.d1 - disp_raw # in normalized disp space
            # grad = torch.sign(self.d1 - disp_raw) 
            # mask = raw_mask_ransac & self.valid_sm
            # grad *= mask # grad that can be trusted

            self.d1 = self.disp_sm_lat
            grad = self.d1 - disp_raw # torch.sign(self.d1 - disp_raw)
            mask = self.valid_sm_lat.to(torch.bool)

            # self.disp_sm: 360x640, this is guidance
            # disp_raw: 45x80, this is latents
            # with torch.enable_grad():
            #     pass

        else:
            disp_raw = disp_raw.squeeze(1)
            B, H, W = disp_raw.shape
            device = disp_raw.device

            with torch.enable_grad():
                w = 0.95
                disp_raw.requires_grad_(True)
                # disp = unnormalize_disp(disp_raw, self.min_disp, self.max_disp, self.shift)
                disp = self.camera.unnormalize_disp(disp_raw)
                
                margin_left = right_images.shape[-1] - W

                xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                xx = xx[None,...].repeat(B, 1, 1).to(device)
                yy = yy[None,...].repeat(B, 1, 1).to(device)

                xx = (xx - disp + margin_left) / ((W +  margin_left - 1) / 2.) - 1 
                yy = yy / ((H - 1) / 2.) - 1

                grid = torch.stack([xx, yy], dim=-1)
                warp_left_image = F.grid_sample(right_images, grid, align_corners=True, mode="bilinear", padding_mode="border")
                # plot_image(warp_left_image[0], f"warped.{offset}")

                # left_image = left_image[...,margin_left:]
                loss = F.l1_loss(left_images[..., margin_left:], warp_left_image, reduction='sum')
                if not self.no_ssim:
                    ssim_loss = self.ssim(left_images[..., margin_left:], warp_left_image).sum()
                    loss = w * ssim_loss + (1-w) * loss

                grads = torch.autograd.grad(loss, disp_raw, create_graph=True)[0]
                # grads = disp_raw.grad
                # grad_mask = (grads.abs() > 0).detach()
                # disp_norm = (disp_raw * grad_mask).norm()
                
                torch.nn.utils.clip_grad_norm_(disp_raw, 1)
            
            disp_raw = disp_raw.detach()
            grad = grads.unsqueeze(1) 
        
        # grad = torch.clip(grad * self.flow_guidance_weight, -10.0, 10.0) # make sense ?
        return grad, self.d1, mask

    def optimize(self, disp_raw, left_image, right_image, min_disp, max_disp, shift, alpha=1e-4, iter=100, lr=1e-2, w=0.85, clip=1e3):
        """ gradient descent on disp 
            disp: inverse depth (or just disparity?), need handle scale ambiguity?
        """

        # disp_raw = disp_raw.squeeze(1)
        with torch.enable_grad():
            disp_raw.requires_grad_(True)
            # disp_raw.register_hook(lambda grad: torch.clamp(grad, 0, clip)) #it actually hurt performance

            optimizer = torch.optim.SGD([disp_raw], lr=lr, momentum=0.9)

            # denormalize
            disp = disp_raw * (max_disp - min_disp) + min_disp + shift
            B, H, W = disp.shape
            device = disp_raw.device
            for _ in range(iter):
                
                margin_left = right_image.shape[-1] - W
                # assert B == 1, "only support batch size 1"
                xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                xx = xx.unsqueeze(0).repeat(B, 1, 1).to(device)
                yy = yy.unsqueeze(0).repeat(B, 1, 1).to(device)

                xx = (xx - disp + margin_left) / ((W +  margin_left - 1) / 2.) - 1 
                yy = yy / ((H - 1) / 2.) - 1

                grid = torch.stack([xx, yy], dim=-1)
                warp_left_image = F.grid_sample(right_image, grid, align_corners=True, mode="bilinear", padding_mode="border")
                # plot_image(warp_left_image[0], f"warped.{offset}")

                # left_image = left_image[...,margin_left:]
                loss = F.l1_loss(left_image[..., margin_left:], warp_left_image, reduction='sum')
                if not self.no_ssim:
                    ssim_loss = self.ssim(left_image[..., margin_left:], warp_left_image).sum()
                    loss = w * ssim_loss + (1-w) * loss

                # gX2 = torch.autograd.grad(loss, disp_raw, create_graph=True)[0]
                # penalty_loss = torch.max(-gX2, 0)[0].sum()
                # loss = loss + 1 * penalty_loss

                optimizer.zero_grad()
                # https://github.com/facebookresearch/PoseDiffusion/blob/1664194a2e9b021f38459aa0bd1b49b8f5045fa4/pose_diffusion/util/geometry_guided_sampling.py#L67
                loss.backward(retain_graph=True)

                grads = disp_raw.grad
                # grad_norm = grads.norm()
                grad_mask = (grads.abs() > 0).detach()
                disp_norm = (disp_raw * grad_mask).norm()

                max_norm = alpha * disp_norm / lr
                total_norm = torch.nn.utils.clip_grad_norm_(disp_raw, max_norm)
                # print(f"total_norm:{total_norm}")
                optimizer.step()

            disp_raw = disp_raw.detach()
        return disp_raw
        # return disp_raw.unsqueeze(1)

if __name__ == "__main__": # DEBUG AUTOGRAD
    B, H, W = 4, 360, 640
    device = torch.device('cuda:0')
    guidance = FlowGuidance()

    x_t = torch.randn((B,1,H,W)).to(device)

    right_images = torch.randn((B,3,H,W)).to(device)
    left_images = torch.randn((B,3,H,W)).to(device)

    fx = 446.31
    K = np.array([[fx*2, 0, W/2-0.5, 0], [0, fx*2, H/2-0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.inv(K)
    baseline = 0.055
    T_lr = np.eye(4, dtype=np.float32) # color to left ir
    T_lr[0,3] = baseline

    T_rl = np.eye(4, dtype=np.float32)
    T_rl[0,3] = -baseline
    assert np.allclose(T_rl, np.linalg.inv(T_lr)), "T_rl != inv(T_lr)" 

    grad = guidance.perturb_bak(x_t, K, inv_K, T_rl, left_images, right_images)
    print(grad.shape)
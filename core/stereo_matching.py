import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import kornia
import copy

def GetGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (torch.arange(ksize, dtype=torch.double, device=sigma.device) - center)
    kernel1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = kernel / kernel.sum()
    return kernel


def BilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    
    # calculate normalized weight matrix
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # obtain the gaussian kernel
    weights_space = GetGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # caluculate the final weight
    weights = weights_space * weights_color
    weights_sum = weights.sum(dim=(-1, -2))
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


# Left corr without zero mean
def CorrLWithoutZeroMean(i, cacheImageL, cacheImageR, filters, padding, eps):
    imageL, _, _, _, _, imageL2Sum = cacheImageL
    imageR, _, _, _, _, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedR = imageR.narrow(3, 0, W - i)                
    cropedR2Sum = imageR2Sum.narrow(3, 0, W - i)        

    shifted = F.pad(cropedR, (i, 0, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedR2Sum, (i, 0, 0, 0), "replicate")

    product = shifted * imageL
    productSum = F.conv2d(product, filters, stride=1, padding=padding)
    corrL = (productSum + eps) / (imageL2Sum.sqrt() * shifted2Sum.sqrt() + eps)
    return corrL


# Right corr without zero mean
def CorrRWithoutZeroMean(i, cacheImageL, cacheImageR, filters, padding, eps):
    imageL, _, _, _, _, imageL2Sum = cacheImageL
    imageR, _, _, _, _, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedL = imageL.narrow(3, i, W - i)  
    cropedL2Sum = imageL2Sum.narrow(3, i, W - i)  
    shifted = F.pad(cropedL, (0, i, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedL2Sum, (0, i, 0, 0), "replicate")
    product = shifted * imageR
    productSum = F.conv2d(product, filters, stride=1, padding=padding).double()
    corrR = (productSum + eps) / (imageR2Sum.sqrt() * shifted2Sum.sqrt() + eps)

    return corrR


# Left Corr
def CorrL(i, cacheImageL, cacheImageR, filters, padding, blockSize, eps):
    imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum = cacheImageL
    imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedR = imageR.narrow(3, 0, W - i)                
    cropedRSum = imageRSum.narrow(3, 0, W - i)          
    cropedR2Sum = imageR2Sum.narrow(3, 0, W - i)        
    cropedRAve = imageRAve.narrow(3, 0, W - i)          
    cropedRAve2 = imageRAve2.narrow(3, 0, W - i)        

    shifted = F.pad(cropedR, (i, 0, 0, 0), "constant", 0.0)
    shiftedSum = F.pad(cropedRSum, (i, 0, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedR2Sum, (i, 0, 0, 0), "constant", 0.0)
    shiftedAve = F.pad(cropedRAve, (i, 0, 0, 0), "constant", 0.0)
    shiftedAve2 = F.pad(cropedRAve2, (i, 0, 0, 0), "constant", 0.0)

    LShifted = imageL * shifted
    LShiftedSum = F.conv2d(LShifted, filters, stride=1, padding=padding).double()
    LAveShifted = imageLAve * shiftedSum
    shiftedAveL = shiftedAve * imageLSum
    LAveShiftedAve = imageLAve * shiftedAve
    productSum = LShiftedSum - LAveShifted - shiftedAveL + blockSize * blockSize * C * LAveShiftedAve

    sqrtL = (imageL2Sum - 2 * imageLAve * imageLSum + blockSize * blockSize * C * imageLAve2 + 1e-5).sqrt()
    sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()
    
    corrL = (productSum + eps) / (sqrtL * sqrtShifted + eps)
    corrL[:, :, :, :i] = 0

    return corrL

# Right Corr
def CorrR(i, cacheImageL, cacheImageR, filters, padding, blockSize, eps):
    imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum = cacheImageL
    imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum = cacheImageR
    B, C, H, W = imageL.shape

    cropedL = imageL.narrow(3, i, W - i)                
    cropedLSum = imageLSum.narrow(3, i, W - i)          
    cropedL2Sum = imageL2Sum.narrow(3, i, W - i)       
    cropedLAve = imageLAve.narrow(3, i, W - i)         
    cropedLAve2 = imageLAve2.narrow(3, i, W - i)       

    shifted = F.pad(cropedL, (0, i, 0, 0), "constant", 0.0)
    shiftedSum = F.pad(cropedLSum, (0, i, 0, 0), "constant", 0.0)
    shifted2Sum = F.pad(cropedL2Sum, (0, i, 0, 0), "constant", 0.0)
    shiftedAve = F.pad(cropedLAve, (0, i, 0, 0), "constant", 0.0)
    shiftedAve2 = F.pad(cropedLAve2, (0, i, 0, 0), "constant", 0.0)

    RShifted = imageR * shifted
    RShiftedSum = F.conv2d(RShifted, filters, stride=1, padding=padding).double()
    RAveShifted = imageRAve * shiftedSum
    shiftedAveR = shiftedAve * imageRSum
    RAveShiftedAve = imageRAve * shiftedAve
    productSum = RShiftedSum - RAveShifted - shiftedAveR + blockSize * blockSize * C * RAveShiftedAve

    sqrtR = (imageR2Sum - 2 * imageRAve * imageRSum + blockSize * blockSize * C * imageRAve2 + 1e-5).sqrt()
    sqrtShifted = (shifted2Sum - 2 * shiftedAve * shiftedSum + blockSize * blockSize * C * shiftedAve2 + 1e-5).sqrt()
    
    corrR = (productSum + eps) / (sqrtR * sqrtShifted + eps)
    corrR[:, :, :, W - i:] = 0

    return corrR


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, dtype=x.dtype, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, dtype=x.dtype, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    vgrid = grid + flo  # B,2,H,W

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.ones(x.size(), dtype=x.dtype, device=x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask

def LRC(dispL, dispR):
    dispRClone = dispR.clone()
    dispSamplesX = dispL.permute(0, 2, 3, 1)
    dispSamplesY = torch.zeros_like(dispSamplesX)
    dispSamples = torch.cat((-dispSamplesX, dispSamplesY), dim=-1)
    dispSamples = dispSamples.permute(0, 3, 1, 2)

    wrapedDispR = warp(dispRClone, dispSamples)
    disp = dispL.clone()
    disp[torch.pow((dispL - wrapedDispR), 2) > 0.1] = 0.0
    return disp


def LRC1(dispL, dispR):
    device = dispL.device
    B, C, H, W = dispL.shape    # C = 1
    dispSamplesX = dispL.permute(0, 2, 3, 1)

    indexX = torch.arange(0, W, 1, device=device)
    indexX = indexX.repeat(H, 1)
    indexY = torch.arange(0, H, 1, device=device)
    indexY = indexY.repeat(W, 1).transpose(0, 1)

    indexX = indexX.repeat(B, 1, 1, 1).permute(0, 2, 3, 1)
    indexY = indexY.repeat(B, 1, 1, 1).permute(0, 2, 3, 1)

    dispSamplesX = indexX - dispSamplesX

    dispSamplesY = indexY
    dispSamples = torch.cat((dispSamplesX, dispSamplesY), dim=-1)

    dispSamples[:, :, :, 0] = 2.0 * dispSamples[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
    dispSamples[:, :, :, 1] = 2.0 * dispSamples[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

    wrapedDispR = torch.nn.functional.grid_sample(dispR, dispSamples, align_corners=True)
    disp = dispL.clone()
    disp[torch.pow((dispL - wrapedDispR), 2) > 1] = 0

    return disp


# costVolume to disparity
# costVolume [D B C H W], dispVolume [D B C H W]
def CostToDisp(costVolume, dispVolume, beta, eps, subPixel):
    # sub = d + (c1 - c2)/(2*(c1 + c2 - 2*c0))
    if subPixel == True:
        D, B, C, H, W = costVolume.shape
        costVolumePad = torch.full((1, B, 1, H, W), 0, device=costVolume.device)  # padding

        dispVolume = (dispVolume + (torch.cat((costVolumePad, costVolume.narrow(0, 0, D - 1)))
                                   - torch.cat((costVolume.narrow(0, 1, D - 1), costVolumePad)) + eps) / \
                                  (2 * (torch.cat((costVolumePad, costVolume.narrow(0, 0, D - 1)))
                                   + torch.cat((costVolume.narrow(0, 1, D - 1), costVolumePad)) - 2 * costVolume) + eps))

    softmaxAttention = F.softmax(costVolume * beta, dim=0)                  # [D B C H W]
    dispVolume = (softmaxAttention * dispVolume).permute(1, 2, 3, 4, 0)     # [B C H W D]

    return torch.sum(dispVolume, 4)


def DispToDepth(dispImage, f, baselineDis, eps):
    depthImage = f * baselineDis / (dispImage + eps)
    return depthImage


def DepthToPointCloud(depthImage, f):
    B, C, H, W = depthImage.shape
    device = depthImage.device
    du = W//2 - 0.5
    dv = H//2 - 0.5

    pointCloud = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = -(torch.arange(0, W, 1, device=device) - du)
    imageIndexY = -(torch.arange(0, H, 1, device=device) - dv)
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    pointCloud[:, :, :, 0] = depthImage/f * imageIndexX
    pointCloud[:, :, :, 1] = (depthImage.transpose(1, 2)/f * imageIndexY.T).transpose(1, 2)
    pointCloud[:, :, :, 2] = depthImage
    pointCloud = pointCloud.view(B, H*W, 3)
    return pointCloud


def lerp(a,b,x):
    return a + x * (b-a)


def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h,x,y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]], dtype=np.float32)
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y


def perlin(x, y, seed=0):
    # permutation table
    if seed != None:
        np.random.seed(seed)
    p = np.arange(640, dtype=np.int32)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi = x.astype(np.int32)
    yi = y.astype(np.int32)
    # internal coordinates
    xf = (x - xi).astype(np.float32)
    yf = (y - yi).astype(np.float32)
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi], xf, yf)
    n01 = gradient(p[p[xi]+yi+1], xf, yf-1)
    n11 = gradient(p[p[xi+1]+yi+1], xf-1, yf-1)
    n10 = gradient(p[p[xi+1]+yi], xf-1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

class StereoMatching(nn.Module):
    def __init__(self, maxDisp = 60, minDisp = 1, blockSize = 9, eps = 1e-6, 
                 subPixel = True, bilateralFilter = True,
                 beta = 100., sigmaColor = 0.05, sigmaSpace = 5.
                 ):
        super(StereoMatching, self).__init__()
        self.maxDisp = maxDisp
        self.minDisp = minDisp
        self.blockSize = blockSize
        self.eps = eps
        self.subPixel = subPixel
        self.bilateralFilter = bilateralFilter

        # beta = torch.autograd.Variable(beta)
        # sigmaColor = torch.autograd.Variable(sigmaColor)
        # sigmaSpace = torch.autograd.Variable(sigmaSpace)

        # self.beta = torch.autograd.Variable(torch.tensor(beta))
        # self.sigmaColor = torch.autograd.Variable(torch.tensor(sigmaColor))
        # self.sigmaSpace = torch.autograd.Variable(torch.tensor(sigmaSpace))

        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("sigmaColor", torch.tensor(sigmaColor))
        self.register_buffer("sigmaSpace", torch.tensor(sigmaSpace))

    def forward(self, imageL, imageR ): # f, blDis
        imageL = imageL.to(self.beta.device)
        imageR = imageR.to(self.beta.device)
        
        # print("forward start")
        # beginTime = time.time()
        # imageL = imageL.type(torch.FloatTensor).cuda()
        # imageR = imageR.type(torch.FloatTensor).cuda()
        B, C, H, W = imageL.shape
        D = int(self.maxDisp - self.minDisp) + 1
        device = imageL.device

        if(self.maxDisp >= imageR.shape[3]):
            raise RuntimeError("The max disparity must be smaller than the width of input image!")

        # Normal distribution noise
        mu = 0.0
        sigma = 1.0
        mu_t = torch.full(imageL.size(), mu)
        std = torch.full(imageL.size(), sigma) 
        eps_l = torch.randn_like(std)  # normal distrubution noise
        eps_r = torch.randn_like(std)  

        if imageL.is_cuda:
            mu_t = mu_t.cuda()
            std = std.cuda()
            eps_l = eps_l.cuda()
            eps_r = eps_r.cuda()

        delta_img_receiver_l = eps_l.mul(std).add_(mu_t)
        delta_img_receiver_r = eps_r.mul(std).add_(mu_t)
        delta_img_receiver_l[delta_img_receiver_l < 0] = 0
        delta_img_receiver_r[delta_img_receiver_r > 255] = 255

        dispVolume = torch.zeros([D, B, 1, H, W], device=device)         # [B C H W D]
        costVolumeL = torch.zeros([D, B, 1, H, W], device=device)        # [B C H W D]
        costVolumeR = torch.zeros([D, B, 1, H, W], device=device)        # [B C H W D]


        filters = Variable(torch.ones(1, C, self.blockSize, self.blockSize, dtype=imageL.dtype, device=device))
        padding = (self.blockSize // 2, self.blockSize // 2)

        imageLSum = F.conv2d(imageL, filters, stride=1, padding=padding)
        imageLAve = imageLSum/(self.blockSize * self.blockSize * C)
        imageLAve2 = imageLAve.pow(2)
        imageL2 = imageL.pow(2)
        imageL2Sum = F.conv2d(imageL2, filters, stride=1, padding=padding)

        imageRSum = F.conv2d(imageR, filters, stride=1, padding=padding)
        imageRAve = imageRSum/(self.blockSize * self.blockSize * C)
        imageRAve2 = imageRAve.pow(2)
        imageR2 = imageR.pow(2)
        imageR2Sum = F.conv2d(imageR2, filters, stride=1, padding=padding)

        cacheImageL = [imageL, imageLSum, imageLAve, imageLAve2, imageL2, imageL2Sum]
        cacheImageR = [imageR, imageRSum, imageRAve, imageRAve2, imageR2, imageR2Sum]

        # calculate costVolume
        # testBeginTime = time.time()
        for i in range(self.minDisp, D + 1, 1):
            # ConcostVolume and dispVolume
            costVolumeL[i - self.minDisp] = CorrL(i, cacheImageL, cacheImageR, filters, padding, self.blockSize, self.eps)
            costVolumeR[i - self.minDisp] = CorrR(i, cacheImageL, cacheImageR, filters, padding, self.blockSize, self.eps)
            dispVolume[i - self.minDisp] = torch.full_like(costVolumeL[0], i)

        # testEndTime = time.time()
        # print("costVolume Time: ", testEndTime - testBeginTime)

        # calculate disparity map
        # testBeginTime = time.time()
        dispL = CostToDisp(costVolumeL, dispVolume, self.beta, self.eps, self.subPixel)
        dispR = CostToDisp(costVolumeR, dispVolume, self.beta, self.eps, self.subPixel)

        # testEndTime = time.time()
        # print("costToDisp Time: ", testEndTime - testBeginTime)

        # dispL[dispL < self.minDisp] = -1.0
        # dispL[dispL > self.maxDisp] = -1.0
        # dispR[dispR < self.minDisp] = -1.0
        # dispR[dispR > self.maxDisp] = -1.0

        dispLRC = LRC(dispL, dispR)
        disp = dispLRC

        if self.bilateralFilter == True:
            # disp: torch.Tensor = kornia.median_blur(disp, (5, 5))
            disp = kornia.filters.median_blur(disp, (5, 5))
            disp = BilateralFilter(disp, 7, sigmaColor=self.sigmaColor * D, sigmaSpace=self.sigmaSpace)

        # disp[disp < self.minDisp] = .0
        # disp[disp > self.maxDisp] = .0
        # valid = torch
        valid = (disp > self.minDisp) & (disp < self.maxDisp)
        disp[~valid] = 0.

        return disp, valid
        # disp = post_processing(disp, 2, 100)
        # disparity to depth
        # depthImage = DispToDepth(disp, f, blDis, self.eps)  # [B H W]

        # # depthImage[depthImage > 4000] = 0.0
        # depthImage[depthImage < 0] = 0.0
        # depthImage[depthImage > 5.0] = 0.0

        # # # depth to point cloud
        # # pointCloud = DepthToPointCloud(depthImage, f)
        # endTime = time.time()
        # print("forward finish")
        # print("forward time: ", endTime - beginTime)
        # return depthImage, None # pointCloud

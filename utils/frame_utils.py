import numpy as np
from PIL import Image
from os.path import *
import re
import json
import imageio
import os
import cv2
import torch
import torch.nn.functional as F
from scipy import interpolate

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())



def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid

# Method taken from /n/fs/raft-depth/RAFT-Stereo/datasets/SintelStereo/sdk/python/sintel_io.py
def readDispSintelStereo(file_name):
    a = np.array(Image.open(file_name))
    d_r, d_g, d_b = np.split(a, axis=2, indices_or_sections=3)
    disp = (d_r * 4 + d_g / (2**6) + d_b / (2**14))[..., 0]
    mask = np.array(Image.open(file_name.replace('disparities', 'occlusions')))
    valid = ((mask == 0) & (disp > 0))
    return disp, valid

# Method taken from https://research.nvidia.com/sites/default/files/pubs/2018-06_Falling-Things/readme_0.txt
def readDispFallingThings(file_name):
    a = np.array(Image.open(file_name))
    with open('/'.join(file_name.split('/')[:-1] + ['_camera_settings.json']), 'r') as f:
        intrinsics = json.load(f)
    fx = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
    disp = (fx * 6.0 * 100) / a.astype(np.float32)
    valid = disp > 0
    return disp, valid

# Method taken from https://github.com/castacks/tartanair_tools/blob/master/data_type.md
def readDispTartanAir(file_name):
    depth = np.load(file_name)
    disp = 80.0 / depth
    valid = disp > 0
    return disp, valid

def readDispSTD_np(filename):
    disp = np.load(filename)
    valid = (disp > 0) & ~ np.isinf(disp)
    return disp, valid

def readDispReal(camera, filename):
    """ 
    read disparity either ground truth depth or simulated disparity
    resize here aligns the file resolution with desired camera resolution 
    """
    if not os.path.exists(filename):
        # hack: prevent dataset errors
        return np.ones(camera.resolution), np.ones(camera.resolution, dtype=bool), 0, 1

    ext = splitext(filename)[-1]
    if ext == ".png":
        data = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    elif ext == ".npy":
        data = np.load(filename)
    elif ext == ".exr":
        data = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if data is None:
            print(f"bug: {filename}")
        if len(data.shape) == 3 and data.shape[-1] == 3:
            data = data[...,0]
    else:
        raise NotImplementedError
    
    scale = data.shape[1] / camera.resolution[1]
    data = cv2.resize(data, dsize=camera.resolution[::-1], interpolation=cv2.INTER_NEAREST)
    valid = ~ np.isinf(data) & ~ np.isnan(data) & (data > 0)

    if "depth" in filename or "Depth" in filename:
        # depth = camera.transform_depth_to_rgb_frame(depth) #if not alreay aligned
        disp = np.zeros_like(data, dtype=np.float32)
        # FIXME: hack 
        depth_unit = 1
        if camera.device == "fxm" or camera.device == "jav" or camera.device == "d435":
            depth_unit = 1e-3
            valid = valid & (data > 200) & (data < 3000)
            data = np.clip(data, a_min=0.0, a_max=3000) # only clip large depth values
        elif camera.device == "clearpose":
            depth_unit = 1e-3
            min_depth = camera.min_depth / depth_unit
            max_depth = camera.max_depth / depth_unit
            valid = valid & (data > min_depth ) & (data < max_depth) # [0.2~10]
            data = np.clip(data, a_min = 0.0, a_max = max_depth) # only clip large depth values

        disp[valid] = camera.fxb_depth / (data[valid]  * depth_unit)
    else:
        # disparity scales with resolution
        disp = data / scale
    
    valid = (disp > camera.min_disp) & (disp < camera.max_disp) & valid
    # disp[valid] = np.clip(disp[valid], camera.min_disp, camera.max_disp) # DEBUG: * 1.333333
    # disp[~valid] = 0.0
    return disp, valid, camera.min_disp, camera.max_disp

def readDispDreds_exr(camera, filename):
    depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(depth.shape) == 3 and depth.shape[-1] == 3:
        depth = depth [...,0]

    if depth.shape[:2] != camera.resolution:
        # be very carefull here !!! only resize in depth space
        depth = cv2.resize(depth, dsize=camera.resolution[::-1], interpolation=cv2.INTER_NEAREST) # same with DREDS

    valid = (~ (np.isinf(depth) | np.isnan(depth))) & (depth > 0.2) & (depth < 2)
    disp = np.zeros_like(depth)
    disp[valid] = camera.fxb / depth[valid]
    # disp[valid] = np.clip(disp[valid], camera.min_disp, camera.max_disp)
    return disp, valid, camera.min_disp, camera.max_disp

def readDispSTD_exr(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    valid = (~ (np.isinf(disp) | np.isnan(disp))) & (disp != 0)
    return disp, valid

def readDispSTD(file_name):
    # depth_rgb = np.load(file_name)
    gt_depth = cv2.imread(str(file_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    gt_depth = cv2.resize(gt_depth, (640*2, 360*2), interpolation=cv2.INTER_NEAREST)
    valid = ~ (np.isnan(gt_depth) | np.isinf(gt_depth))
    gt_depth[~valid] = 0

    fx = 446.31
    focal_length = fx * 2 # original ir size
    baseline = 0.055
    T_lc = np.eye(4) # color to left ir
    T_lc[0,3] = -0.015
    H, W = 360*2, 640*2
    K = np.array([[fx*2, 0, W/2-0.5], [0, fx*2, H/2-0.5], [0, 0, 1]])
    inv_K = np.linalg.inv(K)

    meshgrid = np.meshgrid(range(W), range(H), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    ones = np.ones((1, H * W), dtype=np.float32)
    pix_coords = np.concatenate((id_coords[0].reshape(1, -1), id_coords[1].reshape(1, -1), ones), axis=0)

    gt_depth = gt_depth.reshape(1, H*W)
    cam_points_ir = (inv_K @ pix_coords) * gt_depth
    valid_mask = cam_points_ir[2] > 0. # filter out invalid points

    cam_points_ir = cam_points_ir[:, valid_mask]
    cam_points_color = T_lc[:3,:3] @ cam_points_ir + T_lc[:3,3:] # convert to ir frame

    pix_coords_color = (K @ cam_points_color) # project to ir frame
    pix_coords_color[:2] /= pix_coords_color[2:3] # normalize

    ir_depth = np.zeros((H, W), dtype=np.float32)# * np.inf
    u, v = pix_coords_color[:2]
    u_left, u_right = np.floor(u).astype(np.uint32), np.ceil(u).astype(np.uint32)
    v_up, v_bottom = np.floor(v).astype(np.uint32), np.ceil(v).astype(np.uint32)

    def fill(depth_map, pred_depth, u, v):
        u, v = u.astype(np.uint32), v.astype(np.uint32)
        uv = np.vstack([u,v])
        valid_color = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
        u, v = uv[:, valid_color]
        depth_map[v, u] = pred_depth[0, valid_mask][valid_color]
    
    # an ugly HACK
    fill(ir_depth, gt_depth, u_left, v_up)
    fill(ir_depth, gt_depth, u_left, v_bottom)
    fill(ir_depth, gt_depth, u_right, v_up)
    fill(ir_depth, gt_depth, u_right, v_bottom)

    uv = np.rint(pix_coords_color).astype(np.uint32)
    valid_color = (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
    u, v = uv[:2, valid_color]
    ir_depth[v, u] = gt_depth[0, valid_mask][valid_color]

    # fill holes
    ir_depth_torch = torch.from_numpy(ir_depth).unsqueeze(0).unsqueeze(0)
    holes_mask = (ir_depth == 0) #np.isinf(ir_depth)  # exclude occ-in/occ-out?
    holes_mask[:, -20:] = False # another ugly hack exclude the right 10 cols
    holes_coords = id_coords[:2, holes_mask][(1,0),:]
    holes_coords_normal = holes_coords / np.array(([[H],[W]])) * 2 - 1
    grid = torch.from_numpy(holes_coords_normal, ).transpose(1,0).reshape(1,1,-1,2)
    interp = F.grid_sample(ir_depth_torch, grid.to(torch.float32), mode='nearest', padding_mode='zeros')
    ir_depth[holes_mask] = interp[0,0,0,:].numpy()

    disp = np.zeros_like(ir_depth)
    valid = valid & (ir_depth > 0)
    disp[valid] = focal_length * baseline / ir_depth[valid]
    
    valid = disp > 0
    return disp, valid

def readDispMiddlebury(file_name, extra_info=None): #, image_size 
    import os
    if basename(file_name) == 'disp0GT.pfm':
        disp = readPFM(file_name).astype(np.float32)
        # disp = cv2.resize(disp, image_size[::-1], cv2.INTER_NEAREST)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        # nocc_pix = cv2.resize(nocc_pix, image_size[::-1], cv2.INTER_NEAREST)
        assert np.any(nocc_pix)
        calib_file = file_name.replace('disp0GT.pfm', 'calib.txt')
        if exists(calib_file):
            calib = {}
            with open(calib_file, "r") as f:
                # read line by line
                lines = f.readlines()
                for line in lines:
                    name, var = line.partition("=")[::2]
                    if name.startswith("cam"):
                        # parse matlab mat?
                        arr = var[1:-2].split(';')
                        to_list = lambda str_arr: list(map(float, str_arr.strip().split(' ')))
                        calib[name] = [to_list(a) for a in arr]
                    else:
                        calib[name] = eval(var)

            # convert disp to depth
            depth = np.zeros_like(disp)
            depth[nocc_pix] = calib['baseline'] * calib['cam0'][0][0] / (calib['doffs'] + disp[nocc_pix]) * 1e-3 # meter

            if os.path.exists(file_name.replace("disp0GT.pfm", "im0.png_flow_pred.npy")):
                raft_disp = np.load(file_name.replace("disp0GT.pfm", "im0.png_flow_pred.npy"))
                raw_depth = calib['baseline'] * calib['cam0'][0][0] / (calib['doffs'] + -raft_disp) * 1e-3 # meter
            else:
                raw_depth = depth
            return disp, nocc_pix, depth, np.array(calib["cam0"]), raw_depth
        
        return disp, nocc_pix, np.zeros_like(disp)
        
    elif basename(file_name) == 'disp0.pfm':
        disp = readPFM(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid

def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    
def read_sceneflow(resolution, file_name, pil=False):
    """ 
    train sceneflow with different resolution
    resolution: HxW 
    """
    try:
        disp = np.array(read_gen(file_name, pil)).astype(np.float32)
    except:
        print(f"invalid ground truth file, {file_name}")
        
    assert len(disp.shape) == 2
    scale, min_disp, max_disp = 1., 0.5, 256.
    if resolution is not None and disp.shape != tuple(resolution):
        scale = disp.shape[0] / resolution[0]
        disp = cv2.resize(disp, resolution[::-1], cv2.INTER_NEAREST) #cv2.INTER_LINEAR
        disp = disp / scale
        max_disp = max_disp / scale
        min_disp = min_disp / scale
    return disp, (disp < max_disp) & (disp > min_disp), min_disp, max_disp

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext == ".npy":
        return np.load(file_name).astype(np.float32)
    elif ext == ".exr":
        return cv2.imread(file_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return []


#https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
def interpolate_missing_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = 'nearest',
    fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    assert len(image.shape) == 2, "should pass a 2D image"
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image
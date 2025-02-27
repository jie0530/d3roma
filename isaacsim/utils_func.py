import os, re, math
import numpy as np
from typing import Union, Type, List, Tuple
from pxr import Gf, Sdf, Usd, UsdGeom
from omni.isaac.core.utils.prims import get_prim_at_path
import transforms3d
import omni

def find_next_sequence_id(output_dir):
    import glob
    import os
    files = sorted(glob.glob(os.path.join(output_dir, "*.png")), reverse=True)
    if len(files) == 0: 
        return 0
    return int(files[0].split("/")[-1].split("_")[0]) + 1

def get_visibility_attribute(
    stage: Usd.Stage, prim_path: str
) -> Union[Usd.Attribute, None]:
    #Return the visibility attribute of a prim
    path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    visibility_attribute = prim.GetAttribute("visibility")
    return visibility_attribute

def get_all_child_mesh(parent_prim: Usd.Prim) -> Usd.Prim:
    # Iterates only active, loaded, defined, non-abstract children
    mesh_prims = []
    for model_prim in parent_prim.GetChildren():
        if "model" in model_prim.GetPath().pathString:
            for child_prim in model_prim.GetChildren():
                if child_prim.IsA(UsdGeom.Mesh):
                    mesh_prims.append(child_prim)
    return mesh_prims

def create_materials(self, stage, num, opacity):
    MDL = "OmniPBR.mdl"
    # MDL = "OmniGlass.mdl"
    mtl_name, _ = os.path.splitext(MDL)
    MAT_PATH = "/World/Looks"
    materials = []
    for _ in range(num):
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{MAT_PATH}/{mtl_name}", False)
        mat = self.create_omnipbr_material(mtl_url=MDL, mtl_name=mtl_name, mtl_path=prim_path, cutout_opacity=opacity)
        materials.append(mat)
    return materials

def parse_quadrant(q):
    """ x+-y+-z+-, in isaac sim hssd coordinate system """
    x_, y_, z_ = q.split(',')
    if y_[1:] == '+':
        theta = [0, np.pi/2]
    elif y_[1:] == '-':
        theta = [np.pi/2, np.pi]
    else:
        theta = [0, np.pi]

    if z_[1:] == '+':
        phi = [0, np.pi/2]
    elif z_[1:] == '-':
        phi = [np.pi/2, np.pi]
    else:
        phi = [0, np.pi]

    return theta, phi

def grasp_pose_in_robot(target_grasp, graspnet_offset = np.array([0,0,0])):
    T_table_grasp = np.eye(4)
    T_table_grasp[:3, :3] = transforms3d.quaternions.quat2mat(target_grasp['orientation'])
    T_table_grasp[:3, 3] = target_grasp['position']

    T_world_table = np.eye(4)
    # TODO random table rotation around z
    T_world_table[:3, 3] = graspnet_offset

    T_grasp_ee = np.array([
        [0,  0,  1, 0],
        [0, -1,  0, 0],
        [1,  0,  0, 0],
        [0,  0,  0, 1]
    ])

    T_robot_world = np.eye(4) # should be always be identity due to curobo limitation
    T_ee_hand = np.eye(4)
    T_ee_hand[:3, 3] = np.array([0, 0, -0.10])

    """ T_robot_hand: base_link -> panda_hand """
    T_robot_hand = T_robot_world @ T_world_table @ T_table_grasp @ T_grasp_ee @ T_ee_hand
    target_pose = {
        'position' : T_robot_hand[:3, 3],
        'orientation' : transforms3d.quaternions.mat2quat(T_robot_hand[:3, :3])
    }
    return target_pose

def compute_obb(bbox_cache: UsdGeom.BBoxCache, prim_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Oriented Bounding Box (OBB) of a prim

    .. note::

        * The OBB does not guarantee the smallest possible bounding box, it rotates and scales the default AABB.
        * The rotation matrix incorporates any scale factors applied to the object.
        * The `half_extent` values do not include these scaling effects.

    Args:
        bbox_cache (UsdGeom.BBoxCache): USD Bounding Box Cache object to use for computation
        prim_path (str): Prim path to compute OBB for

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following OBB information:
            - The centroid of the OBB as a NumPy array.
            - The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
            - The half extent of the OBB as a NumPy array.

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> # 1 stage unit length cube centered at (0.0, 0.0, 0.0)
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        >>> half_extent
        [0.5 0.5 0.5]
        >>>
        >>> # the same cube rotated 45 degrees around the z-axis
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> centroid
        [0. 0. 0.]
        >>> axes
        [[ 0.70710678  0.70710678  0.        ]
         [-0.70710678  0.70710678  0.        ]
         [ 0.          0.          1.        ]]
        >>> half_extent
        [0.5 0.5 0.5]
    """
    # Compute the BBox3d for the prim
    prim = get_prim_at_path(prim_path)
    bound = bbox_cache.ComputeWorldBound(prim)

    # Compute the translated centroid of the world bound
    centroid = bound.ComputeCentroid()

    # Compute the axis vectors of the OBB
    # NOTE: The rotation matrix incorporates the scale factors applied to the object
    rotation_matrix = bound.GetMatrix().ExtractRotationMatrix()
    x_axis = rotation_matrix.GetRow(0)
    y_axis = rotation_matrix.GetRow(1)
    z_axis = rotation_matrix.GetRow(2)

    # Compute the half-lengths of the OBB along each axis
    # NOTE the size/extent values do not include any scaling effects
    half_extent = bound.GetRange().GetSize() * 0.5

    return np.array([*centroid]), np.array([[*x_axis], [*y_axis], [*z_axis]]), np.array(half_extent)

def get_obb_corners(centroid: np.ndarray, axes: np.ndarray, half_extent: np.ndarray) -> np.ndarray:
    """Computes the corners of the Oriented Bounding Box (OBB) from the given OBB information

    Args:
        centroid (np.ndarray): The centroid of the OBB as a NumPy array.
        axes (np.ndarray): The axes of the OBB as a 2D NumPy array, where each row represents a different axis.
        half_extent (np.ndarray): The half extent of the OBB as a NumPy array.

    Returns:
        np.ndarray: NumPy array of shape (8, 3) containing each corner location of the OBB

        :math:`c_0 = (x_{min}, y_{min}, z_{min})`
        |br| :math:`c_1 = (x_{min}, y_{min}, z_{max})`
        |br| :math:`c_2 = (x_{min}, y_{max}, z_{min})`
        |br| :math:`c_3 = (x_{min}, y_{max}, z_{max})`
        |br| :math:`c_4 = (x_{max}, y_{min}, z_{min})`
        |br| :math:`c_5 = (x_{max}, y_{min}, z_{max})`
        |br| :math:`c_6 = (x_{max}, y_{max}, z_{min})`
        |br| :math:`c_7 = (x_{max}, y_{max}, z_{max})`

    Example:

    .. code-block:: python

        >>> import omni.isaac.core.utils.bounds as bounds_utils
        >>>
        >>> cache = bounds_utils.create_bbox_cache()
        >>> centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/World/Cube")
        >>> bounds_utils.get_obb_corners(centroid, axes, half_extent)
        [[-0.5 -0.5 -0.5]
         [-0.5 -0.5  0.5]
         [-0.5  0.5 -0.5]
         [-0.5  0.5  0.5]
         [ 0.5 -0.5 -0.5]
         [ 0.5 -0.5  0.5]
         [ 0.5  0.5 -0.5]
         [ 0.5  0.5  0.5]]
    """
    corners = [
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid - axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] - axes[1] * half_extent[1] + axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] - axes[2] * half_extent[2],
        centroid + axes[0] * half_extent[0] + axes[1] * half_extent[1] + axes[2] * half_extent[2],
    ]
    return np.array(corners)

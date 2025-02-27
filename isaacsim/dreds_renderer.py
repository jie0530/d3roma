import os
import math
import copy
import random
import numpy as np
import transforms3d as t3d

# rendered output setting (0: no output, 1: output)
render_mode_list = {'RGB': 1,
                    'IR': 1,
                    'NOCS': 1,
                    'Mask': 1,  
                    'Normal': 1}

# material randomization mode (transparent, specular, mixed, raw)
my_material_randomize_mode = 'mixed'

# set depth sensor parameter
camera_width = 1280
camera_height = 720
camera_fov = 71.28 / 180 * math.pi
baseline_distance = 0.055
# num_frame_per_scene = 50    # number of cameras per scene
LIGHT_EMITTER_ENERGY = 5
LIGHT_ENV_MAP_ENERGY_IR = 0.035
LIGHT_ENV_MAP_ENERGY_RGB = 0.5


# set background parameter
background_size = 3.
background_position = (0., 0., 0.)
background_scale = (1., 1., 1.)

start_point_range = ((0.5, 0.95), (-0.6, 0.6, -0.6, 0.6))
up_range = (-0.18, -0.18, -0.18, 0.18)
look_at_range = (background_position[0] - 0.05, background_position[0] + 0.05, 
            background_position[1] - 0.05, background_position[1] + 0.05,
            background_position[2] - 0.05, background_position[2] + 0.05)

g_syn_light_num_lowbound = 4
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = 0
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 3
g_syn_light_energy_std = 0.5
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1

g_shape_synset_name_pairs_all = {'02691156': 'aeroplane',
                                '02747177': 'ashtray',
                                '02773838': 'backpack',
                                '02801938': 'basket',
                                '02808440': 'tub',  # bathtub
                                '02818832': 'bed',
                                '02828884': 'bench',
                                '02834778': 'bicycle',
                                '02843684': 'mailbox', # missing in objectnet3d, birdhouse, use view distribution of mailbox
                                '02858304': 'boat',
                                '02871439': 'bookshelf',
                                '02876657': 'bottle',
                                '02880940': 'bowl', # missing in objectnet3d, bowl, use view distribution of plate
                                '02924116': 'bus',
                                '02933112': 'cabinet',
                                '02942699': 'camera',
                                '02946921': 'can',
                                '02954340': 'cap',
                                '02958343': 'car',
                                '02992529': 'cellphone',
                                '03001627': 'chair',
                                '03046257': 'clock',
                                '03085013': 'keyboard',
                                '03207941': 'dishwasher',
                                '03211117': 'tvmonitor',
                                '03261776': 'headphone',
                                '03325088': 'faucet',
                                '03337140': 'filing_cabinet',
                                '03467517': 'guitar',
                                '03513137': 'helmet',
                                '03593526': 'jar',
                                '03624134': 'knife',
                                '03636649': 'lamp',
                                '03642806': 'laptop',
                                '03691459': 'speaker',
                                '03710193': 'mailbox',
                                '03759954': 'microphone',
                                '03761084': 'microwave',
                                '03790512': 'motorbike',
                                '03797390': 'mug',  # missing in objectnet3d, mug, use view distribution of cup
                                '03928116': 'piano',
                                '03938244': 'pillow',
                                '03948459': 'rifle',  # missing in objectnet3d, pistol, use view distribution of rifle
                                '03991062': 'pot',
                                '04004475': 'printer',
                                '04074963': 'remote_control',
                                '04090263': 'rifle',
                                '04099429': 'road_pole',  # missing in objectnet3d, rocket, use view distribution of road_pole
                                '04225987': 'skateboard',
                                '04256520': 'sofa',
                                '04330267': 'stove',
                                '04379243': 'diningtable',  # use view distribution of dining_table
                                '04401088': 'telephone',
                                '04460130': 'road_pole',  # missing in objectnet3d, tower, use view distribution of road_pole
                                '04468005': 'train',
                                '04530566': 'washing_machine',
                                '04554684': 'dishwasher'}  # washer, use view distribution of dishwasher

g_synset_name_scale_pairs = {'aeroplane': [0.25, 0.31],
                        'bottle': [0.21, 0.27], 
                        'bowl': [0.15, 0.20],
                        'camera': [0.17, 0.23], 
                        'can': [0.13, 0.17],
                        'car': [0.21, 0.25], 
                        'mug': [0.13, 0.19],
                        'other': [0.13, 0.22]} 

g_synset_name_label_pairs = {'aeroplane': 7,
                        'bottle': 1,
                        'bowl': 2,   
                        'camera': 3,
                        'can': 4,
                        'car': 5,
                        'mug': 6,    
                        'other': 0} 

material_class_instance_pairs = {'specular': ['metal', 'porcelain','plasticsp','paintsp'], # ['metal', 'porcelain'],
                                'transparent': ['glass'],
                                'diffuse': ['plastic','rubber','paper','leather','wood','clay','fabric'], # ['plastic', 'rubber'],
                                'background': ['background']}

class_material_pairs = {'specular': ['bottle', 'bowl', 'can', 'mug', 'aeroplane', 'car', 'other'],
                        'transparent': ['bottle', 'bowl', 'mug'],
                        'diffuse': ['bottle', 'bowl', 'can', 'mug', 'camera', 'aeroplane', 'car', 'other']}

material_name_label_pairs = {'raw': 0,
                            'diffuse': 1,
                            'transparent': 2,
                            'specular': 3}

max_instance_num = 20

###########################
# Utils
###########################
def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r

def quaternionFromRotMat(rotation_matrix):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    quaternion = (w,x,y,z)
    return quaternion

def rotVector(q, vector_ori):
    r = quaternionToRotation(q)
    x_ori = vector_ori[0]
    y_ori = vector_ori[1]
    z_ori = vector_ori[2]
    x_rot = r[0][0] * x_ori + r[1][0] * y_ori + r[2][0] * z_ori
    y_rot = r[0][1] * x_ori + r[1][1] * y_ori + r[2][1] * z_ori
    z_rot = r[0][2] * x_ori + r[1][2] * y_ori + r[2][2] * z_ori
    return (x_rot, y_rot, z_rot)

def cameraLPosToCameraRPos(q_l, pos_l, baseline_dis):
    vector_camera_l_y = (1, 0, 0)
    vector_rot = rotVector(q_l, vector_camera_l_y)
    pos_r = (pos_l[0] + vector_rot[0] * baseline_dis,
             pos_l[1] + vector_rot[1] * baseline_dis,
             pos_l[2] + vector_rot[2] * baseline_dis)
    return pos_r

def getRTFromAToB(pointCloudA, pointCloudB):

    muA = np.mean(pointCloudA, axis=0)
    muB = np.mean(pointCloudB, axis=0)

    zeroMeanA = pointCloudA - muA
    zeroMeanB = pointCloudB - muB

    covMat = np.matmul(np.transpose(zeroMeanA), zeroMeanB)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T

def cameraPositionRandomize(start_point_range, look_at_range, up_range):
    r_range, vector_range = start_point_range
    r_min, r_max = r_range
    x_min, x_max, y_min, y_max = vector_range
    r = random.uniform(r_min, r_max)
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = math.sqrt(1 - x**2 - y**2)
    vector_camera_axis = np.array([x, y, z])

    x_min, x_max, y_min, y_max = up_range
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)    
    z = math.sqrt(1 - x**2 - y**2)
    up = np.array([x, y, z])

    x_min, x_max, y_min, y_max, z_min, z_max = look_at_range
    look_at = np.array([random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max)])
    position = look_at + r * vector_camera_axis

    vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
    vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
    vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))
    
    Rwc = np.vstack([vectorX, vectorY, vectorZ]).T
    q = t3d.quaternions.mat2quat(Rwc)
    t = position

    # points in camera coordinates
    pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

    # points in world coordinates 
    pointWorld = np.array([position,
                            position + vectorX,
                            position + vectorY * 2,
                            position + vectorZ * 3])

    resR, resT = getRTFromAToB(pointSensor, pointWorld)
    resQ = quaternionFromRotMat(resR)

    if resQ[0] < 0:
        resQ = (-resQ[0], -resQ[1], -resQ[2], -resQ[3])

    assert np.allclose(q, resQ) and np.allclose(t, resT[:3,0])
    return resQ, resT     # wxyz

def quanternion_mul(q1, q2):
    s1 = q1[0]
    v1 = np.array(q1[1:])
    s2 = q2[0]
    v2 = np.array(q2[1:])
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return (s, v[0], v[1], v[2])

def generate_CAD_model_list(model_path, g_shape_synset_name_pairs):
    CAD_model_list = {}
    for class_folder in os.listdir(model_path):
        if class_folder[0] == '.':
            continue
        class_path = os.path.join(model_path, class_folder)
        class_name = g_shape_synset_name_pairs[class_folder] if class_folder in g_shape_synset_name_pairs else 'other'
        class_list = []
        for instance_folder in os.listdir(class_path):
            if instance_folder[0] == '.':
                continue

            hackfile = f"{instance_folder}_converted" if "_converted" not in instance_folder else instance_folder
            instance_path = os.path.join(class_path, hackfile, "model_obj.usd")
            class_list.append([instance_path, class_name])
        if class_name == 'other' and 'other' in CAD_model_list:
            CAD_model_list[class_name] = CAD_model_list[class_name] + class_list
        else:
            CAD_model_list[class_name] = class_list

    return CAD_model_list

def generate_material_type(obj_name):
    flag = random.randint(0, 3)
    # select the raw material
    if flag == 0:
        flag = random.randint(0, 1)
        if flag == 0:
            return 'raw'
        else:            
            if obj_name.split('_')[1] in class_material_pairs['transparent']:
                return 'diffuse'                     
    # select one from specular and transparent
    else:
        flag = random.randint(0, 2)
        if flag < 2:
            if obj_name.split('_')[1] in class_material_pairs['transparent']:
                return 'transparent'
            else:
                flag = 2

        if flag == 2:
            if obj_name.split('_')[1] in class_material_pairs['specular']:
                return 'specular'  
            else:
                return 'diffuse'
    return 'raw'

from omni.isaac.core.utils import prims

class DredsRenderer:
    def __init__(self, working_root):
        self.CAD_model_root_path = os.path.join(working_root, "cad_model")                           # CAD model path
        env_map_path = os.path.join(working_root, "envmap_lib")   
        self.output_root_path = os.path.join(working_root, "output") 

        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)

    def set_material_randomize_mode(self, class_material_pairs, mat_randomize_mode, instance_name, material_type_in_mixed_mode):
        if mat_randomize_mode == 'transparent' and instance_name.split('_')[1] in class_material_pairs['transparent']:
            # print(instance_name, 'material mode: transparent')            
            return 'transparent'
            # instance.data.materials.clear()
            # instance.active_material = random.sample(self.my_material['transparent'], 1)[0]

        elif mat_randomize_mode == 'specular' and instance_name.split('_')[1] in class_material_pairs['specular']:
            # print(instance_name, 'material mode: specular')
            return 'transparent'
            # material = random.sample(self.my_material['specular'], 1)[0]
            # set_modify_material(instance, material)     

        elif mat_randomize_mode == 'mixed':
            if material_type_in_mixed_mode == 'diffuse' and instance_name.split('_')[1] in class_material_pairs['diffuse']:
                # print(instance_name, 'material mode: diffuse')
                return 'diffuse'
                # material = random.sample(self.my_material['diffuse'], 1)[0]
                # set_modify_material(instance, material)
            elif material_type_in_mixed_mode == 'transparent' and instance_name.split('_')[1] in class_material_pairs['transparent']:
                # print(instance_name, 'material mode: transparent')
                return 'transparent'
                # instance.data.materials.clear()
                # instance.active_material = random.sample(self.my_material['transparent'], 1)[0]
            elif material_type_in_mixed_mode == 'specular' and instance_name.split('_')[1] in class_material_pairs['specular']:
                # print(instance_name, 'material mode: specular')
                return 'specular'
                # material = random.sample(self.my_material['specular'], 1)[0]
                # set_modify_material(instance, material)       
            else:
                return 'raw'
                # print(instance.name, 'material mode: raw')
                # set_modify_raw_material(instance)
        else:
            # print(instance_name, 'material mode: raw')
            return 'raw'
            # set_modify_raw_material(instance)

    def domain_randomize(self, num_frame_per_scene):
        # TODO should be configurable
        selected_class = ['aeroplane', 'bottle', 'bowl', 'camera', 'can', 'car', 'mug']
        g_shape_synset_name_pairs = copy.deepcopy(g_shape_synset_name_pairs_all)
        g_shape_synset_name_pairs['00000000'] = 'other'

        for item in g_shape_synset_name_pairs_all:
            if not g_shape_synset_name_pairs_all[item] in selected_class:
                g_shape_synset_name_pairs[item] = 'other'

         # generate CAD model list
        self.CAD_model_list = generate_CAD_model_list(self.CAD_model_root_path, g_shape_synset_name_pairs)

        # self.loadImages(env_map_path)
        # self.addEnvMap()
        # self.addBackground(background_size, background_position, background_scale)
        # self.addMaterialLib()
        # self.addMaskMaterial(max_instance_num)
        # self.addNOCSMaterial()
        # self.addNormalMaterial()
        # self.clearModel()

        # camera pose list, environment light list and background material_listz
        quaternion_list = []
        translation_list = []
        for i in range(num_frame_per_scene):
            # generate camara pose list
            quaternion, translation = cameraPositionRandomize(start_point_range, look_at_range, up_range)
            translation_list.append(translation[:,0]) 
            quaternion_list.append(quaternion)

        # read objects from floder
        instance_id = 1
        meta_output = {}
        #select_model_list = []
        select_model_list_other = []
        select_model_list_transparent = []
        select_model_list_dis = []
        select_number = 1

        for item in self.CAD_model_list:
            if item in ['bottle', 'bowl', 'mug']:
                test = random.sample(self.CAD_model_list[item], select_number)
                for model in test:
                    select_model_list_transparent.append(model)
            elif item in ['other']:
                test = random.sample(self.CAD_model_list[item], min(3, len(self.CAD_model_list[item])))
                for model in test:
                    select_model_list_dis.append(model)
            else:
                test = random.sample(self.CAD_model_list[item], select_number)
                for model in test:
                    select_model_list_other.append(model)

        select_model_list_other = random.sample(select_model_list_other, random.randint(1, 4))
        select_model_list_dis = random.sample(select_model_list_dis, random.randint(1, 3))
        select_model_list = select_model_list_transparent + select_model_list_other + select_model_list_dis

        dred_models = []
        for model in select_model_list:
            instance_path = model[0]
            class_name = model[1]
            class_folder = model[0].split('/')[-3]
            instance_folder = model[0].split('/')[-2]
            instance_name = str(instance_id) + "_" + class_name + "_" + class_folder + "_" + instance_folder
            prim_path = f"/World/model_{instance_id}_{class_name}"
            prim_name = f"model_{instance_id}_{class_name}"
            material_type_in_mixed_mode = generate_material_type(instance_name)
            material_type = self.set_material_randomize_mode(class_material_pairs, my_material_randomize_mode, instance_name, material_type_in_mixed_mode)

            class_scale = random.uniform(g_synset_name_scale_pairs[class_name][0], g_synset_name_scale_pairs[class_name][1])

            dred_models.append({
                'instance_id': instance_id,
                'instance_name': instance_name,
                'instance_path': instance_path,
                'class_name': class_name,
                'instance_folder': instance_folder,
                'material_type': material_type,
                'scale': class_scale
            })
            instance_id += 1

        return dred_models, quaternion_list, translation_list
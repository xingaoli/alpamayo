import argparse
import pickle
import numpy as np
from os import path as osp
from pyquaternion import Quaternion

NameMapping = {
    # =================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',

    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    # ===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    # =================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

CLASSES = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'
]

def parse_args():
    parser = argparse.ArgumentParser(description='format gt for visualization')
    parser.add_argument('--gt_path', help='inference result file path')
    parser.add_argument('--save_path', help='the dir to save visualization results')
    args = parser.parse_args()

    return args

def invert_pose(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    return inv_pose

def get_ann_info(index):
    """Get annotation info according to the given index.

    Args:
        index (int): Index of the annotation data to get.

    Returns:
        dict: Annotation information consists of the following keys:

            - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                3D ground truth bboxes
            - gt_labels_3d (np.ndarray): Labels of ground truths.
            - gt_names (list[str]): Class names of ground truths.
    """
    info = data_infos[index]

    for i in range(len(info['gt_names'])):
        if info['gt_names'][i] in NameMapping.keys():
            info['gt_names'][i] = NameMapping[info['gt_names'][i]]

    # filter out bbox containing no points
    mask = (info['num_points'] != 0)

    gt_bboxes_3d_wo_id = info['gt_boxes']
    cur_frame_agent_id = info['gt_ids']
    gt_bboxes_3d = np.concatenate([gt_bboxes_3d_wo_id, cur_frame_agent_id.reshape(-1, 1)], axis=1)
    gt_bboxes_3d = gt_bboxes_3d[mask]
    gt_names_3d = info['gt_names'][mask]

    attr_labels = get_box_attr_labels(index,sample_interval,future_frames)
    attr_labels = attr_labels[mask]
    gt_fut_trajs = attr_labels[:, :future_frames*2].reshape(-1, future_frames, 2)
    gt_fut_masks = attr_labels[:, future_frames*2:future_frames*3]
    anns_results = dict(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_names=gt_names_3d,
        gt_fut_trajs = gt_fut_trajs,
        gt_fut_masks = gt_fut_masks,
        attr_labels = attr_labels
        )
    return anns_results

def get_box_attr_labels(idx,sample_rate,frames):
    adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
    cur_frame = data_infos[idx]
    cur_box_names = cur_frame['gt_names']
    cur_boxes = cur_frame['gt_boxes'].copy()
    box_ids = cur_frame['gt_ids']
    future_track = np.zeros((len(box_ids),frames+1,2))
    future_mask = np.zeros((len(box_ids),frames+1))
    future_yaw = np.zeros((len(box_ids),frames+1))
    gt_fut_goal = np.zeros((len(box_ids),1))
    agent_lcf_feat = np.zeros((len(box_ids),9))
    world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
    for i in range(len(box_ids)):
        agent_lcf_feat[i,0:2] = cur_boxes[i,0:2] # 自车相对于lidar坐标系的偏移
        agent_lcf_feat[i,2] = cur_boxes[i,6] # 自车相对于lidar坐标系的夹角
        agent_lcf_feat[i,3:5] = cur_boxes[i,7:] # lidar坐标系下的速度
        agent_lcf_feat[i,5:8] = cur_boxes[i,3:6] # w l h
        cur_box_name = cur_box_names[i]
        if cur_box_name in CLASSES:
            agent_lcf_feat[i, 8] = CLASSES.index(cur_box_name) # class类别id
        else:
            agent_lcf_feat[i, 8] = -1
        # 通过以上可以推测出：
        # 拿到的bbox的9维分别是：相对于lidar坐标系的偏移量x, y, z, w, l, h, yaw, vx, vy

        box_id = box_ids[i]
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(data_infos):
                break
            adj_frame = data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                break
            if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                continue
            assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
            adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
            adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
            adj_xy = adj_box2lidar[0:2,3]
            future_track[i,j,:] = adj_xy
            future_mask[i,j] = 1
            future_yaw[i,j] = np.arctan2(adj_box2lidar[1,0],adj_box2lidar[0,0])

        coord_diff = future_track[i,-1] - future_track[i,0]
        if coord_diff.max() < 1.0: # static
            gt_fut_goal[i] = 9
        else:
            box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
            gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class

    future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
    future_mask_offset = future_mask[:,1:]
    future_track_offset[future_mask_offset==0] = 0
    future_yaw_offset = future_yaw[:,1:] - future_yaw[:,:-1]
    mask1 = np.where(future_yaw_offset>np.pi)
    mask2 = np.where(future_yaw_offset<-np.pi)
    future_yaw_offset[mask1] -=np.pi*2
    future_yaw_offset[mask2] +=np.pi*2
    attr_labels = np.concatenate([future_track_offset.reshape(-1,frames*2), future_mask_offset, gt_fut_goal, agent_lcf_feat, future_yaw_offset],axis=-1).astype(np.float32)
    return attr_labels.copy()

# 直接转为AR1需要的自车坐标系下的轨迹形式，并带旋转矩阵
def get_ego_trajs(idx, sample_rate, past_frames, future_frames):

    adj_idx_list = range(idx - past_frames * sample_rate, idx + (future_frames + 1) * sample_rate, sample_rate)
    cur_frame = data_infos[idx]
    full_adj_track = np.zeros((past_frames + future_frames + 1, 2))
    full_adj_rot = np.zeros((past_frames + future_frames + 1, 3, 3))
    full_adj_adj_mask = np.zeros(past_frames + future_frames + 1)
    # world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
    world2ego_ego_cur = cur_frame['world2ego']

    for j in range(len(adj_idx_list)):
        adj_idx = adj_idx_list[j]
        if adj_idx < 0 or adj_idx >= len(data_infos):
            break
        adj_frame = data_infos[adj_idx]
        if adj_frame['folder'] != cur_frame['folder']:
            break
        
        ego2world_ego_adj = invert_pose(adj_frame['world2ego'])
        adj2cur_ego = world2ego_ego_cur @ ego2world_ego_adj

        rot = adj2cur_ego[:3, :3]
        xy = adj2cur_ego[0:2, 3]
        
        # world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
        # adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
        # xy = adj2cur_lidar[0:2, 3]
        full_adj_rot[j] = rot
        full_adj_track[j, 0:2] = xy
        full_adj_adj_mask[j] = 1

    # offset_track = full_adj_track[1:] - full_adj_track[:-1]

    # for j in range(past_frames - 1, -1, -1):
    #     if full_adj_adj_mask[j] == 0:
    #         offset_track[j] = offset_track[j + 1]

    # for j in range(past_frames, past_frames + future_frames, 1):
    #     if full_adj_adj_mask[j + 1] == 0:
    #         offset_track[j] = 0

    for j in range(past_frames - 1, -1, -1):
        if full_adj_adj_mask[j] == 0:
            full_adj_track[j] = full_adj_track[j + 1]
            full_adj_rot[j] = full_adj_rot[j + 1]

    for j in range(past_frames, past_frames + future_frames, 1):
        if full_adj_adj_mask[j + 1] == 0:
            full_adj_track[j + 1] = full_adj_track[j]
            full_adj_rot[j + 1] = full_adj_rot[j]

    command = command2hot(cur_frame['command_near'])
    return full_adj_track[:past_frames+1].copy(), full_adj_track[past_frames+1:].copy(), full_adj_rot[:past_frames+1].copy(), full_adj_rot[past_frames+1:].copy(), full_adj_adj_mask[-future_frames:].copy(), command


def command2hot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot

def filter_useless_key(data_infos_list):
    data_infos_list_filter = []
    for index, value in enumerate(data_infos_list):
        for k_1, v_1 in value.items():
            if k_1 == 'sensors':
                for k_2, v_2 in v_1.items():
                    if k_2 != 'LIDAR_TOP':
                        # del v_1['intrinsic']
                        del v_2['world2cam']
                    else:
                        v_2['map_pts_world2lidar'] = v_2['world2lidar']
                        del v_2['world2lidar']

        need_keys = ['scene_token', 'sample_token', 'frame_idx', 'ego2global_translation',
                     'ego2global_rotation', 'sensors', 'timestamp', 'gt_boxes_anno_info',
                     'can_bus', 'ego_his_trajs', 'ego_fut_trajs', 'ego_his_rots', 'ego_fut_rots', 'ego_fut_masks', 'ego_fut_cmd',
                     'ego_lcf_feat', 'fut_valid_flag', 'map_location', 'ego_fut_cmd_name', 'num_points']

        data_infos_list_filter.append({k: v for k, v in value.items() if k in need_keys})

    return data_infos_list_filter

def get_items(index):
    info = data_infos[index]
    scene_token = info['folder']
    frame_idx = info['frame_idx']
    sample_token = f'{scene_token}_{frame_idx}'

    ego2global = invert_pose(info['world2ego'])

    input_dict = dict(
        scene_token=scene_token,
        sample_token=sample_token,
        frame_idx=frame_idx,
        # 已经转到东北天坐标系下了，相当于跟正东方向的夹角
        ego_yaw=np.nan_to_num(info['ego_yaw'], nan=np.pi / 2),
        ego2global_translation=ego2global[:3, 3],
        ego2global_rotation=ego2global[:3, :3],
        sensors=info['sensors'],
        gt_ids=info['gt_ids'],
        gt_boxes=info['gt_boxes'],
        gt_names=info['gt_names'],
        ego_vel=info['ego_vel'],
        ego_accel=info['ego_accel'],
        ego_rotation_rate=info['ego_rotation_rate'],
        npc2world=info['npc2world'],
        timestamp=info['frame_idx'] / 10,
        map_location=info['town_name'],
        num_points=info['num_points']
    )

    lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
    lidar2global = invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
    for sensor_type, cam_info in info['sensors'].items():
        if 'LIDAR' in sensor_type:
            continue
        # obtain lidar to image transformation matrix
        cam2ego = cam_info['cam2ego']
        intrinsic = cam_info['intrinsic']
        intrinsic_pad = np.eye(4)
        intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2cam = invert_pose(cam2ego) @ lidar2ego
        lidar2img = intrinsic_pad @ lidar2cam
        info['sensors'][sensor_type].update({'lidar2cam': lidar2cam})
        info['sensors'][sensor_type].update({'lidar2img': lidar2img})
        # info['sensors'][sensor_type].update({'cam_intrinsics_pad': intrinsic_pad})

    annos = get_ann_info(index)
    input_dict['gt_boxes_anno_info'] = annos
    yaw = input_dict['ego_yaw']
    rotation = Quaternion(axis=[0, 0, 1], radians=yaw)

    if yaw < 0:
        yaw += 2 * np.pi
    yaw_in_degree = yaw / np.pi * 180
    can_bus = np.zeros(18)
    can_bus[:3] = input_dict['ego2global_translation']  # 自车相对global位移
    can_bus[3:7] = list(rotation)  # 自车相对global旋转
    can_bus[7:10] = input_dict['ego_vel']  # 自车线性速度，只有第一维有值，vad中这三维在[13:16]
    can_bus[10:13] = input_dict['ego_accel']  # 自车速度，IMU坐标系三个加速度分量
    can_bus[13:16] = input_dict['ego_rotation_rate']  # 角速度，yaw速度就是与正东方向夹角的变化率
    can_bus[16] = yaw  # 自车车头与正东方向的夹角
    can_bus[17] = yaw_in_degree  # 自车车头与正东方向的夹角，弧度制
    input_dict['can_bus'] = can_bus
    ego_lcf_feat = np.zeros(9)
    #################################################
    '''fix bug'''
    ego_lcf_feat[0:2] = input_dict['ego2global_translation'][0:2]  # vad是自车速度分解到正北、正东两个方向
    ####################################################
    ego_lcf_feat[2:4] = input_dict['ego_accel'][0:2]  # 自车速度，IMU坐标系三个加速度分量
    ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]  # yaw角的角速度
    ego_lcf_feat[5] = info['ego_size'][1]  # length
    ego_lcf_feat[6] = info['ego_size'][0]  # width
    ##################################################
    '''fix bug'''
    # 这里也错了，原始vad里是自车的线性速度
    ego_lcf_feat[7] = np.sqrt(
        input_dict['ego2global_translation'][0] ** 2 + input_dict['ego2global_translation'][1] ** 2)
    #####################################################
    ego_lcf_feat[8] = info['steer']  # 轮速
    ego_his_trajs, ego_fut_trajs, ego_his_rots, ego_fut_rots, ego_fut_masks, command = get_ego_trajs(index, sample_interval,
                                                                         past_frames, future_frames)

    input_dict['ego_his_trajs'] = ego_his_trajs
    input_dict['ego_fut_trajs'] = ego_fut_trajs
    input_dict['ego_his_rots'] = ego_his_rots
    input_dict['ego_fut_rots'] = ego_fut_rots
    input_dict['ego_fut_masks'] = ego_fut_masks
    input_dict['ego_fut_cmd'] = command
    input_dict['ego_lcf_feat'] = ego_lcf_feat
    input_dict['fut_valid_flag'] = (ego_fut_masks == 1).all()

    cmd_list = ['Turn Left', 'Turn Right', 'Go Straight', 'LaneFollow', 'ChangeLaneLeft', 'ChangeLaneRight']
    input_dict['ego_fut_cmd_name'] = cmd_list[np.argmax(command)]

    return input_dict

if __name__ == '__main__':
    args = parse_args()

    sample_interval = 1
    past_frames = 16
    future_frames = 64
    data_infos = pickle.load(open(args.gt_path, 'rb'))

    new_data_infos_list = []
    for index, v in enumerate(data_infos):
        new_data_infos_list.append(get_items(index))

    new_data_infos_list = filter_useless_key(new_data_infos_list)
    pickle.dump(new_data_infos_list, open(args.save_path, 'wb'))

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.chdir("..")
print(os.path.abspath(os.curdir))
import copy
import numpy as np
import mediapy as mp
import pandas as pd
import torch
import pickle
import PIL.Image as Image
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from einops import rearrange
import huggingface_hub
from unittest.mock import Mock
import os
from tqdm import tqdm

def load_custom_b2d_datasets(scenario_name, t0_us=16):
    data_dir = 'data/bench2drive_full/bench2drive'
    label_dir = 'output/format_data/v1'
    labels = pickle.load(open(os.path.join(label_dir, scenario_name, 'b2d_infos_custom.pkl'), 'rb'))
    labels_list = list(labels)

    camera_features = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    image_tensors = []
    for cam_feature in camera_features:
        for past_image_idx in range(t0_us-3, t0_us+1):
            image_dir = labels_list[past_image_idx]['sensors'][cam_feature]['data_path']
            image = np.array(Image.open(os.path.join(data_dir, image_dir)))
            image_tensor = torch.from_numpy(image)
            image_tensors.append(image_tensor)
    image_tensors = torch.stack(image_tensors, dim=0).reshape(len(camera_features), 4, image.shape[0], image.shape[1], 3)
    image_tensors = rearrange(image_tensors, 'n t h w c -> n t c h w')

    cur_label = labels_list[t0_us]
    ego_history_xy_tensor = torch.as_tensor(cur_label['ego_his_trajs'])[:16]
    ego_history_xyz_tensor = torch.cat([ego_history_xy_tensor, torch.zeros(16, 1)], dim=1).reshape(1, 1, 16, 3)
    ego_history_rot_tensor = torch.as_tensor(cur_label['ego_his_rots'])[:16].reshape(1, 1, 16, 3, 3)
    ego_future_xy_tensor = torch.as_tensor(cur_label['ego_fut_trajs'])
    ego_future_xyz_tensor = torch.cat([ego_future_xy_tensor, torch.zeros(64, 1)], dim=1).reshape(1, 1, 64, 3)
    ego_future_rot_tensor = torch.as_tensor(cur_label['ego_fut_rots']).reshape(1, 1, 64, 3, 3)

    return {
        "image_frames": image_tensors,  # (N_cameras, num_frames, 3, H, W)
        "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
        "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
        "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
        "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
        "t0_us": t0_us,
        "scenario_name": scenario_name,
    }

model = AlpamayoR1.from_pretrained("ckpts/Alpamayo-R1-10B/", dtype=torch.bfloat16).to("cuda")
model.eval()

processor = helper.get_processor(model.tokenizer)

scenario_name = "ParkedObstacle_Town10HD_Route371_Weather7"
data_dir = 'data/bench2drive_full/bench2drive'
label_dir = 'output/format_data/v1'
labels = pickle.load(open(os.path.join(label_dir, scenario_name, 'b2d_infos_custom.pkl'), 'rb'))

results = []

for i in tqdm(range(len(labels))):
    
    data = load_custom_b2d_datasets(scenario_name, i)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print("seq length:", inputs.input_ids.shape)
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=2,  # Feel free to raise this for more output trajectories and CoC traces.
            max_generation_length=256,
            return_extra=True,
        )

    # the size is [batch_size, num_traj_sets, num_traj_samples]
    print("Chain-of-Causation (per trajectory):\n", extra)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    coc = extra['cot'][0, 0].tolist()
    label = labels[i]
    # convert trajs to vis format
    pred_xy = np.concat([pred_xy, np.ones((pred_xy.shape[0], pred_xy.shape[1], 2))], axis=2) # 2, 64, 4
    num_trajs, fut_point = pred_xy.shape[0], pred_xy.shape[1]
    pred_xy = pred_xy.reshape(num_trajs*fut_point, -1)
    lidar2ego = label['sensors']['LIDAR_TOP']['lidar2ego']
    pred_xy_lidar = np.linalg.inv(lidar2ego) @ pred_xy.T
    pred_xy_lidar = pred_xy_lidar.T.reshape(num_trajs, fut_point, -1)[:, :, :2]
    pred_xy_lidar = np.concat([np.zeros((num_trajs, 1, 2)), pred_xy_lidar], axis=1)
    pred_xy_lidar_offset = pred_xy_lidar[:, 1:, :] - pred_xy_lidar[:, :-1, :]

    gt_xy = label['ego_fut_trajs']
    gt_xy = np.concat([gt_xy, np.ones((gt_xy.shape[0], 2))], axis=1)
    gt_xy_lidar = np.linalg.inv(lidar2ego) @ gt_xy.T
    gt_xy_lidar = gt_xy_lidar.T[:, :2]
    gt_xy_lidar = np.concat([np.zeros((1, 2)), gt_xy_lidar], axis=0)
    gt_xy_lidar_offset = gt_xy_lidar[1:, :] - gt_xy_lidar[:-1, :]
    labels[i]['ego_fut_trajs'] = gt_xy_lidar_offset
    labels[i]['pred_ego_fut_trajs'] = pred_xy_lidar_offset
    labels[i]['text'] = coc
data = pickle.load(open("output/format_data/v1/ParkedObstacle_Town10HD_Route371_Weather7/b2d_infos_val.pkl", 'rb'))
pickle.dump(labels, open(os.path.join(label_dir, scenario_name, 'b2d_infos_vis.pkl'), 'wb'))
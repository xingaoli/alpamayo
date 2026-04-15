import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.chdir("..")
print(os.path.abspath(os.curdir))
import copy
import numpy as np
import mediapy as mp
import pandas as pd
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset_local
from alpamayo_r1 import helper

import huggingface_hub
from unittest.mock import Mock
import os

model = AlpamayoR1.from_pretrained("ckpts/Alpamayo-R1-10B/", dtype=torch.bfloat16).to("cuda")
model.eval()
processor = helper.get_processor(model.tokenizer)
clip_id = "5c8a7587-d850-474c-b297-7a633d0538d1"
data = load_physical_aiavdataset_local(clip_id)

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

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
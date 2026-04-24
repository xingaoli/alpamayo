# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any

import torch
from alpamayo_r1.load_physical_aiavdataset_local import load_physical_aiavdataset_local
from alpamayo_r1.common import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class COCDataset(Dataset):
    """Dataset that loads COC reasoning text and joins with PAI trajectory data.

    Each sample in the COC JSONL has a clip_id and frame_idx. The frame_idx is
    in units of 0.1s per frame, so t0_us = frame_idx * 100_000 (e.g. frame_idx=52
    -> 5.2s -> 5_200_000 us). The CoC text is injected as the "cot" field.
    """

    def __init__(
        self,
        coc_jsonl_path: str | None = None,
        local_dir: str | None = None,
        include_extr_intr: bool = False,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        clip_index_metadata: str = "clip_index.parquet",
    ):
        if coc_jsonl_path is None:
            coc_jsonl_path = os.environ.get(
                "ALPAMAYO_COC_JSONL", ""
            )
        if local_dir is None:
            local_dir = os.environ.get(
                "ALPAMAYO_DATA_DIR", ""
            )

        assert coc_jsonl_path and os.path.exists(coc_jsonl_path), (
            f"COC JSONL not found: {coc_jsonl_path}. "
            f"Set ALPAMAYO_COC_JSONL env var or pass coc_jsonl_path."
        )
        assert local_dir and os.path.exists(local_dir), (
            f"Data dir not found: {local_dir}. "
            f"Set ALPAMAYO_DATA_DIR env var or pass local_dir."
        )

        # Load the full JSONL into memory as a flat list of (clip_id, frame_idx, text)
        logger.info(f"Loading COC annotations from {coc_jsonl_path} ...")
        self.samples: list[dict[str, Any]] = []
        with open(coc_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.samples.append({
                    "clip_id": row["clip_id"],
                    "frame_idx": row["frame_idx"],
                    "cot": row["text"],
                    "long_action": row.get("long_action", ""),
                    "lat_action": row.get("lat_action", ""),
                })
        logger.info(f"Loaded {len(self.samples)} COC samples")

        self.include_extr_intr = include_extr_intr
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.local_dir = local_dir
        self.clip_index_metadata = clip_index_metadata

        # Check that t0 fits within history requirements
        future_range_us = int(num_future_steps * time_step * 1_000_000)
        min_t0_us = int(num_history_steps * time_step * 1_000_000) + 1
        valid_indices = []
        dropped = 0
        for i, s in enumerate(self.samples):
            t0_us = s["frame_idx"] * 100_000
            max_future = 20_000_000 - t0_us  # clip max ~20s
            if t0_us > min_t0_us and max_future >= future_range_us:
                valid_indices.append(i)
            else:
                dropped += 1
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} COC samples whose frame_idx is out of "
                f"trajectory range (valid: {len(valid_indices)})"
            )
        self.valid_indices = valid_indices

        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.samples[self.valid_indices[idx]]
        clip_id = sample["clip_id"]
        t0_us = sample["frame_idx"] * 100_000

        sample_data = load_physical_aiavdataset_local(
            clip_id,
            t0_us=t0_us,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.include_extr_intr:
            from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface
            if not hasattr(self, "_avdi"):
                self._avdi = PhysicalAIAVDatasetLocalInterface(
                    local_dir=self.local_dir,
                    clip_index_metadata=self.clip_index_metadata,
                )
            sample_data["extr"] = self._avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample_data["intr"] = self._avdi.get_clip_feature(clip_id, "camera_intrinsics")

            vehicle_dimensions = self._avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            sample_data["ego_lwh"] = torch.tensor(
                [vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]
            )
            sample_data["ego_length_offset"] = torch.tensor(
                vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length
            )

        # Inject CoC reasoning text
        sample_data["cot"] = sample["cot"]

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data

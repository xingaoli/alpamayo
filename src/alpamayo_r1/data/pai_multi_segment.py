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

from typing import Any

from alpamayo_r1.data.pai import PAIDataset
from alpamayo_r1.load_physical_aiavdataset_local import load_physical_aiavdataset_local
from alpamayo_r1.common import logging

import torch

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class PAIMultiSegmentDataset(PAIDataset):
    """Dataset that samples multiple segments per clip for data augmentation.

    Instead of using a single fixed t0 per clip, this dataset creates multiple
    training samples by sampling different t0 timestamps within each clip.
    """

    def __init__(
        self,
        num_segments: int = 20,
        segment_start_us: int = 2_000_000,  # 2.0s
        segment_step_us: int = 500_000,  # 0.5s
        *args: Any,
        **kwargs: Any,
    ):
        # Don't pass use_default_keyframe to parent — t0 is determined by segments
        kwargs.pop("use_default_keyframe", None)
        super().__init__(*args, **kwargs)

        self.num_segments = num_segments
        self.segment_start_us = segment_start_us
        self.segment_step_us = segment_step_us

        # Pre-compute (clip_id, t0_us) pairs
        future_range_us = int(self.num_future_steps * self.time_step * 1_000_000)
        max_clip_duration_us = 19_900_000  # 19.9s safety margin
        self.clip_segment_t0s: list[tuple[str, int]] = []
        removed_count = 0

        for clip_id in self.clip_ids:
            for seg_idx in range(num_segments):
                t0_us = segment_start_us + seg_idx * segment_step_us
                if t0_us <= max_clip_duration_us - future_range_us:
                    self.clip_segment_t0s.append((clip_id, t0_us))
                else:
                    removed_count += 1

        if removed_count > 0:
            logger.info(f"Removed {removed_count} out-of-bounds segments (effective size: {len(self.clip_segment_t0s)})")

    def __len__(self) -> int:
        return len(self.clip_segment_t0s)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        clip_id, t0_us = self.clip_segment_t0s[idx]

        sample_data = load_physical_aiavdataset_local(
            clip_id,
            t0_us=t0_us,
            # avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        # squeeze ego motion shape
        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.include_extr_intr:
            sample_data["extr"] = self.avdi.get_clip_feature(clip_id, "sensor_extrinsics")
            sample_data["intr"] = self.avdi.get_clip_feature(clip_id, "camera_intrinsics")

            vehicle_dimensions = self.avdi.get_clip_feature(clip_id, "vehicle_dimensions")
            sample_data["ego_lwh"] = torch.tensor(
                [vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]
            )
            sample_data["ego_length_offset"] = torch.tensor(
                vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length
            )

        if self.reshape_tensors_for_rl:
            image_frames = sample_data["image_frames"]
            camera_indices = sample_data["camera_indices"]
            absolute_timestamps = sample_data["absolute_timestamps"]
            relative_timestamps = sample_data["relative_timestamps"]

            n_cam, n_frame = image_frames.shape[0], image_frames.shape[1]
            image_frames = image_frames.reshape(n_cam * n_frame, *image_frames.shape[2:]).unsqueeze(1)
            camera_indices = camera_indices.repeat_interleave(n_frame)
            absolute_timestamps = absolute_timestamps.reshape(-1)
            relative_timestamps = relative_timestamps.reshape(-1)

            sample_data["image_frames"] = image_frames
            sample_data["camera_indices"] = camera_indices
            sample_data["absolute_timestamps"] = absolute_timestamps
            sample_data["relative_timestamps"] = relative_timestamps

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data

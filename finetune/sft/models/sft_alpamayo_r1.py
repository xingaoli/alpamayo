# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import nullcontext
from typing import Any

import einops
import torch

from finetune.sft.models.sft_base_model import ReasoningVLAOutput, load_alpamayo1_vlm
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.common import misc
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class TrainableAlpamayoR1(AlpamayoR1):
    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        stage1_vlm_checkpoint_path: str | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size)

        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        # we only need the text config for the expert model
        if stage1_vlm_checkpoint_path is not None:
            self.vlm = load_alpamayo1_vlm(stage1_vlm_checkpoint_path, self.vlm)

        if not self.cotrain_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        # print the param count
        logger.info("Model parameter count:")
        param_count = misc.get_param_count(self)
        for key, value in param_count.items():
            logger.info(f"{key}: {value:,}")

    @classmethod
    def from_stage1_checkpoint(
        cls,
        stage1_checkpoint_path: str,
        vlm_name_or_path: str,
        traj_vocab_size: int = 768,
        tokens_per_future_traj: int = 64,
        tokens_per_history_traj: int = 16,
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> "TrainableAlpamayoR1":
        """Load Stage 1 VLM checkpoint and initialize expert module from scratch.

        Creates the full AlpamayoR1 architecture with VLM weights loaded from
        a Stage 1 training checkpoint. The expert module, action space, diffusion,
        and projection layers are initialized randomly (no pretrained weights).

        Args:
            stage1_checkpoint_path: Path to the Stage 1 checkpoint directory.
            vlm_name_or_path: Base VLM checkpoint (for processor/tokenizer).
            traj_vocab_size: Number of discrete trajectory tokens.
            tokens_per_future_traj: Tokens per future trajectory.
            tokens_per_history_traj: Tokens per history trajectory.
            traj_tokenizer_cfg: Config for future trajectory tokenizer.
            hist_traj_tokenizer_cfg: Config for history trajectory tokenizer.
            diffusion_cfg: Config for the diffusion model.
            action_space_cfg: Config for the action space.
            action_in_proj_cfg: Config for the action input projection.
            action_out_proj_cfg: Config for the action output projection.
            expert_cfg: Config overrides for the expert transformer.
            keep_same_dtype: Whether to cast action modules to expert dtype.
            expert_non_causal_attention: Whether the expert uses non-causal attention.
            cotrain_vlm: Whether to co-train VLM (typically False in Stage 2).
            stop_grad_from_vlm: Whether to stop gradients from VLM to expert.
            model_dtype: Data type for the VLM.
            attn_implementation: Attention implementation.
            min_pixels: Minimum pixels for image preprocessing.
            max_pixels: Maximum pixels for image preprocessing.
            add_special_tokens: Whether to add all special tokens.

        Returns:
            TrainableAlpamayoR1 model with frozen VLM and fresh expert.
        """
        from hydra.utils import instantiate as hydra_instantiate

        # 1. Build AlpamayoR1Config from scratch
        config = AlpamayoR1Config(
            vlm_name_or_path=vlm_name_or_path,
            vlm_backend="qwenvl3",
            traj_vocab_size=traj_vocab_size,
            tokens_per_future_traj=tokens_per_future_traj,
            tokens_per_history_traj=tokens_per_history_traj,
            traj_tokenizer_cfg=traj_tokenizer_cfg,
            hist_traj_tokenizer_cfg=hist_traj_tokenizer_cfg,
            diffusion_cfg=diffusion_cfg,
            action_space_cfg=action_space_cfg,
            action_in_proj_cfg=action_in_proj_cfg,
            action_out_proj_cfg=action_out_proj_cfg,
            expert_cfg=expert_cfg,
            keep_same_dtype=keep_same_dtype,
            expert_non_causal_attention=expert_non_causal_attention,
            model_dtype=model_dtype,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            add_special_tokens=add_special_tokens,
        )

        # 2. Instantiate trajectory tokenizer if config provided
        pretrained_modules = {}
        if config.traj_tokenizer_cfg is not None:
            traj_tokenizer = hydra_instantiate(config.traj_tokenizer_cfg)
            pretrained_modules["traj_tokenizer"] = traj_tokenizer

        # 3. Create model — VLM arch created with skip_init, then Stage 1 weights
        #    loaded via load_alpamayo1_vlm(); expert/action_space/diffusion/projections
        #    are initialized from scratch by AlpamayoR1.__init__
        model = cls(
            config,
            pretrained_modules=pretrained_modules or None,
            cotrain_vlm=cotrain_vlm,
            stop_grad_from_vlm=stop_grad_from_vlm,
            stage1_vlm_checkpoint_path=stage1_checkpoint_path,
        )

        logger.info(
            f"Loaded Stage 1 VLM from {stage1_checkpoint_path}, "
            f"expert module initialized from scratch",
        )

        return model

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Process the trajectory future data for training."""
        ego_history_xyz = traj_data["ego_history_xyz"]
        ego_history_rot = traj_data["ego_history_rot"]
        ego_future_xyz = traj_data["ego_future_xyz"]
        ego_future_rot = traj_data["ego_future_rot"]
        action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            traj_future_xyz=ego_future_xyz,
            traj_future_rot=ego_future_rot,
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        training_data: dict[str, Any] = self.diffusion.construct_training_data(action)
        return training_data

    def _process_position_ids_qwen2_5_vl(
        self, vlm_outputs: Any, batch_size: int, num_expert_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Process the position ids for the expert model.

        Qwen 2.5 VL has a special RoPE, so we need to process the position ids
        Args:
            vlm_outputs: The outputs of the VLM model.
            batch_size: The batch size.
            num_expert_tokens: The number of expert tokens.
            device: The device.
        Returns:
            The processed position ids.
        """
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model."""
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        batch_size = input_ids.shape[0]
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. vlm forward pass
        if self.cotrain_vlm:
            context = nullcontext()
        else:
            context = torch.no_grad()

        with context:
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
                **tokenized_data,
            )

        future_start_token_id = self.config.traj_token_ids["future_start"]
        last_traj_future_start_idx = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1

        future_traj_data = self._process_traj_future_training(traj_data)
        # [B, n_token_per_future_traj, hidden_size]
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )
        # [B, n_token_per_history_traj + n_token_per_future_traj, hidden_size]
        expert_embeds = action_embeds
        # NOTE: we don't need to update the rope deltas as we assume after <traj_future_start> there
        # will be no more vision tokens.
        kv_cache = vlm_outputs.past_key_values
        # crop the kv cache to the last <traj_future_start> token
        kv_cache.crop(last_traj_future_start_idx)
        if self.stop_grad_from_vlm:
            for layer in kv_cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()
        position_ids = self._process_position_ids_qwen2_5_vl(
            vlm_outputs, batch_size, expert_embeds.shape[1], expert_embeds.device
        )
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        future_traj_loss = (
            self.diffusion.compute_loss_from_pred(
                training_data=future_traj_data,
                pred=pred,
            )
            # TODO: only support traj finetune for now, so no weight, add weight later when other losses added
            # * self.config.traj_loss_weight
        )
        loss = future_traj_loss
        if self.cotrain_vlm:
            loss += vlm_outputs.loss

        return ReasoningVLAOutput(
            loss=loss,
        )

    def sample_trajectories_from_data(  # type: ignore[override]
        self,
        data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data.

        Args:
            with_vlm_rollout: Whether to use VLM rollout.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.sample_trajectories_from_data_with_vlm_rollout(
            data,
            *args,
            **kwargs,
        )

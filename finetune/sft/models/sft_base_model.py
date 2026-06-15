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

from dataclasses import dataclass
from typing import Any, Mapping
import json
from collections import defaultdict
from pathlib import Path
import einops
import numpy as np
import torch
from transformers.utils import ModelOutput
from hydra.utils import instantiate
from safetensors.torch import load_file as load_safetensors_file

from alpamayo_r1.models.base_model import (
    ReasoningVLA,
    ReasoningVLAConfig,
    TrajectoryFusionMixin,
    tokenize_history_trajectory,
    replace_pad_token,
)
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.token_utils import extract_traj_tokens, extract_text_tokens
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def tokenize_future_trajectory(
    tokenizer, traj_data: Mapping[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the future trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: BaseTrajectoryTokenizer
        traj_data: dict[str, Any], needs to at least contain:
            - "ego_history_xyz"
            - "ego_history_rot"
            - "ego_future_xyz"
            - "ego_future_rot"
        start_idx: int, start of token index of the future trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_future_traj]
    """
    assert "ego_future_xyz" in traj_data
    assert traj_data["ego_future_xyz"].ndim == 4, "ego_future_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)

    fut_idx = (
        tokenizer.encode(hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot)
        + start_idx
    )
    fut_idx = einops.rearrange(fut_idx, "(b n_traj) n -> b (n_traj n)", b=B)
    return fut_idx


def load_alpamayo1_vlm(checkpoint_path: str, model: Any):
    # Load fine-tuned vlm.* weights from the checkpoint directory.
    # The caller may pass either the wrapper model (keys need "vlm." prefix)
    # or the VLM sub-module (keys must NOT have "vlm." prefix).
    # Detect which one by checking the model's own state_dict keys.
    model_sd_keys = set(model.state_dict().keys())
    needs_vlm_prefix = any(k.startswith("vlm.") for k in model_sd_keys)

    checkpoint_dir = Path(checkpoint_path)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    vlm_state_dict: dict[str, torch.Tensor] = {}

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f).get("weight_map", {})

        shard_to_keys: defaultdict[str, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            if key.startswith("vlm."):
                shard_to_keys[shard_name].append(key)

        for shard_name, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_name
            shard_sd = load_safetensors_file(str(shard_path), device="cpu")
            for key in keys:
                if key in shard_sd:
                    out_key = key if needs_vlm_prefix else key.removeprefix("vlm.")
                    vlm_state_dict[out_key] = shard_sd[key]
    else:
        # Single safetensors file (no index)
        safetensors_files = sorted(checkpoint_dir.glob("model*.safetensors"))
        for safetensors_file in safetensors_files:
            shard_sd = load_safetensors_file(str(safetensors_file), device="cpu")
            for key in shard_sd:
                if key.startswith("vlm."):
                    out_key = key if needs_vlm_prefix else key.removeprefix("vlm.")
                    vlm_state_dict[out_key] = shard_sd[key]

    if not vlm_state_dict:
        raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")

    # Guard: ckpt sometimes omits lm_head weight (Qwen3-VL uses weight tying with
    # embed_tokens; if the tied lm_head is filtered out by the saver, it must be
    # re-bound from embed_tokens on load — otherwise inference produces garbled tokens
    # because the original (152064,) HF lm_head doesn't cover the new 155697 vocab).
    lm_head_key = "vlm.lm_head.weight" if needs_vlm_prefix else "lm_head.weight"
    embed_key = "vlm.model.language_model.embed_tokens.weight" if needs_vlm_prefix else "model.language_model.embed_tokens.weight"
    if lm_head_key not in vlm_state_dict:
        if embed_key in vlm_state_dict:
            vlm_state_dict[lm_head_key] = vlm_state_dict[embed_key]
            logger.info(
                f"{lm_head_key} missing in checkpoint, bound from embed_tokens (weight tying)."
            )

    load_result = model.load_state_dict(vlm_state_dict, strict=False, assign=True)
    logger.info(
        f"Loaded {len(vlm_state_dict)} VLM tensors from {checkpoint_dir} (missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})",
    )

    return model


class TrajectoryFusionWithFutureMixin(TrajectoryFusionMixin):
    def fuse_traj_tokens(
        self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Fuse the trajectory tokens into the input ids.

        Args:
            input_ids: [B, n_token]
            traj_data: dict containing ego_history_xyz, ego_history_rot, etc.

        Returns:
            input_ids: [B, n_token] with trajectory tokens fused
        """
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            return input_ids

        has_future = "ego_future_xyz" in traj_data and traj_data["ego_future_xyz"] is not None
        attrs = self._validate_mixin_requirements(require_future=has_future)

        hist_idx = tokenize_history_trajectory(
            attrs["hist_traj_tokenizer"], traj_data, attrs["hist_token_start_idx"]
        )
        input_ids = replace_pad_token(
            input_ids, hist_idx, attrs["config"].traj_token_ids["history"]
        )

        if has_future:
            future_idx = tokenize_future_trajectory(
                attrs["traj_tokenizer"], traj_data, attrs["future_token_start_idx"]
            )
            input_ids = replace_pad_token(
                input_ids, future_idx, attrs["config"].traj_token_ids["future"]
            )

        return input_ids


@dataclass
class ReasoningVLAOutput(ModelOutput):
    """Output of the ReasoningVLA model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class TrainableReasoningVLA(ReasoningVLA, TrajectoryFusionWithFutureMixin):
    """Trainable ReasoningVLA model."""

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count)

    @classmethod
    def from_alpamayo_checkpoint(
        cls,
        checkpoint_path: str,
        vlm_name_or_path: str,
        **kwargs: Any,
    ) -> "ReasoningVLA":
        """Load the base ReasoningVLA from a full Alpamayo checkpoint.

        Creates the VLM architecture from config (without loading base weights),
        then loads the fine-tuned ``vlm.*`` weights from the Alpamayo checkpoint.
        The trajectory tokenizer is instantiated separately with its own pretrained
        weights.

        Unlike ``from_pretrained_submodules``, this avoids loading the base VLM
        weights from HuggingFace and the expensive ``resize_token_embeddings``
        call, since the checkpoint already contains correctly-sized embeddings.

        Args:
            checkpoint_path: Path to full Alpamayo checkpoint directory.
            vlm_name_or_path: Base VLM checkpoint used to build processor/tokenizer.
            **kwargs: Extra keyword arguments (currently unused).

        Returns:
            Initialized ``ReasoningVLA`` model with Alpamayo VLM weights.
        """
        checkpoint_config_path = Path(checkpoint_path) / "config.json"
        if not checkpoint_config_path.exists():
            raise FileNotFoundError(f"Missing config file: {checkpoint_config_path}")

        with checkpoint_config_path.open("r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)

        config_kwargs = {
            "vlm_name_or_path": vlm_name_or_path,
            "vlm_backend": checkpoint_config.get("vlm_backend", "qwenvl3"),
            "traj_tokenizer_cfg": checkpoint_config.get("traj_tokenizer_cfg"),
            "hist_traj_tokenizer_cfg": checkpoint_config.get("hist_traj_tokenizer_cfg"),
            "traj_vocab_size": checkpoint_config.get("traj_vocab_size"),
            "tokens_per_history_traj": checkpoint_config.get("tokens_per_history_traj"),
            "tokens_per_future_traj": checkpoint_config.get("tokens_per_future_traj"),
            "model_dtype": checkpoint_config.get("model_dtype", "bfloat16"),
            "attn_implementation": checkpoint_config.get(
                "attn_implementation", "flash_attention_2"
            ),
            "min_pixels": checkpoint_config.get("min_pixels"),
            "max_pixels": checkpoint_config.get("max_pixels"),
            "add_special_tokens": checkpoint_config.get("add_special_tokens", True),
        }
        config = instantiate(
            {
                "_target_": f"alpamayo_r1.models.base_model.{cls.config_class.__name__}",
                "_recursive_": False,
                "_convert_": "all",
                **config_kwargs,
            }
        )

        pretrained_modules = {}
        if config.traj_tokenizer_cfg is not None:
            pretrained_modules["traj_tokenizer"] = instantiate(config.traj_tokenizer_cfg)

        # Build the model with real (materialized) parameters and load checkpoint
        # weights with DeepSpeed ZeRO-3's auto-partitioning temporarily disabled.
        #
        # Creating ``TrainingArguments(deepspeed=...)`` beforehand installs a HF
        # ``HfDeepSpeedConfig``; its weak ref makes ``is_deepspeed_zero3_enabled()``
        # return True, and transformers then wraps *every* model construction in
        # ``deepspeed.zero.Init()`` (enabled=True). Parameters therefore come up
        # empty/partitioned (shape [0]) and the ``load_state_dict`` below fails with
        # size-mismatch errors. Temporarily clearing that weak ref disables the
        # auto-partitioning for this construction + weight load; restoring it lets
        # the HF Trainer re-partition the returned model afterward. The
        # ``zero.Init(enabled=False)`` wrapper is a belt-and-suspenders second layer.
        _ds_ref_module = None
        try:
            from transformers.integrations import deepspeed as _ds_ref_module
        except ImportError:
            try:
                import transformers.deepspeed as _ds_ref_module  # type: ignore
            except ImportError:
                _ds_ref_module = None
        _saved_ds_ref = getattr(_ds_ref_module, "_hf_deepspeed_config_weak_ref", "missing")
        _cleared_ds_ref = (
            _ds_ref_module is not None and _saved_ds_ref != "missing"
        )
        if _cleared_ds_ref:
            _ds_ref_module._hf_deepspeed_config_weak_ref = None

        try:
            import deepspeed

            init_ctx = deepspeed.zero.Init(enabled=False)
        except Exception:
            from contextlib import nullcontext

            init_ctx = nullcontext()

        try:
            with init_ctx:
                model = cls(config, pretrained_modules=pretrained_modules or None)
                model = load_alpamayo1_vlm(checkpoint_path, model)
        finally:
            if _cleared_ds_ref:
                _ds_ref_module._hf_deepspeed_config_weak_ref = _saved_ds_ref

        return model

    @classmethod
    def from_qwen3vl_checkpoint(
        cls,
        vlm_name_or_path: str,
        traj_vocab_size: int = 768,
        tokens_per_future_traj: int = 64,
        tokens_per_history_traj: int = 16,
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> "TrainableReasoningVLA":
        """Load a fresh Qwen3-VL checkpoint for Stage 1 training.

        Unlike ``from_alpamayo_checkpoint`` which loads fine-tuned VLM weights from an
        existing Alpamayo checkpoint, this loads the full base VLM from HuggingFace,
        resizes token embeddings for trajectory tokens, and instantiates the trajectory
        tokenizer -- providing a clean starting point for Stage 1 SFT.

        Args:
            vlm_name_or_path: HuggingFace path or local path to a Qwen3-VL model.
            traj_vocab_size: Number of discrete trajectory tokens to add.
            tokens_per_future_traj: Number of tokens per future trajectory.
            tokens_per_history_traj: Number of tokens per history trajectory.
            traj_tokenizer_cfg: Hydra config dict for the future trajectory tokenizer.
            hist_traj_tokenizer_cfg: Hydra config dict for the history trajectory tokenizer.
            model_dtype: Data type for the VLM.
            attn_implementation: Attention implementation (e.g. "flash_attention_2").
            min_pixels: Minimum pixels for image preprocessing.
            max_pixels: Maximum pixels for image preprocessing.
            add_special_tokens: Whether to add all special tokens (vs. only traj tokens).

        Returns:
            TrainableReasoningVLA model ready for Stage 1 SFT training.
        """
        from alpamayo_r1.models.base_model import ReasoningVLAConfig
        from transformers import Qwen3VLForConditionalGeneration

        # 1. Build ReasoningVLAConfig from scratch
        config = ReasoningVLAConfig(
            vlm_name_or_path=vlm_name_or_path,
            vlm_backend="qwenvl3",
            traj_vocab_size=traj_vocab_size,
            tokens_per_future_traj=tokens_per_future_traj,
            tokens_per_history_traj=tokens_per_history_traj,
            traj_tokenizer_cfg=traj_tokenizer_cfg,
            hist_traj_tokenizer_cfg=hist_traj_tokenizer_cfg,
            model_dtype=model_dtype,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            add_special_tokens=add_special_tokens,
        )

        # 2. Load VLM from HuggingFace and resize embeddings
        pretrained_modules = {}
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_name_or_path,
            torch_dtype=getattr(torch, config.model_dtype),
            attn_implementation=config.attn_implementation,
        )
        original_vocab_size = vlm.config.text_config.vocab_size
        vlm.resize_token_embeddings(config.vocab_size)
        vlm.config.text_config.vocab_size = config.vocab_size
        vlm.config.vocab_size = config.vocab_size
        pretrained_modules["vlm"] = vlm

        # 3. Instantiate trajectory tokenizer if config provided
        if config.traj_tokenizer_cfg is not None:
            traj_tokenizer = instantiate(config.traj_tokenizer_cfg)
            pretrained_modules["traj_tokenizer"] = traj_tokenizer

        # 4. Create model
        model = cls(
            config,
            pretrained_modules=pretrained_modules or None,
            original_vocab_size=original_vocab_size,
        )

        logger.info(
            f"Loaded fresh VLM from {vlm_name_or_path} "
            f"(original_vocab_size={original_vocab_size}, new_vocab_size={config.vocab_size})",
        )

        return model

    def tie_weights(
        self,
        recompute_mapping: bool = False,
        missing_keys: list[str] = None,
        unexpected_keys: list[str] = None,
    ) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the next token prediction.

        Args:
            outputs: ModelOutput containing logits of shape (B, L, V)
            labels: [B, L]
            labels_mask: [B, L], indicates which tokens in the sequence are valid for loss
                computation
            token_mask: [V], indicates which tokens ids are valid for logits computation

        Returns:
            torch.Tensor: (,) loss value
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        # Shift labels to the left by 1 position (predict next token)
        shift_labels = labels[..., 1:]
        # The logits should also be trimmed to match the shifted labels
        # NOTE: we clone the logits to avoid in-place operations if token_mask is present that will
        # modify the original
        shift_logits = outputs.logits[..., :-1, :].clone()

        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        loss = torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
            ),
            nan=0.0,
        )
        return loss

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
        outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)

        losses = {}
        # Identify trajectory tokens (tokens between traj_future and next special token)
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(outputs, labels, traj_mask)
        #  * self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX

        # Include all other tokens in the loss
        losses["others"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)
        #  * self.config.loss_weights.get("others", 1.0)

        # Replace the original loss
        outputs.loss = sum(losses.values())

        return ReasoningVLAOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        if "tokenized_data" not in data:
            tokenized_data = self._get_generation_mode_tokenized_data_online(
                data, last_component=last_component
            )
        else:
            tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # extract trajectory tokens from generated sequences
        traj_token_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=einops.repeat(
                ego_history_xyz[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            hist_rot=einops.repeat(
                ego_history_rot[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            tokens=traj_token_ids,
        )
        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )

        # return additional information
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

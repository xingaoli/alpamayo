"""Debug script for testing PAIDataset + collate_fn pipeline outside of training."""

import json
import hydra.utils as hyu
from pathlib import Path


def load_model_config(checkpoint_path: str, vlm_name_or_path: str):
    """Build ReasoningVLAConfig from checkpoint config.json without loading the model."""
    with open(Path(checkpoint_path) / "config.json", "r") as f:
        ckpt_config = json.load(f)

    config_kwargs = {
        "vlm_name_or_path": vlm_name_or_path,
        "vlm_backend": ckpt_config.get("vlm_backend", "qwenvl3"),
        "traj_tokenizer_cfg": ckpt_config.get("traj_tokenizer_cfg"),
        "hist_traj_tokenizer_cfg": ckpt_config.get("hist_traj_tokenizer_cfg"),
        "traj_vocab_size": ckpt_config.get("traj_vocab_size"),
        "tokens_per_history_traj": ckpt_config.get("tokens_per_history_traj"),
        "tokens_per_future_traj": ckpt_config.get("tokens_per_future_traj"),
        "model_dtype": ckpt_config.get("model_dtype", "bfloat16"),
        "attn_implementation": ckpt_config.get("attn_implementation", "flash_attention_2"),
        "min_pixels": ckpt_config.get("min_pixels"),
        "max_pixels": ckpt_config.get("max_pixels"),
        "add_special_tokens": ckpt_config.get("add_special_tokens", True),
    }
    return hyu.instantiate(
        {
            "_target_": "alpamayo_r1.config.ReasoningVLAConfig",
            "_recursive_": False,
            "_convert_": "all",
            **config_kwargs,
        }
    )


def main():
    # 1. Load model config only (no model weights, fast)
    CHECKPOINT_PATH = "ckpts/Alpamayo-1.5-10B"
    VLM_NAME_OR_PATH = "ckpts/Qwen3-VL-8B-Instruct-config"
    model_config = load_model_config(CHECKPOINT_PATH, VLM_NAME_OR_PATH)
    print(f"Model config loaded: vocab_size={model_config.vocab_size}, vlm={model_config.vlm_name_or_path}")

    # vla_preprocess_args needs the full config from vla_processor.yaml
    vla_preprocess_args = {
        "_target_": "alpamayo_r1.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        "components_order": ["image", "traj_history", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "label_components": ["traj_future"],
    }

    # 2. Create train dataset (multi-segment)
    print("=" * 60)
    print("Creating train dataset (multi-segment)...")
    train_dataset = hyu.instantiate(
        {
            "_recursive_": False,
            "_target_": "alpamayo_r1.data.pai_multi_segment.PAIMultiSegmentDataset",
            "local_dir": "data/PhysicalAI-Autonomous-Vehicles",
            "chunk_ids": "0-48",
            "num_segments": 20,
            "segment_start_us": 2000000,
            "segment_step_us": 500000,
            "vla_preprocess_args": {**vla_preprocess_args, "generation_mode": False},
        },
        model_config=model_config,
    )
    print(f"Train dataset size: {len(train_dataset)} segments")
    print(f"Expected: {len(train_dataset.clip_ids)} clips × 20 = {len(train_dataset.clip_ids) * 20}")
    print(f"Clip IDs (first 5): {train_dataset.clip_ids[:5]}")
    if len(train_dataset.clip_segment_t0s) > 0:
        print(f"Sample t0s for first clip: {[f'{t0/1e6:.1f}s' for _, t0 in train_dataset.clip_segment_t0s[:5]]}")

    # 3. Create eval dataset
    print("=" * 60)
    print("Creating eval dataset...")
    eval_dataset = hyu.instantiate(
        {
            "_recursive_": False,
            "_target_": "alpamayo_r1.data.pai.PAIDataset",
            "local_dir": "data/PhysicalAI-Autonomous-Vehicles",
            "chunk_ids": "48-50",
            "use_default_keyframe": True,
            "vla_preprocess_args": {**vla_preprocess_args, "generation_mode": True},
        },
        model_config=model_config,
    )
    print(f"Eval dataset size: {len(eval_dataset)} clips")

    # 4. Create collate_fn
    print("=" * 60)
    print("Creating collate_fn...")
    collate_fn = hyu.instantiate(
        {
            "_target_": "alpamayo_r1.processor.qwen_processor.collate_fn_from_model_config",
            "_partial_": True,
        },
        model_config=model_config,
    )

    # 5. Test single __getitem__
    print("=" * 60)
    print("Testing train_dataset[0]...")
    sample = train_dataset[0]
    print(f"Keys: {list(sample.keys())}")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
        elif hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")

    # 6. Test collate_fn with a batch of samples
    print("=" * 60)
    print("Testing collate_fn with 2 samples...")
    samples = [train_dataset[i] for i in range(2)]
    batch = collate_fn(samples)
    print(f"Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"  {key}: list[{len(value)}], first shape={value[0].shape}")
            else:
                print(f"  {key}: list[{len(value)}]")
        else:
            print(f"  {key}: {type(value).__name__}")

    print("=" * 60)
    print("Done.")


import torch

if __name__ == "__main__":
    main()

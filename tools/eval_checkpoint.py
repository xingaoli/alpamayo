"""Evaluate a trained Stage 1 or Stage 2 checkpoint on the eval dataset.

Usage (multi-GPU):
    torchrun --nproc_per_node=2 tools/eval_checkpoint.py --stage 2 --checkpoint ckpts/Alpamayo-1.5-10B

Usage (single-GPU):
    python tools/eval_checkpoint.py --stage 1 --checkpoint outputs/output_stage1/checkpoint-1000
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate
from tqdm.auto import tqdm

from alpamayo_r1.common import logging

logging.setup_logging()
logger = logging.RankedLogger("eval", rank_zero_only=True)
logger.setLevel("INFO")


def setup_distributed():
    """Initialize distributed training. Returns (rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_config(checkpoint_path: str, vlm_name_or_path: str):
    """Build ReasoningVLAConfig from checkpoint config.json without loading model weights."""
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
    return instantiate(
        {
            "_target_": "alpamayo_r1.config.ReasoningVLAConfig",
            "_recursive_": False,
            "_convert_": "all",
            **config_kwargs,
        }
    )


def load_model(stage: int, checkpoint_path: str, vlm_name_or_path: str):
    """Load model for the given stage."""
    if stage == 1:
        logger.info(f"Loading Stage 1 model from {checkpoint_path}...")
        model = instantiate(
            {
                "_target_": "finetune.sft.models.sft_base_model.TrainableReasoningVLA.from_alpamayo_checkpoint",
                "_recursive_": False,
                "checkpoint_path": checkpoint_path,
                "vlm_name_or_path": vlm_name_or_path,
            }
        )
    elif stage == 2:
        logger.info(f"Loading Stage 2 model from {checkpoint_path}...")
        model = instantiate(
            {
                "_target_": "finetune.sft.models.sft_alpamayo_r1.TrainableAlpamayoR1.from_pretrained",
                "_recursive_": False,
                "pretrained_model_name_or_path": checkpoint_path,
                "dtype": "auto",
                "cotrain_vlm": False,
                "stage1_vlm_checkpoint_path": None,
            }
        )
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1 or 2.")

    model = model.cuda().eval()
    return model


def to_device(d, device="cuda"):
    """Recursively move tensors/dicts/lists to device."""
    if isinstance(d, dict):
        return {k: to_device(v, device) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_device(v, device) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    return d


def gather_metric(metric_tensor):
    """Gather metric tensors across all GPUs."""
    if not dist.is_initialized():
        return metric_tensor
    gathered = [torch.zeros_like(metric_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, metric_tensor)
    return torch.cat(gathered)


def main():
    rank, world_size = setup_distributed()
    is_main = rank == 0

    parser = argparse.ArgumentParser(description="Evaluate Stage 1/2 checkpoint")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1 or 2")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default=os.getenv("ALPAMAYO_DATA_DIR", "data/PhysicalAI-Autonomous-Vehicles"))
    parser.add_argument("--eval_chunks", type=str, default="48-50")
    parser.add_argument("--vlm", type=str, default="ckpts/Qwen3-VL-8B-Instruct-config")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="-1 for all")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # --- Load model ---
    if is_main:
        print("Loading model config...")
    model_config = load_model_config(args.checkpoint, args.vlm)
    model = load_model(args.stage, args.checkpoint, args.vlm)
    if is_main:
        print(f"Model loaded. Total params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Load eval dataset ---
    vla_preprocess_args = {
        "_target_": "alpamayo_r1.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        "components_order": ["image", "traj_history", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "label_components": ["traj_future"],
        "generation_mode": True,
    }

    if is_main:
        print("Creating eval dataset...")
    eval_dataset = instantiate(
        {
            "_recursive_": False,
            "_target_": "alpamayo_r1.data.pai_multi_segment.PAIMultiSegmentDataset",
            "local_dir": args.data_dir,
            "chunk_ids": args.eval_chunks,
            "num_segments": 20,
            "segment_start_us": 2_000_000,
            "segment_step_us": 500_000,
            "vla_preprocess_args": vla_preprocess_args,
        },
        model_config=model_config,
    )
    if is_main:
        print(f"Eval dataset size: {len(eval_dataset)} segments")

    # --- Load collate_fn ---
    collate_fn = instantiate(
        {
            "_target_": "alpamayo_r1.processor.qwen_processor.collate_fn_from_model_config",
            "_partial_": True,
        },
        model_config=model_config,
    )

    # --- Setup distributed sampler ---
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers)

    num_total = len(dataloader)

    # --- Setup metrics ---
    metric_runner = instantiate(
        {
            "_target_": "alpamayo_r1.metrics.metric_runner.MetricRunner",
            "metrics": [
                {
                    "_target_": "alpamayo_r1.metrics.metric_api.ReasoningSampler",
                    "num_traj_sets": 1,
                    "num_traj_samples": args.num_traj_samples,
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "traj_only_generation": False,
                    "max_generation_length": args.max_new_tokens,
                },
                {
                    "_target_": "alpamayo_r1.metrics.metric_api.DistanceMetrics",
                },
            ],
        }
    )

    # --- Run evaluation ---
    if is_main:
        print(f"Starting Stage {args.stage} evaluation on {world_size} GPU(s)...")
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    eval_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=num_total, disable=not is_main, desc="Evaluating")
        for batch in pbar:
            batch = to_device(batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch = {}
                metric_runner.run(model, batch, output_batch)

            for k, v in output_batch.items():
                if not k.startswith("metric/"):
                    continue
                gathered = gather_metric(v.float())
                if is_main:
                    metric_sums[k] += gathered.sum().item()
                    metric_counts[k] += gathered.numel()
            eval_count += 1

    # --- Print results ---
    if not is_main:
        cleanup_distributed()
        return

    print("\n" + "=" * 60)
    print(f"Stage {args.stage} | Evaluated {eval_count} samples | Checkpoint: {args.checkpoint}")
    print("-" * 60)
    for key in sorted(metric_sums.keys()):
        if metric_counts[key] > 0:
            val = metric_sums[key] / metric_counts[key]
            print(f"  {key:<30} {val:.4f}")
    print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()

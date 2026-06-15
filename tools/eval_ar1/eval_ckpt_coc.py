"""Evaluate a CoC-trained Stage 1 or Stage 2 checkpoint.

Generates both CoC reasoning text and trajectory predictions, compares with
ground-truth CoC from the dataset, and writes per-sample results to JSONL.

Unlike eval_checkpoint.py (which uses the standard vla processor and only
generates trajectory), this script explicitly sets the assistant prefix to
"<|cot_start|>" so the model generates CoC from scratch (matching the
inference pattern in tools/coc_analysis/1_batch_coc_inference.py).

Usage (multi-GPU):
    torchrun --nproc_per_node=2 tools/eval_ar1/eval_ckpt_coc.py \
        --stage 2 --checkpoint outputs/output_coc_stage2_2b/checkpoint-500 \
        --coc_jsonl data/coc.jsonl --vlm ckpts/Qwen3-VL-2B-Instruct

Usage (single-GPU):
    python tools/eval_ar1/eval_ckpt_coc.py \
        --stage 1 --checkpoint outputs/output_coc_stage1_2b/checkpoint-500 \
        --coc_jsonl data/coc.jsonl --vlm ckpts/Qwen3-VL-2B-Instruct
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm.auto import tqdm

from alpamayo_r1 import helper
from alpamayo_r1.common import logging
from alpamayo_r1.data.coc import COCDataset
from alpamayo_r1.metrics.distance_metrics import (
    compute_ade,
    compute_minade,
    compute_grouped_corner_distance,
)

logging.setup_logging()
logger = logging.RankedLogger("eval_coc", rank_zero_only=True)
logger.setLevel("INFO")

# Reuse helpers from eval_checkpoint.py
from eval_checkpoint import (
    setup_distributed,
    cleanup_distributed,
    load_model_config,
    load_model,
    to_device,
    gather_metric,
)


# ---------------------------------------------------------------------------
# Wrapper dataset: inject clip_id / ts into the batch so we can write them
# to JSONL without reaching back into the dataset internals. We deliberately
# do NOT pass vla_preprocess_args — tokenization is done in the eval loop
# using helper.create_message so we can inject "<|cot_start|>" as the
# assistant prefix and force the model to generate CoC from scratch.
# ---------------------------------------------------------------------------
class COCEvalDataset(COCDataset):
    """Thin wrapper that adds clip_id and ts to the returned sample dict."""

    def __getitem__(self, idx: int) -> dict:
        sample_data = super().__getitem__(idx)
        raw = self.samples[self.valid_indices[idx]]
        sample_data["clip_id"] = raw["clip_id"]
        sample_data["ts"] = raw["ts"]
        # We need these top-level keys for sample_trajectories_from_data.
        # They are stacked by the default collate.
        return sample_data


def build_tokenized_data(sample: dict, processor) -> dict:
    """Tokenize one sample with assistant prefix = '<|cot_start|>'.

    The model will then autoregressively generate CoC text until
    '<|cot_end|>' (or '<|traj_future_start|>' for stage 2).
    """
    image_frames = sample["image_frames"]  # [N, C, H, W] or [B, N, C, H, W]
    if image_frames.ndim == 5:
        image_frames = image_frames[0]  # drop leading batch dim from COCDataset

    messages = helper.create_message(image_frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def main():
    rank, world_size = setup_distributed()
    is_main = rank == 0

    # ------------------------------------------------------------------ args
    parser = argparse.ArgumentParser(description="Evaluate CoC checkpoint")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coc_jsonl", type=str, default=os.getenv("ALPAMAYO_COC_JSONL", ""))
    parser.add_argument("--data_dir", type=str,
                        default=os.getenv("ALPAMAYO_DATA_DIR", "data/PhysicalAI-Autonomous-Vehicles"))
    parser.add_argument("--vlm", type=str, default="ckpts/Qwen3-VL-8B-Instruct-config")
    parser.add_argument("--stage1_vlm_checkpoint_path", type=str, default=None,
                        help="Path to Stage 1 VLM checkpoint for correct lm_head loading (Stage 2 only)")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=-1,
                        help="-1 = auto (256 for both stages when evaluating CoC)")
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_jsonl", type=str, default="eval_coc_results.jsonl")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        args.max_new_tokens = 256

    assert args.coc_jsonl, (
        "Must provide --coc_jsonl or set ALPAMAYO_COC_JSONL env var"
    )

    if is_main:
        print(f"[eval_coc] stage={args.stage}  checkpoint={args.checkpoint}")
        print(f"[eval_coc] coc_jsonl={args.coc_jsonl}  data_dir={args.data_dir}")
        print(f"[eval_coc] K={args.num_traj_samples}  top_p={args.top_p}  "
              f"temperature={args.temperature}  max_new_tokens={args.max_new_tokens}")

    # -------------------------------------------------------------- model
    model_config = load_model_config(args.checkpoint, args.vlm)
    model = load_model(args.stage, args.checkpoint, args.vlm, stage1_vlm_checkpoint_path=args.stage1_vlm_checkpoint_path)
    if is_main:
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[eval_coc] Model loaded. total params={n_total:,}")

    # Processor used for tokenization in the eval loop
    processor = helper.get_processor(model.tokenizer)

    # -------------------------------------------------------------- dataset
    # Pass vla_preprocess_args=None: COCDataset returns raw sample data,
    # we tokenize manually in the eval loop with assistant prefix =
    # "<|cot_start|>".
    eval_dataset = COCEvalDataset(
        coc_jsonl_path=args.coc_jsonl,
        local_dir=args.data_dir,
        model_config=model_config,
        vla_preprocess_args=None,
    )
    total_size = len(eval_dataset)
    if args.max_eval_samples > 0 and args.max_eval_samples < total_size:
        eval_dataset = Subset(eval_dataset, list(range(args.max_eval_samples)))
        if is_main:
            print(f"[eval_coc] dataset: {total_size} -> subset to {args.max_eval_samples}")
    elif is_main:
        print(f"[eval_coc] dataset: {total_size} samples")

    # --------------------------------------------------------- dataloader
    # Force batch_size=1: helper.create_message tokenizes per-sample.
    if args.batch_size != 1:
        if is_main:
            print(f"[eval_coc] WARNING: --batch_size {args.batch_size} not supported, "
                  f"using 1 (per-sample tokenization required for CoC)")
        args.batch_size = 1
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size,
                                 rank=rank, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, batch_size=1, sampler=sampler,
        collate_fn=lambda x: x[0],  # return single sample dict, not a list
        num_workers=args.num_workers,
    )
    if is_main:
        print(f"[eval_coc] dataloader: {len(dataloader)} samples/rank  "
              f"(world={world_size})")

    # Corner dimensions for corner_distance metric
    corner_dims = torch.tensor([4.0, 3.0, 2.0], device="cuda")

    # --------------------------------------------------------------- eval
    rank_jsonl = f"{args.output_jsonl}.rank_{rank}"
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    eval_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), disable=not is_main,
                    desc="Evaluating CoC")
        for batch_idx, sample in enumerate(pbar):
            sample = to_device(sample)

            # ----- tokenize this single sample with "<|cot_start|>" prefix -----
            tokenized = build_tokenized_data(sample, processor)
            # apply_chat_template returns BatchEncoding (UserDict subclass);
            # eval_checkpoint.to_device only handles plain dict, so convert first.
            tokenized = {k: v for k, v in tokenized.items()}
            tokenized = to_device(tokenized)

            # Build a data dict compatible with sample_trajectories_from_data.
            # COCDataset squeezes the batch dim (ego_* are 3D), but the model
            # expects [B, n_traj, T, 3]. Add batch dim back.
            data = {
                "tokenized_data": tokenized,
                "ego_history_xyz": sample["ego_history_xyz"].unsqueeze(0),
                "ego_history_rot": sample["ego_history_rot"].unsqueeze(0),
            }

            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data(
                    data=data,
                    num_traj_samples=args.num_traj_samples,
                    num_traj_sets=1,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    traj_only_generation=False,
                    max_generation_length=args.max_new_tokens,
                    return_extra=True,
                )

            if len(result) == 3:
                pred_xyz, pred_rot, extra = result
            else:
                pred_xyz, pred_rot = result[:2]
                extra = {}

            # ----- generated CoC text -----
            # extra["cot"] has shape [B, N=1, K] — VLM generate() produces K
            # sequences, each with its own CoC text. Save all K for downstream
            # judging; gen_cot = first candidate for quick inspection.
            if extra and "cot" in extra:
                cot_arr = np.asarray(extra["cot"])  # [B=1, ns, nj]
                gen_cot_all = [str(c) for c in cot_arr[0, 0, :].tolist()]
                gen_cot = gen_cot_all[0] if gen_cot_all else ""
            else:
                gen_cot_all = []
                gen_cot = ""

            # ----- per-sample trajectory metrics -----
            # Add leading batch dim for metric functions
            gt_xyz = sample["ego_future_xyz"].unsqueeze(0)[:, -1]  # [1, T, 3]
            gt_rot = sample["ego_future_rot"].unsqueeze(0)[:, -1]  # [1, T, 3, 3]

            minade_dict = compute_minade(pred_xyz, gt_xyz, disable_summary=True)
            ade_all = compute_ade(pred_xyz, gt_xyz)  # [B, N, K]
            corner_dict = compute_grouped_corner_distance(
                pred_xyz, pred_rot, gt_xyz, gt_rot,
                dims=corner_dims, disable_summary=True,
            )

            # ----- write per-sample JSONL -----
            entry = {
                "clip_id": str(sample["clip_id"]),
                "ts": int(sample["ts"]),
                "gt_cot": sample["cot"],
                "gen_cot": gen_cot,
                "gen_cot_all": gen_cot_all,
                "min_ade": float(minade_dict["min_ade"][0].item()),
                "ade_best": float(ade_all[0].min().item()),
                "ade_mean": float(ade_all[0].mean().item()),
                "corner_distance": float(corner_dict["corner_distance"][0].item()),
            }
            with open(rank_jsonl, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # ----- aggregate for summary -----
            for k, v in minade_dict.items():
                gathered = gather_metric(v.float())
                if is_main:
                    metric_sums[f"metric/{k}"] += gathered.sum().item()
                    metric_counts[f"metric/{k}"] += gathered.numel()

            ade_gathered = gather_metric(ade_all.float())
            if is_main:
                metric_sums["metric/ade_best"] += ade_gathered.min(dim=-1).values.sum().item()
                metric_counts["metric/ade_best"] += ade_gathered.shape[0]
                metric_sums["metric/ade_mean"] += ade_gathered.mean(dim=-1).sum().item()
                metric_counts["metric/ade_mean"] += ade_gathered.shape[0] * ade_gathered.shape[-1]

            for k, v in corner_dict.items():
                gathered = gather_metric(v.float())
                if is_main:
                    metric_sums[f"metric/{k}"] += gathered.sum().item()
                    metric_counts[f"metric/{k}"] += gathered.numel()

            eval_count += 1

    # -------------------------------------------------------- merge JSONL
    if dist.is_initialized():
        dist.barrier()

    if is_main:
        with open(args.output_jsonl, "w") as out_f:
            for r in range(world_size):
                rpath = f"{args.output_jsonl}.rank_{r}"
                if os.path.exists(rpath):
                    with open(rpath) as in_f:
                        out_f.write(in_f.read())
                    os.remove(rpath)
        print(f"\n[eval_coc] Per-sample results written to {args.output_jsonl}")

    # -------------------------------------------------------- print summary
    if not is_main:
        cleanup_distributed()
        return

    print("\n" + "=" * 60)
    print(f"Stage {args.stage} CoC | {eval_count} samples | {args.checkpoint}")
    print("-" * 60)
    for key in sorted(metric_sums.keys()):
        if metric_counts[key] > 0:
            val = metric_sums[key] / metric_counts[key]
            print(f"  {key:<35} {val:.4f}")
    print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()

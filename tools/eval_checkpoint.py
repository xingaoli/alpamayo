"""Evaluate a trained Stage 1 or Stage 2 checkpoint on the eval dataset.

Usage (multi-GPU):
    torchrun --nproc_per_node=2 tools/eval_checkpoint.py --stage 2 --checkpoint ckpts/Alpamayo-1.5-10B

Usage (single-GPU):
    python tools/eval_checkpoint.py --stage 1 --checkpoint outputs/output_stage1/checkpoint-1000

By default only the final per-metric averages are printed. To inspect a single
batch in detail (shapes, GT, pred_xyz, per-candidate ADE), set DEBUG_EVAL=1.
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from hydra.utils import instantiate
from tqdm.auto import tqdm

from alpamayo_r1.common import logging

logging.setup_logging()
logger = logging.RankedLogger("eval", rank_zero_only=True)
logger.setLevel("INFO")

# ---------------------------------------------------------------------------
# 评测链路总览
# ---------------------------------------------------------------------------
# 0) torchrun 启动 N 个 rank,每个 rank 独立执行 main()
# 1) load_model_config() → 读 checkpoint_path/config.json 重构 ReasoningVLAConfig
#    (不读权重),把 vlm_name_or_path / traj_vocab_size / tokens_per_*_traj 等
#    训练时存下的字段重新装好,后面实例化 tokenizer / metric_runner 都要用
# 2) load_model(stage, ckpt, vlm)
#    stage==1 → TrainableReasoningVLA.from_alpamayo_checkpoint():只加载 vlm.* 权重
#       没有 flow-matching expert;轨迹通过 128 个 <i_k> 离散 token 表示,
#       token → xyz 由 DiscreteTrajectoryTokenizer.decode 完成
#       重要的兜底:如果 ckpt 漏存 vlm.lm_head.weight(weight tying 跳过),从
#       embed_tokens 重新绑定,否则 152064 维的原始 lm_head 跟 155697 维训过的
#       词表不匹配,推理会出乱码
#    stage==2 → TrainableAlpamayoR1.from_pretrained():加载完整 AR1 (VLM + expert)
# 3) 构造 PAIMultiSegmentDataset(generation_mode=True)
#    - 每个 clip 滑窗 20 个 t0 → 训练/eval 用同一函数
#    - vla_preprocess_args: components_order=["image","traj_history","prompt","traj_future"]
#      eval 时 user 段只问 "output the future trajectory",等模型生成
# 4) DataLoader → collate_fn 一次性 tokenize + padding
# 5) 每个 batch 走 MetricRunner:
#       ReasoningSampler  → VLM 自回归生成 → pred_xyz [B, N=1, K=6, T=64, 3]
#       DistanceMetrics   → minADE / ADE / corner_distance
# 6) gather_metric → all_gather 跨 rank 求平均 → 打印
# ---------------------------------------------------------------------------

DEBUG_EVAL = int(os.environ.get("DEBUG_EVAL", "0"))


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
    """Build ReasoningVLAConfig from checkpoint config.json without loading model weights.

    把训练时存到 config.json 里的所有超参重新装好,这样 eval 时能正确:
      - 实例化 DiscreteTrajectoryTokenizer (UnicycleAccelCurvatureActionSpace)
      - 知道 traj_vocab_size=4000, tokens_per_future_traj=128 等
    """
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
    """Load model for the given stage.

    stage==1 → TrainableReasoningVLA:只有 VLM,没有 diffusion expert,
              预测未来轨迹通过自回归生成 128 个 <i_k> 离散 token,
              再由 DiscreteTrajectoryTokenizer.decode → xyz。
    stage==2 → TrainableAlpamayoR1:在 stage1 之上接 flow-matching expert,
              cotrain_vlm=False 时 VLM 冻结,只训 expert。
    """
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
    """Gather metric tensors across all GPUs.

    每个 rank 算出来一个 [local_b, ...] 的 metric,这里 all_gather 拼成
    [world_b, ...],然后在 rank 0 上做 sum/num 求平均。
    """
    if not dist.is_initialized():
        return metric_tensor
    gathered = [torch.zeros_like(metric_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, metric_tensor)
    return torch.cat(gathered)


# ---------------------------------------------------------------------------
# 调试辅助
# ---------------------------------------------------------------------------
def _fmt(x, max_n=6, prec=4):
    """把张量/数组截短到前 max_n 个元素,方便打日志。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x).reshape(-1)[:max_n]
    return np.array2string(x, precision=prec, separator=", ")


def debug_print_batch(batch, model_config, output_batch, is_main, rank, batch_idx):
    """DEBUG_EVAL=1:每个 batch 打印关键值。

    内容:输入张量形状、GT 前 3 步、6 条候选 ADE、最佳/最差候选、逐点 L2 距离。
    """
    if not is_main:
        return
    img0 = batch["image_frames"][0]
    print(f"  [batch {batch_idx}] image_frames[0]: {tuple(img0.shape)} ({img0.shape[0]}cam x {img0.shape[1]}frames)")
    print(f"  [batch {batch_idx}] ego_history_xyz : {tuple(batch['ego_history_xyz'].shape)}  ego_future_xyz(GT): {tuple(batch['ego_future_xyz'].shape)}")
    gt_xyz0 = batch["ego_future_xyz"][0, -1, :3]
    print(f"  [batch {batch_idx}] GT xyz[0,:3] (m): {_fmt(gt_xyz0)}")

    if "pred_xyz" not in output_batch:
        return
    pred_xyz = output_batch["pred_xyz"]            # [B, N, K, T, 3]
    gt_xyz = batch["ego_future_xyz"][:, -1]        # [B, T, 3]
    B, N, K, T, _ = pred_xyz.shape
    print(f"  [batch {batch_idx}] pred_xyz.shape  : B={B} N={N} K={K} T={T}")
    print(f"  [batch {batch_idx}] pred[0,0,0,:3]  : {_fmt(pred_xyz[0, 0, 0, :3])}")
    diff = (pred_xyz - gt_xyz[:, None, None]).norm(dim=-1)[..., :2]
    per_candidate_ade = diff.mean(dim=-1)[0, 0]
    print(f"  [batch {batch_idx}] per-candidate ADE: {_fmt(per_candidate_ade, max_n=K)}")
    best = int(per_candidate_ade.argmin())
    worst = int(per_candidate_ade.argmax())
    print(f"  [batch {batch_idx}] best=#{best} ADE={per_candidate_ade[best]:.4f}  worst=#{worst} ADE={per_candidate_ade[worst]:.4f}")


def main():
    rank, world_size = setup_distributed()
    is_main = rank == 0

    parser = argparse.ArgumentParser(description="Evaluate Stage 1/2 checkpoint")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1 or 2")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_dir", type=str, default=os.getenv("ALPAMAYO_DATA_DIR", "data/PhysicalAI-Autonomous-Vehicles"))
    parser.add_argument("--eval_chunks", type=str, default="48-50")
    parser.add_argument("--vlm", type=str, default="ckpts/Qwen3-VL-8B-Instruct-config")
    parser.add_argument("--num_traj_samples", type=int, default=6, help="K, candidates per sample")
    parser.add_argument("--max_new_tokens", type=int, default=-1,
                        help="Max new tokens to generate. -1 = auto (stage1: 128, stage2: 256)")
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="-1 for all; if set, take first N segments")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # Auto-pick max_new_tokens by stage if user didn't override:
    #   stage 1 (VLM-only)   : 128 = 128 trajectory tokens + traj_future_end + im_end
    #   stage 2 (VLM+expert) : 256 = leaves headroom for the CoC reasoning text
    if args.max_new_tokens < 0:
        args.max_new_tokens = 128 if args.stage == 1 else 256

    # ---------------- Load model ----------------
    if is_main:
        print(f"[eval] stage={args.stage}  checkpoint={args.checkpoint}")
        print(f"[eval] K(num_traj_samples)={args.num_traj_samples}  "
              f"top_p={args.top_p}  temperature={args.temperature}  max_new_tokens={args.max_new_tokens}  "
              f"DEBUG_EVAL={DEBUG_EVAL}")
    model_config = load_model_config(args.checkpoint, args.vlm)
    model = load_model(args.stage, args.checkpoint, args.vlm)
    if is_main:
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[eval] Model loaded. total params={n_total:,}  "
              f"traj_vocab_size={model_config.traj_vocab_size}  "
              f"tok/hist={model_config.tokens_per_history_traj}  tok/fut={model_config.tokens_per_future_traj}")

    # ---------------- Load eval dataset ----------------
    # 关键:vla_preprocess_args 用 generation_mode=True,
    # user prompt 不会包含未来轨迹的答案,只给历史 + 图像,
    # 等模型自己生成 traj_future。
    vla_preprocess_args = {
        "_target_": "alpamayo_r1.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
        "components_order": ["image", "traj_history", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "label_components": ["traj_future"],
        "generation_mode": True,
    }
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
    total_size = len(eval_dataset)
    if args.max_eval_samples > 0 and args.max_eval_samples < total_size:
        eval_dataset = Subset(eval_dataset, list(range(args.max_eval_samples)))
        if is_main:
            print(f"[eval] dataset: {total_size} segments -> subset to {args.max_eval_samples}")
    elif is_main:
        print(f"[eval] dataset: {total_size} segments")

    # ---------------- Collate / sampler / metric runner ----------------
    collate_fn = instantiate(
        {
            "_target_": "alpamayo_r1.processor.qwen_processor.collate_fn_from_model_config",
            "_partial_": True,
        },
        model_config=model_config,
    )
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    if is_main:
        print(f"[eval] dataloader: {len(dataloader)} batches/rank  (batch_size={args.batch_size}, world={world_size})")

    # MetricRunner 包含两步:
    #   ReasoningSampler   → VLM 生成 K 条候选 → pred_xyz [B,1,K,64,3]
    #   DistanceMetrics    → minADE / ADE / corner_distance
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

    # ---------------- Run eval ----------------
    if is_main:
        print(f"Starting Stage {args.stage} evaluation on {world_size} GPU(s)...")
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    eval_count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), disable=not is_main, desc="Evaluating")
        for batch_idx, batch in enumerate(pbar):
            batch = to_device(batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch = {}
                metric_runner.run(model, batch, output_batch)

            if DEBUG_EVAL >= 1 and is_main:
                debug_print_batch(batch, model_config, output_batch, is_main, rank, batch_idx)

            for k, v in output_batch.items():
                if not k.startswith("metric/"):
                    continue
                gathered = gather_metric(v.float())
                if is_main:
                    metric_sums[k] += gathered.sum().item()
                    metric_counts[k] += gathered.numel()
            eval_count += 1

    # ---------------- Print results ----------------
    if not is_main:
        cleanup_distributed()
        return

    print("\n" + "=" * 60)
    print(f"Stage {args.stage} | Evaluated {eval_count} batches | Checkpoint: {args.checkpoint}")
    print("-" * 60)
    for key in sorted(metric_sums.keys()):
        if metric_counts[key] > 0:
            val = metric_sums[key] / metric_counts[key]
            print(f"  {key:<30} {val:.4f}")
    print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()

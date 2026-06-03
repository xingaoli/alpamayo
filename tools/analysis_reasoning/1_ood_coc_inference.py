#!/usr/bin/env python3
"""
CoC Inference for OOD Reasoning Clips — Dual-Model Parallel Version.

Loads two Alpamayo models on separate GPUs and runs CoC inference
in parallel on OOD reasoning clips (chunks 0-49 by default).
Output places predicted_coc from both models side-by-side with the
annotated CoC for easy comparison.

Model assignment:
  - GPU 0: model from .env (ALPAMAYO_MODEL_CKPT, default ckpts/Alpamayo-R1-10B)
  - GPU 1: hardcoded fallback (ckpts/Alpamayo-R1-10B-from-1.5)

Usage:
    python3 tools/analysis_reasoning/ood_coc_inference.py
    python3 tools/analysis_reasoning/ood_coc_inference.py --max-chunk 49
    python3 tools/analysis_reasoning/ood_coc_inference.py --resume
    python3 tools/analysis_reasoning/ood_coc_inference.py --debug       # process only 2 events
    python3 tools/analysis_reasoning/ood_coc_inference.py --chunk 0     # only chunk 0
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def load_env(env_path: str = ".env") -> dict:
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def build_task_list(data_dir: Path, max_chunk: int, chunk_only: int | None):
    """Build the list of inference tasks from ood_reasoning + clip_index."""
    ood_df = pd.read_parquet(data_dir / "reasoning" / "ood_reasoning.parquet")
    clip_index = pd.read_parquet(data_dir / "clip_index.parquet")

    ood_clip_ids = set(ood_df.index)
    clip_index_ood = clip_index[clip_index.index.isin(ood_clip_ids)]

    if chunk_only is not None:
        target = clip_index_ood[clip_index_ood["chunk"] == chunk_only]
    else:
        target = clip_index_ood[clip_index_ood["chunk"] <= max_chunk]

    tasks = []
    for clip_id in target.index:
        row = ood_df.loc[clip_id]
        events_str = row["events"]
        if not isinstance(events_str, str):
            continue
        try:
            events = json.loads(events_str)
        except (json.JSONDecodeError, TypeError):
            continue
        chunk = int(target.loc[clip_id, "chunk"])
        for evt in events:
            if not isinstance(evt, dict):
                continue
            ts = evt.get("event_start_timestamp")
            if ts is None or (isinstance(ts, float) and (ts != ts)):
                continue
            tasks.append(
                {
                    "clip_id": clip_id,
                    "chunk": chunk,
                    "event_cluster": row["event_cluster"],
                    "event_start_timestamp": int(ts),
                    "event_start_frame": evt.get("event_start_frame"),
                    "annotated_coc": evt.get("coc"),
                }
            )
    return tasks


# ---------------------------------------------------------------------------
# worker process — each receives its own task list (no shared queue)
# ---------------------------------------------------------------------------

def _worker(
    gpu_id: int,
    model_path: str,
    model_name: str,
    data_dir: str,
    tasks: list[dict],
    result_queue: mp.Queue,
):
    """
    Worker process: load model on *gpu_id*, iterate over *tasks*,
    and send results (plus a final None sentinel) to *result_queue*.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset_local import (
        load_physical_aiavdataset_local,
    )
    from alpamayo_r1 import helper

    # --- attempt to load model ---
    try:
        print(f"[{model_name}] GPU {gpu_id}: loading from {model_path} ...")
        model = AlpamayoR1.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda")
        processor = helper.get_processor(model.tokenizer)
        print(f"[{model_name}] GPU {gpu_id}: model loaded.")
    except Exception as e:
        print(f"[{model_name}] GPU {gpu_id}: FAILED to load model: {e}")
        for task in tasks:
            result_queue.put(
                {
                    "model_name": model_name,
                    "clip_id": task["clip_id"],
                    "event_start_timestamp": task["event_start_timestamp"],
                    "cot": None,
                    "error": f"model_load_failed: {e}",
                }
            )
        result_queue.put(None)
        return

    processed = 0
    for task in tasks:
        clip_id = task["clip_id"]
        ts_us = task["event_start_timestamp"]

        try:
            data = load_physical_aiavdataset_local(clip_id, data_dir, t0_us=ts_us)
            messages = helper.create_message(
                data["image_frames"].flatten(0, 1),
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
            model_inputs = helper.to_device(model_inputs, "cuda")
            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = (
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=1,
                        max_generation_length=256,
                        return_extra=True,
                    )
                )
            cot = extra["cot"][0]
            error = None
        except Exception as e:
            print(
                f"[{model_name}] inference failed "
                f"clip={clip_id} t0={ts_us}: {e}"
            )
            cot = None
            error = str(e)

        result_queue.put(
            {
                "model_name": model_name,
                "clip_id": clip_id,
                "event_start_timestamp": ts_us,
                "cot": cot,
                "error": error,
            }
        )
        processed += 1

    result_queue.put(None)
    print(f"[{model_name}] GPU {gpu_id}: finished {processed} tasks.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CoC Inference for OOD Reasoning Clips — Dual-Model Parallel"
    )
    parser.add_argument(
        "--model-path-0",
        type=str,
        default=None,
        help="Model path for GPU 0 (default: ALPAMAYO_MODEL_CKPT from .env "
        "or ckpts/Alpamayo-R1-10B)",
    )
    parser.add_argument(
        "--model-path-1",
        type=str,
        default="ckpts/Alpamayo-R1-10B-from-1.5",
        help="Model path for GPU 1 (hardcoded default: "
        "ckpts/Alpamayo-R1-10B-from-1.5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: ALPAMAYO_DATA_DIR from .env)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir/reasoning/ood_coc)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output, skipping already-completed events",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=49,
        help="Maximum chunk index to process (default: 49)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only first 2 events",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Process only a specific chunk",
    )
    args = parser.parse_args()

    # ---- resolve paths ----
    script_dir = Path(__file__).parent.parent.parent
    env_vars = load_env(script_dir / ".env")

    data_dir = args.data_dir or env_vars.get(
        "ALPAMAYO_DATA_DIR", "data/PhysicalAI-Autonomous-Vehicles"
    )
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir

    # output inside reasoning/ next to ood_reasoning.parquet
    output_dir = args.output_dir or str(data_dir / "reasoning" / "ood_coc")
    output_dir = Path(output_dir)

    def _resolve(p: str) -> str:
        pp = Path(p)
        if not pp.is_absolute():
            pp = script_dir / pp
        return str(pp)

    model_path_0 = args.model_path_0 or env_vars.get(
        "ALPAMAYO_MODEL_CKPT", "ckpts/Alpamayo-R1-10B"
    )
    model_path_1 = args.model_path_1

    model_name_0 = Path(model_path_0).name
    model_name_1 = Path(model_path_1).name

    model_path_0 = _resolve(model_path_0)
    model_path_1 = _resolve(model_path_1)

    print(f"Data directory:      {data_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"Model 0 (GPU 0):     {model_path_0}  ({model_name_0})")
    print(f"Model 1 (GPU 1):     {model_path_1}  ({model_name_1})")
    print(f"Resume:              {args.resume}")
    print(f"Max chunk:           {args.max_chunk}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- build task list ----
    print("\nLoading OOD reasoning data ...")
    tasks = build_task_list(data_dir, args.max_chunk, args.chunk)
    print(f"Total events to process: {len(tasks)}")

    if args.debug:
        tasks = tasks[:2]
        print("DEBUG MODE: only processing first 2 events")

    if not tasks:
        print("No tasks. Exiting.")
        return

    # ---- resume support ----
    completed_keys: set[tuple] = set()
    if args.resume:
        for fpath in sorted(output_dir.glob("ood_coc.chunk_*.json")):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for r in data.get("results", []):
                    if (
                        r.get(f"predicted_coc_{model_name_0}") is not None
                        and r.get(f"predicted_coc_{model_name_1}") is not None
                    ):
                        key = (r["clip_id"], r["event_start_timestamp"])
                        completed_keys.add(key)
            except Exception:
                pass
        print(f"Resume: {len(completed_keys)} already-completed events found.")

    tasks_to_run = [
        t
        for t in tasks
        if (t["clip_id"], t["event_start_timestamp"]) not in completed_keys
    ]
    print(f"Events remaining: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All events already completed. Nothing to do.")
        return

    # ---- launch workers (spawn — each gets its own CUDA context) ----
    mp.set_start_method("spawn", force=True)

    result_queue: mp.Queue = mp.Queue()

    workers = [
        mp.Process(
            target=_worker,
            args=(
                0, model_path_0, model_name_0, str(data_dir),
                tasks_to_run, result_queue,
            ),
            name="worker-0",
        ),
        mp.Process(
            target=_worker,
            args=(
                1, model_path_1, model_name_1, str(data_dir),
                tasks_to_run, result_queue,
            ),
            name="worker-1",
        ),
    ]

    for w in workers:
        w.start()

    # ---- collect results until both workers signal done ----
    raw_results: list[dict] = []
    done_count = 0
    pbar = tqdm(desc="Collecting results")
    while done_count < len(workers):
        res = result_queue.get()
        if res is None:
            done_count += 1
        else:
            raw_results.append(res)
            pbar.update(1)
    pbar.close()

    # ---- merge results: group by (clip_id, timestamp) ----
    print("Merging results from both models ...")
    merged: dict[tuple, dict] = {}

    for t in tasks:
        key = (t["clip_id"], t["event_start_timestamp"])
        merged[key] = {
            "clip_id": t["clip_id"],
            "chunk": t["chunk"],
            "event_cluster": t["event_cluster"],
            "event_start_timestamp": t["event_start_timestamp"],
            "event_start_frame": t["event_start_frame"],
            "annotated_coc": t["annotated_coc"],
        }

    for r in raw_results:
        key = (r["clip_id"], r["event_start_timestamp"])
        name = r["model_name"]
        if key not in merged:
            merged[key] = {
                "clip_id": r["clip_id"],
                "event_start_timestamp": r["event_start_timestamp"],
            }
        merged[key][f"predicted_coc_{name}"] = r.get("cot")
        if r.get("error"):
            merged[key][f"error_{name}"] = r["error"]

    all_results = list(merged.values())

    # ---- save per-chunk files ----
    results_by_chunk: dict[int, list] = {}
    for r in all_results:
        chunk = r.get("chunk", -1)
        results_by_chunk.setdefault(chunk, []).append(r)

    for chunk_id, chunk_results in sorted(results_by_chunk.items()):
        chunk_file = output_dir / f"ood_coc.chunk_{chunk_id:04d}.json"
        with open(chunk_file, "w") as f:
            json.dump(
                {
                    "chunk": chunk_id,
                    "num_results": len(chunk_results),
                    "inference_timestamp": datetime.now().isoformat(),
                    "results": chunk_results,
                },
                f,
                indent=2,
                cls=NumpyEncoder,
            )
        print(f"  Saved {len(chunk_results)} results -> {chunk_file}")

    # ---- summary ----
    def _cot_ok(r, name):
        val = r.get(f"predicted_coc_{name}")
        return val is not None and val != ""

    ok_0 = sum(1 for r in all_results if _cot_ok(r, model_name_0))
    ok_1 = sum(1 for r in all_results if _cot_ok(r, model_name_1))
    ok_both = sum(
        1 for r in all_results
        if _cot_ok(r, model_name_0) and _cot_ok(r, model_name_1)
    )

    print(f"\nSummary ({len(all_results)} events total):")
    print(f"  {model_name_0}: {ok_0}/{len(all_results)} successful")
    print(f"  {model_name_1}: {ok_1}/{len(all_results)} successful")
    print(f"  Both models:     {ok_both}/{len(all_results)}")

    # ---- save combined ----
    combined_file = output_dir / "ood_coc_all.json"
    with open(combined_file, "w") as f:
        json.dump(
            {
                "total_results": len(all_results),
                "inference_timestamp": datetime.now().isoformat(),
                "models": {
                    "model_0": model_name_0,
                    "model_1": model_name_1,
                },
                "success": {
                    model_name_0: ok_0,
                    model_name_1: ok_1,
                    "both": ok_both,
                },
                "results": all_results,
            },
            f,
            indent=2,
            cls=NumpyEncoder,
        )
    print(f"Combined results saved to {combined_file}")

    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()


if __name__ == "__main__":
    main()

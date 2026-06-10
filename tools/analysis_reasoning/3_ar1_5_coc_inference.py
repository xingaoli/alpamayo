#!/usr/bin/env python3
"""
AR1.5-only CoC inference on cached OOD reasoning samples (time-filtered).

Filters events from ood_reasoning.parquet by event_start_timestamp range
(default: 1.6s <= t <= 13.5s, in microseconds), then looks up each
(clip_id, timestamp) in the cached pkl files under data/pai_reasoning_cache/sample/
and runs Alpamayo-R1-10B-from-1.5 to produce a CoC.

Output: dict[clip_id][str(event_start_timestamp)] = {coc, coc_from_ar1_5, error}

Usage:
    python3 tools/analysis_reasoning/3_ar1_5_coc_inference.py
    python3 tools/analysis_reasoning/3_ar1_5_coc_inference.py --debug
    python3 tools/analysis_reasoning/3_ar1_5_coc_inference.py --resume
    python3 tools/analysis_reasoning/3_ar1_5_coc_inference.py --min-us 1600000 --max-us 13500000
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# build the (clip_id, timestamp) task list by time range from parquet
# ---------------------------------------------------------------------------

def build_tasks(
    ood_df: pd.DataFrame,
    min_us: int,
    max_us: int,
) -> list[dict]:
    """
    Enumerate events from ood_reasoning.parquet whose event_start_timestamp
    falls within [min_us, max_us] (inclusive).
    """
    tasks: list[dict] = []
    skipped_invalid = 0
    for clip_id, row in ood_df.iterrows():
        events_str = row.get("events")
        if not isinstance(events_str, str):
            skipped_invalid += 1
            continue
        try:
            events = json.loads(events_str)
        except (json.JSONDecodeError, TypeError):
            skipped_invalid += 1
            continue
        if not isinstance(events, list):
            skipped_invalid += 1
            continue
        for evt in events:
            if not isinstance(evt, dict):
                continue
            ts = evt.get("event_start_timestamp")
            if ts is None:
                continue
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                continue
            if ts_int < min_us or ts_int > max_us:
                continue
            tasks.append(
                {
                    "clip_id": clip_id,
                    "event_start_timestamp": ts_int,
                    "annotated_coc": evt.get("coc"),
                }
            )
    if skipped_invalid:
        print(f"  (skipped {skipped_invalid} clips with malformed `events`)")
    return tasks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AR1.5-only CoC inference on cached OOD reasoning samples (time-filtered)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/pai_reasoning_cache",
        help="Cache directory (default: data/pai_reasoning_cache)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ckpts/Alpamayo-R1-10B-from-1.5",
        help="Path to the AR1.5 model checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <cache_dir>/ar1_5_coc.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (clip_id, timestamp) pairs already present in the output JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only the first 2 events",
    )
    parser.add_argument(
        "--min-us",
        type=int,
        default=1_600_000,
        help="Min event_start_timestamp in microseconds (default: 1.6s = 1600000)",
    )
    parser.add_argument(
        "--max-us",
        type=int,
        default=13_500_000,
        help="Max event_start_timestamp in microseconds (default: 13.5s = 13500000)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent.parent
    env_vars = load_env(script_dir / ".env")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    sample_dir = cache_dir / "sample"
    ood_parquet = cache_dir / "ood_reasoning.parquet"
    output_path = Path(args.output or (cache_dir / "ar1_5_coc.json"))
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    model_path = str(model_path)

    print(f"Cache dir:   {cache_dir}")
    print(f"Model:       {model_path}")
    print(f"Output:      {output_path}")
    print(f"Time window: [{args.min_us} µs, {args.max_us} µs]  "
          f"({args.min_us / 1e6:.2f}s .. {args.max_us / 1e6:.2f}s)")

    # ---- load parquet & build task list by time range ----
    if not ood_parquet.exists():
        print(f"ERROR: {ood_parquet} not found", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoading {ood_parquet} ...")
    ood_df = pd.read_parquet(ood_parquet)
    print(f"  parquet rows: {len(ood_df)}")

    tasks = build_tasks(ood_df, args.min_us, args.max_us)
    print(f"Events in time window: {len(tasks)}")
    print(f"  unique clip_ids:     {len({t['clip_id'] for t in tasks})}")

    if args.debug:
        tasks = tasks[:2]
        print("DEBUG MODE: only first 2 events")

    if not tasks:
        print("No events in window. Exiting.")
        return

    # ---- build (clip_id, ts) -> pkl_path index lazily (only open pkls we need) ----
    # The pkl filename is the clip_id stem, so we can map clip_id -> pkl_path
    # without enumerating every file. But a clip_id's pkl may have multiple
    # timestamps; once we know the (clip_id, ts) we want, we open that one
    # pkl and look up ts in the dict.
    def pkl_path_for(clip_id: str) -> Path:
        return sample_dir / f"{clip_id}.pkl"

    # quick sanity check: warn about tasks whose pkl is missing
    missing_pkls = [t["clip_id"] for t in tasks if not pkl_path_for(t["clip_id"]).exists()]
    if missing_pkls:
        uniq = sorted(set(missing_pkls))
        print(f"  WARN: {len(missing_pkls)} events reference {len(uniq)} "
              f"pkl(s) not found in sample/ (will be skipped). "
              f"First few: {uniq[:3]}")

    # ---- resume support ----
    completed: set[tuple[str, int]] = set()
    if args.resume and output_path.exists():
        try:
            with open(output_path) as f:
                prev = json.load(f)
            for cid, by_ts in prev.items():
                for ts_str, entry in by_ts.items():
                    if entry.get("coc_from_ar1_5") is not None:
                        completed.add((cid, int(ts_str)))
            print(f"Resume: {len(completed)} events already inferred.")
        except Exception as e:
            print(f"WARN: failed to read existing output for resume: {e}")

    pending = [t for t in tasks if (t["clip_id"], t["event_start_timestamp"]) not in completed]
    print(f"Events to infer: {len(pending)}")
    if not pending:
        print("Nothing to do. Exiting.")
        return

    # ---- load model ----
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    print(f"Loading model from {model_path} on GPU {args.gpu} ...")
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.")

    # ---- run inference ----
    results_map: dict[tuple[str, int], dict] = {}

    # small cache so we don't re-open the same pkl many times
    pkl_cache: dict[str, dict] = {}

    for task in tqdm(pending, desc="AR1.5 inference"):
        clip_id = task["clip_id"]
        ts = task["event_start_timestamp"]

        coc_pred = None
        error = None
        try:
            pkl_path = pkl_path_for(clip_id)
            if not pkl_path.exists():
                raise FileNotFoundError(f"pkl not found: {pkl_path}")

            payload = pkl_cache.get(str(pkl_path))
            if payload is None:
                with open(pkl_path, "rb") as f:
                    payload = pickle.load(f)
                pkl_cache[str(pkl_path)] = payload

            # ts may be stored as int or as a stringified int
            inner = payload.get(ts)
            if inner is None:
                inner = payload.get(str(ts))
            if inner is None:
                raise KeyError(
                    f"timestamp {ts} not in {pkl_path.name} "
                    f"(available: {list(payload.keys())[:3]}...)"
                )

            data = inner["data"]
            image_frames = data["image_frames"]
            ego_history_xyz = data["ego_history_xyz"]
            ego_history_rot = data["ego_history_rot"]

            messages = helper.create_message(image_frames.flatten(0, 1))
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
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
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
            coc_pred = extra["cot"][0]
        except Exception as e:
            print(f"  WARN: inference failed clip={clip_id} t={ts}: {e}")
            error = str(e)

        results_map[(clip_id, ts)] = {
            "coc": task.get("annotated_coc"),
            "coc_from_ar1_5": coc_pred,
            "error": error,
        }

    # ---- merge with previously completed entries (resume) ----
    if args.resume and output_path.exists():
        try:
            with open(output_path) as f:
                prev = json.load(f)
            for cid, by_ts in prev.items():
                for ts_str, entry in by_ts.items():
                    key = (cid, int(ts_str))
                    if key not in results_map:
                        results_map[key] = entry
        except Exception:
            pass

    # ---- shape into nested dict[clip_id][ts_str] ----
    final: dict[str, dict[str, dict]] = {}
    for (cid, ts), entry in results_map.items():
        final.setdefault(cid, {})[str(ts)] = entry

    # ---- write output ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved {len(final)} clips / "
          f"{sum(len(v) for v in final.values())} events -> {output_path}")

    ok = sum(
        1
        for by_ts in final.values()
        for e in by_ts.values()
        if e.get("coc_from_ar1_5") is not None
    )
    total = sum(len(v) for v in final.values())
    err = total - ok
    print(f"Inference OK: {ok}/{total}  (errors: {err})")


if __name__ == "__main__":
    main()

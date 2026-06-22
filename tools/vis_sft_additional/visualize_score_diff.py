#!/usr/bin/env python3
"""
Visualize clips where ar1_5 sim_score is low but 10epoch sim_score is high.

Organized by category from category_index.json (test split). For each category:
  - If >=2 clips satisfy min_diff: randomly pick 2
  - If 1 clip satisfies min_diff: pick it + the highest-diff remaining clip
  - If 0 clips satisfy min_diff: pick the 2 highest-diff clips
  - If <2 clips total in category: visualize all available

For clips in chunk 0-49: decode front-wide frame from camera zip files.
For clips in chunk 50+: read from pai_reasoning_cache/sample pickle files.

Output: tools/vis_sft_additional/vis_output/
"""

import argparse
import io
import json
import os
import pickle
import random
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "PhysicalAI-Autonomous-Vehicles"
CACHE_DIR = PROJECT_ROOT / "data" / "pai_reasoning_cache" / "sample"
CAMERA_DIR = DATA_DIR / "camera" / "camera_front_wide_120fov"
CLIP_INDEX_PATH = DATA_DIR / "clip_index.parquet"
CATEGORY_INDEX_PATH = DATA_DIR / "coc_labels" / "additional" / "category_index.json"

AR1_SCORE_PATH = PROJECT_ROOT / "outputs" / "ar1_5_additional_score.jsonl"
E10_SCORE_PATH = PROJECT_ROOT / "outputs" / "output_coc_stage1_coc_eval_10epoch_score.jsonl"

OUTPUT_DIR = SCRIPT_DIR / "vis_output"


def load_data(split: str = "test"):
    """Load score data and category index, return per-category clip diffs (deduplicated)."""
    ar1_data = {}
    with open(AR1_SCORE_PATH) as f:
        for line in f:
            obj = json.loads(line)
            ar1_data[obj["clip_id"]] = obj

    e10_data = {}
    with open(E10_SCORE_PATH) as f:
        for line in f:
            obj = json.loads(line)
            e10_data[obj["clip_id"]] = obj

    with open(CATEGORY_INDEX_PATH) as f:
        cat_idx = json.load(f)

    cat_diffs: dict[str, list[tuple[str, dict, dict, float]]] = defaultdict(list)

    for cat, items in cat_idx[split].items():
        seen_cids: dict[str, tuple[dict, dict, float]] = {}
        for item in items:
            cid = item.split(":")[0]
            if cid in ar1_data and cid in e10_data:
                ar1 = ar1_data[cid]["sim_score"]
                e10 = e10_data[cid]["sim_score"]
                diff = e10 - ar1
                if cid not in seen_cids or diff > seen_cids[cid][2]:
                    seen_cids[cid] = (ar1_data[cid], e10_data[cid], diff)
        for cid, (ar1, e10, diff) in seen_cids.items():
            cat_diffs[cat].append((cid, ar1, e10, diff))

    return cat_diffs


def select_samples(
    cat_diffs: dict[str, list], min_diff: float, seed: int = 42
) -> list[tuple[str, str, dict, dict, float]]:
    """Select samples per category according to the selection rules."""
    rng = random.Random(seed)
    selected: list[tuple[str, str, dict, dict, float]] = []

    for cat in sorted(cat_diffs.keys()):
        items = cat_diffs[cat]
        items.sort(key=lambda x: x[3], reverse=True)

        above = [(cid, a, e, d) for cid, a, e, d in items if d >= min_diff]
        below = [(cid, a, e, d) for cid, a, e, d in items if d < min_diff]

        if len(above) >= 2:
            picks = rng.sample(above, 2)
        elif len(above) == 1:
            picks = [above[0]]
            if below:
                picks.append(below[0])
        else:
            picks = items[:2]

        for cid, ar1, e10, diff in picks:
            selected.append((cat, cid, ar1, e10, diff))

    return selected


def decode_from_zip(clip_id: str, chunk: int, timestamp_us: int) -> np.ndarray | None:
    """Decode front-wide frame from camera zip for chunk 0-49."""
    cam_feature = "camera_front_wide_120fov"
    zip_path = CAMERA_DIR / f"{cam_feature}.chunk_{chunk:04d}.zip"

    if not zip_path.exists():
        print(f"  WARN: zip not found: {zip_path}")
        return None

    try:
        from physical_ai_av import video

        with zipfile.ZipFile(zip_path, "r") as zf:
            video_bytes = io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.mp4"))
            ts_df = pq.read_table(
                io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.timestamps.parquet"))
            ).to_pandas()
            frame_timestamps = ts_df["timestamp"].values

            reader = video.SeekVideoReader(
                video_data=video_bytes, timestamps=frame_timestamps
            )
            frames, _ = reader.decode_images_from_timestamps(
                np.array([timestamp_us], dtype=np.int64)
            )
            reader.close()
        return frames[0]  # (H, W, 3) RGB
    except Exception as e:
        print(f"  WARN: failed to decode from zip for {clip_id}: {e}")
        return None


def decode_from_cache(clip_id: str, timestamp_us: int) -> np.ndarray | None:
    """Decode front-wide frame from pai_reasoning_cache pickle."""
    import torch

    cache_path = CACHE_DIR / f"{clip_id}.pkl"
    if not cache_path.exists():
        print(f"  WARN: cache not found: {cache_path}")
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        best_key = min(data.keys(), key=lambda k: abs(k - timestamp_us))
        entry = data[best_key]["data"]

        image_frames = entry["image_frames"]  # (4, 4, 3, H, W)
        abs_ts = entry["absolute_timestamps"]  # (4, 4)
        camera_indices = entry["camera_indices"]  # (4,)

        front_idx = None
        for i, ci in enumerate(camera_indices):
            if ci == 1:
                front_idx = i
                break
        if front_idx is None:
            print(f"  WARN: no front_wide camera in cache for {clip_id}")
            return None

        cam_ts = abs_ts[front_idx]
        frame_idx = int(torch.argmin(torch.abs(cam_ts - timestamp_us)))

        frame = image_frames[front_idx, frame_idx]  # (3, H, W)
        frame = frame.permute(1, 2, 0).numpy()  # (H, W, 3) RGB
        return frame
    except Exception as e:
        print(f"  WARN: failed to decode from cache for {clip_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def render_comparison(
    frame: np.ndarray,
    category: str,
    clip_id: str,
    chunk: int,
    ts: int,
    ar1_entry: dict,
    e10_entry: dict,
    diff: float,
    source: str,
) -> plt.Figure:
    """Render a comparison figure with frame and CoC text using matplotlib."""
    dpi = 100
    img_h, img_w = frame.shape[:2]
    img_w_in = img_w / dpi
    img_h_in = img_h / dpi

    margin = 0.15  # inches

    # Build text blocks
    lines: list[tuple[str, str, bool]] = []  # (label, body, is_highlight)
    lines.append(("Category", category, False))
    lines.append(("Clip ID", f"{clip_id}  (chunk={chunk}, source={source})", False))
    lines.append(("Timestamp", f"{ts} us", False))
    lines.append(("ar1_5_pretrain", ar1_entry.get("gen_cot", "(none)"), False))
    lines.append(("GT", ar1_entry.get("gt_cot", "(none)"), False))
    lines.append(("ar1_5_sft", e10_entry.get("gen_cot", "(none)"), False))
    lines.append(("Score Diff", f"ar1_5_pretrain={ar1_entry['sim_score']:.2f}  →  ar1_5_sft={e10_entry['sim_score']:.2f}  (Δ={diff:+.2f})", True))

    # Estimate text height: each label line + wrapped body lines
    wrap_chars = 80
    line_height_in = 0.45  # inches per line of text
    gap_in = 0.15  # gap between blocks
    total_text_h = 0.0
    for label, body, _ in lines:
        body_wrapped = textwrap.wrap(str(body), width=wrap_chars)
        total_text_h += line_height_in  # label
        total_text_h += len(body_wrapped) * line_height_in  # body
        total_text_h += gap_in  # gap

    fig_w = img_w_in + 2 * margin
    fig_h = margin + img_h_in + margin + total_text_h + margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Image axes
    ax_img = fig.add_axes([
        margin / fig_w,
        (margin + total_text_h + margin) / fig_h,
        img_w_in / fig_w,
        img_h_in / fig_h,
    ])
    ax_img.imshow(frame)
    ax_img.axis("off")

    # Text axes - one big text area below the image
    text_ax = fig.add_axes([
        margin / fig_w,
        margin / fig_h,
        img_w_in / fig_w,
        total_text_h / fig_h,
    ])
    text_ax.axis("off")
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(0, 1)

    y_pos = 1.0
    for label, body, is_highlight in lines:
        body_wrapped = textwrap.wrap(str(body), width=wrap_chars)
        # Label
        color = "#b42828" if is_highlight else "#3c3cb4"
        text_ax.text(0.0, y_pos, label, fontsize=28, fontweight="bold",
                     color=color, va="top", ha="left",
                     transform=text_ax.transAxes)
        y_pos -= line_height_in / total_text_h
        # Body
        for line_text in body_wrapped:
            text_ax.text(0.02, y_pos, line_text, fontsize=24,
                         color="#1e1e1e", va="top", ha="left",
                         transform=text_ax.transAxes)
            y_pos -= line_height_in / total_text_h
        y_pos -= gap_in / total_text_h

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize score diff clips by category")
    parser.add_argument("--min-diff", type=float, default=0.3,
                        help="Minimum sim_score diff for 'good' samples (default: 0.3)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--split", type=str, default="test",
                        help="Split in category_index.json (default: test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data (split={args.split})...")
    cat_diffs = load_data(split=args.split)

    # Print stats
    total_clips = sum(len(v) for v in cat_diffs.values())
    print(f"Categories: {len(cat_diffs)}, Total unique clips in scores: {total_clips}")
    for cat in sorted(cat_diffs.keys()):
        items = cat_diffs[cat]
        above = sum(1 for _, _, _, d in items if d >= args.min_diff)
        print(f"  {cat}: {len(items)} clips, {above} above min_diff={args.min_diff}")

    # Select samples
    selected = select_samples(cat_diffs, args.min_diff, seed=args.seed)
    print(f"\nSelected {len(selected)} samples across {len(cat_diffs)} categories")

    # Load clip_index
    print("Loading clip_index...")
    clip_df = pq.read_table(CLIP_INDEX_PATH).to_pandas()

    # Render
    idx = 0
    for cat, cid, ar1_entry, e10_entry, diff in tqdm(selected, desc="Rendering"):
        ts = ar1_entry["ts"]
        chunk = int(clip_df.loc[cid, "chunk"]) if cid in clip_df.index else -1

        if chunk < 50:
            frame = decode_from_zip(cid, chunk, ts)
            source = f"zip(chunk_{chunk:04d})"
        else:
            frame = decode_from_cache(cid, ts)
            source = f"cache(chunk_{chunk})"

        if frame is None:
            print(f"  SKIP [{cat}] {cid}: could not decode frame")
            continue

        fig = render_comparison(frame, cat, cid, chunk, ts, ar1_entry, e10_entry, diff, source)

        safe_cat = cat.replace("/", "_").replace(" ", "_")[:40]
        idx += 1
        out_name = f"{idx:03d}_{safe_cat}_{cid[:12]}_d{diff:.2f}.jpg"
        fig.savefig(output_dir / out_name, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved {out_name}")

    print(f"\nDone. {idx} images saved to {output_dir}")


if __name__ == "__main__":
    main()

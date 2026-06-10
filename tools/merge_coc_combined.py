"""Merge all coc_combined chunk JSONs into a single training file.

Each output entry is an independent sample with:
  - clip_id: source clip identifier
  - ts: timestamp in microseconds (frame_idx * 100000)
  - is_key: whether this frame is the keyframe
  - coc: the reasoning text (extracted from the nested list)

Only 7 frames are kept per keyframe: the keyframe itself and 3 frames
before/after at 0.5s intervals (keyframe ± {0, 5, 10, 15}).

Train/test split is by chunk: chunks 0-47 -> train, chunks 48-49 -> test.

Usage:
    python tools/merge_coc_combined.py \
        --input data/PhysicalAI-Autonomous-Vehicles/labels/coc_combined \
        --output data/PhysicalAI-Autonomous-Vehicles/coc_labels/normal
"""

import json
import os
import argparse
from pathlib import Path
from glob import glob

TRAIN_CHUNKS = set(range(48))   # 0-47
TEST_CHUNKS = set(range(48, 50))  # 48-49


def get_chunk_id(chunk_dir: str) -> int:
    """Extract chunk number from directory name like 'coc_combined.chunk_0042'."""
    name = os.path.basename(chunk_dir)
    return int(name.rsplit("_", 1)[-1])


def process_file(jf: str) -> list[dict]:
    """Extract samples from a single coc_combined JSON file."""
    samples = []
    with open(jf, "r") as f:
        data = json.load(f)

    clip_id = data["clip_id"]
    for kf_key, kf_val in data.items():
        if kf_key == "clip_id":
            continue
        coc_list = kf_val["coc_list"]
        keyframe_frame = int(kf_key)
        for frame_idx_str, text_list in coc_list.items():
            frame_idx = int(frame_idx_str)
            if abs(frame_idx - keyframe_frame) not in (0, 5, 10, 15):
                continue
            coc = text_list[0][0] if text_list and text_list[0] else ""
            samples.append({
                "clip_id": clip_id,
                "ts": frame_idx * 100000,
                "is_key": frame_idx == keyframe_frame,
                "coc": coc,
            })
    return samples


def merge_chunks(input_dir: str, output_dir: str):
    chunk_dirs = sorted(glob(os.path.join(input_dir, "coc_combined.chunk_*")))
    if not chunk_dirs:
        print(f"No chunk directories found in {input_dir}")
        return

    train_samples = []
    test_samples = []
    for chunk_dir in chunk_dirs:
        chunk_id = get_chunk_id(chunk_dir)
        json_files = glob(os.path.join(chunk_dir, "*.coc_combined.json"))
        for jf in json_files:
            samples = process_file(jf)
            if chunk_id in TRAIN_CHUNKS:
                train_samples.extend(samples)
            elif chunk_id in TEST_CHUNKS:
                test_samples.extend(samples)
            else:
                print(f"Warning: chunk {chunk_id} not in train or test set, skipping")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    for path, data_split in [(train_path, train_samples), (test_path, test_samples)]:
        with open(path, "w") as f:
            for s in data_split:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    train_key = sum(1 for s in train_samples if s["is_key"])
    test_key = sum(1 for s in test_samples if s["is_key"])
    print(f"Done:")
    print(f"  train: {len(train_samples)} samples (keyframes: {train_key}, non-key: {len(train_samples) - train_key}) -> {train_path}")
    print(f"  test:  {len(test_samples)} samples (keyframes: {test_key}, non-key: {len(test_samples) - test_key}) -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to coc_combined directory")
    parser.add_argument("--output", required=True, help="Output directory for train.jsonl and test.jsonl")
    args = parser.parse_args()
    merge_chunks(args.input, args.output)

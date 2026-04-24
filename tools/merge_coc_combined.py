"""Merge all coc_combined chunk JSONs into a single training file.

Each output entry is an independent sample with:
  - clip_id: source clip identifier
  - frame_idx: key from coc_list
  - keyframe_index: keyframe index within the clip
  - long_action: longitudinal action label
  - lat_action: lateral action label
  - text: the reasoning text (extracted from the nested list)

Usage:
    python tools/merge_coc_combined.py \
        --input data/PhysicalAI-Autonomous-Vehicles/labels/coc_combined \
        --output data/PhysicalAI-Autonomous-Vehicles/labels/coc_train/coc_combined_merged.jsonl
"""

import json
import os
import argparse
from pathlib import Path
from glob import glob


def merge_chunks(input_dir: str, output_path: str):
    chunk_dirs = sorted(glob(os.path.join(input_dir, "coc_combined.chunk_*")))
    if not chunk_dirs:
        print(f"No chunk directories found in {input_dir}")
        return

    samples = []
    total_files = 0
    for chunk_dir in chunk_dirs:
        json_files = glob(os.path.join(chunk_dir, "*.coc_combined.json"))
        for jf in json_files:
            total_files += 1
            with open(jf, "r") as f:
                data = json.load(f)

            clip_id = data["clip_id"]
            for kf_key, kf_val in data.items():
                if kf_key == "clip_id":
                    continue
                coc_list = kf_val["coc_list"]
                keyframe_index = kf_val["keyframe_index"]
                long_action = kf_val.get("long_action", "")
                lat_action = kf_val.get("lat_action", "")
                for frame_idx, text_list in coc_list.items():
                    text = text_list[0][0] if text_list and text_list[0] else ""
                    samples.append({
                        "clip_id": clip_id,
                        "frame_idx": int(frame_idx),
                        "keyframe_index": keyframe_index,
                        "long_action": long_action,
                        "lat_action": lat_action,
                        "text": text,
                    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Done: {total_files} files, {len(samples)} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to coc_combined directory")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()
    merge_chunks(args.input, args.output)

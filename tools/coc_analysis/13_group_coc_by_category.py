#!/usr/bin/env python3
"""
Group CoC samples by category from coc_train_category.json.
Outputs a JSON file mapping category ID to list of samples,
following the format of coc_action_cluster_results.json.

Usage:
    python tools/coc_analysis/13_group_coc_by_category.py
    python tools/coc_analysis/13_group_coc_by_category.py --input-file <path> --output-file <path>
"""

import json
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv


CATEGORIES = {
    1: "Slow for the lead vehicle",
    2: "Stop for the lead vehicle",
    3: "Stop for traffic light (TL) / traffic sign (TS)",
    4: "Resume at TL / TS",
    5: "Lane change (LC)",
    6: "Yield to VRUs",
    7: "Vehicle cut-in",
    8: "Speed bump",
    9: "Nudge",
    10: "Bypass construction objects",
    11: "Risky driving",
    12: "Curvy road",
    13: "Passing intersection",
    14: "No yield to VRUs",
    15: "Others",
}


def main():
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    default_data_dir = "data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    parser = argparse.ArgumentParser(description="Group CoC samples by category")
    parser.add_argument("--input-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_category.json"))
    parser.add_argument("--output-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_category_results.json"))
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by category
    groups = {}
    total = 0
    missing_cat = 0

    for sample_idx, item in enumerate(data):
        clip_id = item.get("clip_id", "unknown")
        coc_lists = item.get("coc_lists", {})
        category_lists = item.get("category_lists", {})

        for frame_idx, coc_text in coc_lists.items():
            cat_id = category_lists.get(frame_idx)
            if cat_id is None:
                cat_id = 15
                missing_cat += 1

            cat_key = str(cat_id)
            if cat_key not in groups:
                groups[cat_key] = []

            groups[cat_key].append({
                "sample_idx": sample_idx,
                "clip_id": clip_id,
                "frame_index": str(frame_idx),
                "source": "coc",
                "text": coc_text,
            })
            total += 1

    # Sort keys numerically
    sorted_groups = {k: groups[k] for k in sorted(groups.keys(), key=int)}

    # Save output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_groups, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Total CoC samples: {total}")
    if missing_cat:
        print(f"Missing category (defaulted to Others): {missing_cat}")
    print(f"\nCategory distribution:")
    print(f"{'ID':>3}  {'Count':>6}  Category")
    print(f"{'---':>3}  {'-----':>6}  --------")
    for cat_id, samples in sorted_groups.items():
        cat_name = CATEGORIES.get(int(cat_id), "Unknown")
        print(f"{cat_id:>3}  {len(samples):>6}  {cat_name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Classify COC sentences into pre-defined driving behavior categories using LLM.
Reads coc_train_change.json, classifies each CoC into one of 15 categories,
and saves to a new JSON file with category_lists.

Usage Examples:
    # 1. Dry run: only process first 5 items (for debugging)
    python tools/coc_analysis/12_classify_coc.py --dry-run --max-items 5

    # 2. Process a specific range
    python tools/coc_analysis/12_classify_coc.py --start-idx 0 --end-idx 100

    # 3. Process all
    python tools/coc_analysis/12_classify_coc.py
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
import re
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


def build_system_prompt() -> str:
    return (
        "You are an expert autonomous driving data annotator. "
        "Your task is to classify driving behaviors based on the provided description (CoC - Chain of Causation). "
        "Analyze the input sentence describing a driving behavior and its cause. "
        "Map this description to exactly one of the pre-defined categories listed below."
    )


def build_user_prompt(coc_sentence: str) -> str:
    prompt = f"""### Category Definitions
1. **Slow for the lead vehicle**: Decelerating due to the vehicle ahead.
2. **Stop for the lead vehicle**: Coming to a halt because of the vehicle ahead.
3. **Stop for traffic light (TL) / traffic sign (TS)**: Stopping due to red lights, stop signs, etc.
4. **Resume at TL / TS**: Starting to move again after a light/sign stop.
5. **Lane change (LC)**: Moving to a different lane.
6. **Yield to VRUs**: Yielding to Vulnerable Road Users (pedestrians, cyclists, etc.).
7. **Vehicle cut-in**: Reacting to a vehicle cutting into the ego lane.
8. **Speed bump**: Reacting to a speed bump or road hump.
9. **Nudge**: Moving away from the lane center to avoid/give space to an obstacle (without a full lane change).
10. **Bypass construction objects**: Going around construction zones or objects.
11. **Risky driving**: Decelerating, nudging, or reversing due to a risky event or unexpected obstacle.
12. **Curvy road**: Adjusting driving due to road curvature.
13. **Passing intersection**: Proceeding through an intersection (e.g., when clear).
14. **No yield to VRUs**: Ego has right of way, or VRUs yield to ego.
15. **Others**: Any scenario not fitting the above (e.g., normal driving, keeping lane, etc.).

### Few-Shot Examples
- Input: "Stop to yield to the cross-traffic truck" -> Output: 2
- Input: "Yield to the pedestrian since they are crossing at the crosswalk ahead" -> Output: 6
- Input: "Turn right at the intersection since cross-traffic has cleared" -> Output: 13
- Input: "Keep lane since the lane is clear ahead" -> Output: 15
- Input: "Slow down due to the vehicle ahead" -> Output: 1
- Input: "Stop at the red traffic light" -> Output: 3
- Input: "Start moving again as the traffic light turns green" -> Output: 4
- Input: "Change lane to the left because our right lane is ending" -> Output: 5
- Input: "Yield to the cyclist on the right" -> Output: 6
- Input: "Brake hard as a car suddenly cuts into our lane" -> Output: 7
- Input: "Slow down for the speed bump ahead" -> Output: 8
- Input: "Slightly steer left to give space to the parked car" -> Output: 9
- Input: "Move around the construction barrier on the right" -> Output: 10
- Input: "Reverse due to an unexpected obstacle blocking the road" -> Output: 11
- Input: "Reduce speed due to the sharp curve ahead" -> Output: 12
- Input: "Proceed through the intersection since it is clear" -> Output: 13
- Input: "Continue driving as the pedestrian waits for us to pass" -> Output: 14

### Output Rules
- You must output **ONLY** the integer ID of the category (e.g., "1", "6", "15").
- Do not output the category name.
- Do not output any explanation, punctuation, or extra text.

### Input Text
"{coc_sentence}"
"""
    return prompt


def call_llm(prompt: str, system_prompt: str, client: OpenAI, model: str = "default") -> int:
    """Call LLM and parse the category ID from response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=16,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result_text = response.choices[0].message.content.strip()

        # Remove thinking tags if present
        result_text = re.sub(r'<[^>]*>', '', result_text).strip()

        # Extract the integer from the response
        match = re.search(r'\b(\d{1,2})\b', result_text)
        if match:
            cat_id = int(match.group(1))
            if 1 <= cat_id <= 15:
                return cat_id
            else:
                print(f"  Warning: LLM returned out-of-range category ID: {cat_id} (raw: {result_text[:50]})")
                return 15  # fallback to Others

        print(f"  Warning: Could not parse category from LLM response: {result_text[:50]}")
        return 15  # fallback to Others

    except Exception as e:
        print(f"  Error: LLM call failed: {e}")
        return 15  # fallback to Others


def process_single_coc(coc_text: str, client: OpenAI, system_prompt: str,
                       dry_run: bool = False, model: str = "default") -> int:
    """Process single COC sentence and return category ID."""
    if dry_run:
        print(f"    [Dry Run] Processing: {coc_text[:80]}...")
        return 0

    return call_llm(build_user_prompt(coc_text), system_prompt, client, model)


def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")

    # Default paths
    default_data_dir = "/home/xingao/code/NVlabs-alpamayo/alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    parser = argparse.ArgumentParser(description="Classify COC sentences into driving behavior categories using LLM")
    parser.add_argument("--input-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_change.json"),
                        help="Input JSON file path")
    parser.add_argument("--output-file", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", "coc_train_category.json"),
                        help="Output JSON file path")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI API Key (default EMPTY)")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-FP8",
                        help="Model name to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode, do not call LLM")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max number of items to process (for debugging)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index for processing")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Save checkpoint every N items (default: 100)")

    args = parser.parse_args()

    print(f"Data directory: {data_dir}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    if args.max_items:
        print(f"Max items to process: {args.max_items}")
    if args.start_idx > 0 or args.end_idx:
        print(f"Processing range: {args.start_idx} to {args.end_idx or 'end'}")

    # Load input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    print(f"Loaded {len(input_data)} items from input file")

    # Slice data if needed
    start_idx = args.start_idx
    end_idx = args.end_idx or len(input_data)
    if args.max_items:
        end_idx = min(end_idx, start_idx + args.max_items)

    data_to_process = input_data[start_idx:end_idx]
    print(f"Processing {len(data_to_process)} items (from index {start_idx} to {end_idx})")

    # Initialize client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    system_prompt = build_system_prompt()

    # Process data
    output_data = []
    success_count = 0
    error_count = 0
    category_stats = {i: 0 for i in range(1, 16)}

    for idx, item in enumerate(tqdm(data_to_process, desc="Processing")):
        actual_idx = start_idx + idx

        output_item = {
            "chunk_id": item.get("chunk_id", "unknown"),
            "clip_id": item.get("clip_id", "unknown"),
            "ori_key_frame_long_action": item.get("ori_key_frame_long_action", ""),
            "ori_key_frame_lat_action": item.get("ori_key_frame_lat_action", ""),
            "coc_lists": item.get("coc_lists", {}),
            "category_lists": {}
        }

        coc_lists = item.get("coc_lists", {})

        for frame_idx, coc_text in coc_lists.items():
            try:
                cat_id = process_single_coc(coc_text, client, system_prompt, args.dry_run, args.model)

                output_item["category_lists"][frame_idx] = cat_id
                if not args.dry_run:
                    category_stats[cat_id] = category_stats.get(cat_id, 0) + 1

                success_count += 1
            except Exception as e:
                print(f"\n  Error processing frame {frame_idx} in item {actual_idx}: {e}")
                output_item["category_lists"][frame_idx] = 15
                error_count += 1

        output_data.append(output_item)

        # Save checkpoint periodically
        if (idx + 1) % args.batch_size == 0:
            checkpoint_path = args.output_file.replace('.json', f'_checkpoint_{actual_idx}.json')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n  Checkpoint saved: {checkpoint_path}")

    # Save final output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Output saved to: {args.output_file}")
    print(f"Total items processed: {len(output_data)}")
    print(f"Successful classifications: {success_count}")
    print(f"Errors: {error_count}")
    if not args.dry_run:
        print(f"\nCategory distribution:")
        for cat_id, count in sorted(category_stats.items()):
            if count > 0:
                print(f"  {cat_id:2d}. {CATEGORIES[cat_id]:45s} {count:5d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

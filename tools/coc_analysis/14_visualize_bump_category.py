#!/usr/bin/env python3
"""
Visualize bump category (category 8) clips with 21-frame extraction.

For each clip_id in the bump category, extracts 21 frames (±1s around frame_index)
and saves them as a grid image under vis_bump/.

Usage:
    python tools/coc_analysis/14_visualize_bump_category.py
"""

import json
import os
import zipfile
import io
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import physical_ai_av.video as video
from dotenv import load_dotenv


def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int,
                               timestamp_us: int, camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
    """Extract a frame from video at the specified timestamp."""
    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature,
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )

    if not os.path.exists(camera_zip_path):
        raise FileNotFoundError(f"Camera zip not found: {camera_zip_path}")

    with zipfile.ZipFile(camera_zip_path, 'r') as zf:
        video_data = io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.mp4"))

        timestamps_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.timestamps.parquet"))
        )
        frame_timestamps = timestamps_df["timestamp"].values

        reader = video.SeekVideoReader(
            video_data=video_data,
            timestamps=frame_timestamps,
        )

        target_timestamps = np.array([timestamp_us], dtype=np.int64)
        frames, _ = reader.decode_images_from_timestamps(target_timestamps)

        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")

        return frames[0]


def build_clip_to_chunk_map(coc_train_path: Path) -> dict:
    """Build a mapping from clip_id to (chunk_id, chunk_number)."""
    with open(coc_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    clip_to_chunk = {}
    for entry in train_data:
        clip_id = entry['clip_id']
        chunk_id = entry['chunk_id']
        chunk_number = int(chunk_id.split('_')[1])
        clip_to_chunk[clip_id] = (chunk_id, chunk_number)
    return clip_to_chunk


def draw_text_on_image(image: np.ndarray, text: str, position: tuple = (10, 30),
                       font_scale: float = 0.6, thickness: int = 2,
                       text_color: tuple = (255, 255, 255),
                       bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Draw text with background on image."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color_bgr = tuple(text_color[::-1])
    bg_color_bgr = tuple(bg_color[::-1])

    # Split long text into multiple lines
    max_chars_per_line = 50
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_chars_per_line:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(' '.join(current_line))

    x, y = position
    line_height = 25

    for i, line in enumerate(lines):
        y_pos = y + i * line_height
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, thickness
        )
        cv2.rectangle(
            img_bgr,
            (x - 5, y_pos - text_height - 5),
            (x + text_width + 5, y_pos + baseline + 5),
            bg_color_bgr,
            -1
        )
        cv2.putText(
            img_bgr, line, (x, y_pos),
            font, font_scale, text_color_bgr, thickness
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def process_clip(clip_id: str, frame_index: int, text: str,
                 chunk_number: int, data_dir: str, clip_output_dir: Path):
    """Extract 21 frames (±1s, 10fps => ±10 frames) around frame_index, save each individually."""
    camera_feature = "camera_front_wide_120fov"
    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature,
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )

    if not os.path.exists(camera_zip_path):
        print(f"  Warning: Camera zip not found for chunk {chunk_number}: {camera_zip_path}")
        return

    clip_output_dir.mkdir(parents=True, exist_ok=True)

    # 10fps: each frame = 0.1s, ±1s => ±10 frames, total 21 frames
    offset_range = list(range(-10, 11))  # [-10, -9, ..., 0, ..., 9, 10]

    with zipfile.ZipFile(camera_zip_path, 'r') as zf:
        video_data = io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.mp4"))

        timestamps_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.timestamps.parquet"))
        )
        frame_timestamps = timestamps_df["timestamp"].values

        reader = video.SeekVideoReader(
            video_data=video_data,
            timestamps=frame_timestamps,
        )

        for offset in offset_range:
            target_frame_idx = frame_index + offset
            target_timestamp_us = int(target_frame_idx / 10 * 1_000_000)

            try:
                target_ts = np.array([target_timestamp_us], dtype=np.int64)
                decoded_frames, _ = reader.decode_images_from_timestamps(target_ts)

                if decoded_frames.shape[0] > 0:
                    frame_img = decoded_frames[0]

                    # Draw frame info
                    label = f"frame={target_frame_idx}"
                    if offset == 0:
                        label += f" [KEY] {text}"
                    frame_img = draw_text_on_image(
                        frame_img, label,
                        position=(10, 30),
                        font_scale=0.5, thickness=1,
                        text_color=(0, 255, 0),
                        bg_color=(0, 0, 0)
                    )

                    # Save each frame individually, named by frame index
                    output_path = clip_output_dir / f"frame_{target_frame_idx:04d}.jpg"
                    img_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception:
                continue


def main():
    # Load environment
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)

    data_dir = os.getenv(
        "ALPAMAYO_DATA_DIR",
        "data/PhysicalAI-Autonomous-Vehicles"
    )

    # Paths
    category_json = Path(data_dir) / "labels" / "coc_train" / "coc_category_results.json"
    coc_train_json = Path(data_dir) / "labels" / "coc_train" / "coc_train_change.json"
    output_dir = Path(data_dir) / "labels" / "coc_train" / "vis_bump"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load category 8 data
    with open(category_json, 'r', encoding='utf-8') as f:
        category_data = json.load(f)

    bump_entries = category_data.get("8", [])
    print(f"Category 8 (bump): {len(bump_entries)} entries")

    # Build clip_id -> chunk mapping
    print("Building clip-to-chunk mapping...")
    clip_to_chunk = build_clip_to_chunk_map(coc_train_json)

    # Deduplicate by (clip_id, frame_index) to avoid processing same frame multiple times
    seen = set()
    unique_entries = []
    for entry in bump_entries:
        key = (entry['clip_id'], entry['frame_index'])
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    print(f"Unique (clip_id, frame_index) pairs: {len(unique_entries)}")

    # Process each entry
    success = 0
    failed = 0
    not_found = 0

    for entry in tqdm(unique_entries, desc="Processing bump clips"):
        clip_id = entry['clip_id']
        frame_index = int(entry['frame_index'])
        text = entry['text']

        if clip_id not in clip_to_chunk:
            print(f"  Warning: clip_id {clip_id} not found in coc_train_change.json")
            not_found += 1
            continue

        chunk_id, chunk_number = clip_to_chunk[clip_id]

        clip_output_dir = output_dir / clip_id
        try:
            process_clip(clip_id, frame_index, text, chunk_number, data_dir, clip_output_dir)
            success += 1
        except Exception as e:
            print(f"  Error processing {clip_id} frame {frame_index}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done! Results:")
    print(f"  Success:  {success}")
    print(f"  Failed:   {failed}")
    print(f"  Not found (clip not in train): {not_found}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

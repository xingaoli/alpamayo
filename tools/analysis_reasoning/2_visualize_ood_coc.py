#!/usr/bin/env python3
"""
Visualize OOD CoC inference results.

For each event where both models produced a valid CoC, decode the camera
frame at the trigger timestamp and render all three CoCs below the image.

Output: data_dir/reasoning/ood_coc_vis/

Usage:
    python3 tools/analysis_reasoning/visualize_ood_coc.py
    python3 tools/analysis_reasoning/visualize_ood_coc.py --max-samples 10
"""
import os
import argparse
import io
import json
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import physical_ai_av.video as video


# ---- helpers ----

def _find_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a CJK-capable font available on the system."""
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, 18)
    return ImageFont.load_default()


def _extract_cot(cot_val) -> str:
    """Unpack model CoC from its storage format."""
    if cot_val is None:
        return "(none)"
    if isinstance(cot_val, list):
        parts = []
        for item in cot_val:
            if isinstance(item, list):
                parts.extend(str(s) for s in item)
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return str(cot_val)


def _render_single(
    clip_id: str,
    chunk: int,
    timestamp_us: int,
    annotated_coc: str,
    coc_0: str,
    coc_1: str,
    model_name_0: str,
    model_name_1: str,
    data_dir: str,
) -> Image.Image | None:
    """Decode front-wide frame and compose the comparison image."""
    cam_feature = "camera_front_wide_120fov"
    camera_zip_path = os.path.join(
        data_dir, "camera", cam_feature,
        f"{cam_feature}.chunk_{chunk:04d}.zip",
    )

    try:
        with zipfile.ZipFile(camera_zip_path, "r") as zf:
            video_bytes = io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.mp4"))
            ts_df = pd.read_parquet(
                io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.timestamps.parquet"))
            )
            frame_timestamps = ts_df["timestamp"].values

            reader = video.SeekVideoReader(
                video_data=video_bytes, timestamps=frame_timestamps
            )
            frames, _ = reader.decode_images_from_timestamps(
                np.array([timestamp_us], dtype=np.int64)
            )
            reader.close()
    except Exception as e:
        print(f"  WARN: failed to decode frame for {clip_id} t={timestamp_us}: {e}")
        return None

    frame = Image.fromarray(frames[0])  # (H, W, C) RGB

    # ---- build composite ----
    font = _find_font()
    font_small = font.font_variant(size=14) if hasattr(font, "font_variant") else font

    img_w, img_h = frame.size
    margin = 20
    line_h = font.size + 4
    label_color = (60, 60, 180)
    text_color = (30, 30, 30)

    sections = [
        ("[Annotated]", annotated_coc),
        (f"[{model_name_0}]", coc_0),
        (f"[{model_name_1}]", coc_1),
    ]

    # measure total text height
    wrap_width_chars = max(60, img_w // 8)
    total_text_h = 0
    wrapped_lines: list[tuple[str, str, list[str]]] = []
    for label, body in sections:
        body_wrapped = textwrap.wrap(body or "(none)", width=wrap_width_chars)
        wrapped_lines.append((label, body, body_wrapped))
        total_text_h += line_h + len(body_wrapped) * line_h + 8

    canvas_h = img_h + margin + total_text_h + margin
    canvas = Image.new("RGB", (max(img_w, 800), canvas_h), (255, 255, 255))

    # paste image
    canvas.paste(frame, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = img_h + margin

    for label, body, body_wrapped in wrapped_lines:
        # label
        draw.text((margin, y), label, fill=label_color, font=font)
        y += line_h
        # body
        for line in body_wrapped:
            draw.text((margin + 10, y), line, fill=text_color, font=font_small)
            y += line_h
        y += 8  # gap between sections

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OOD CoC inference results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/PhysicalAI-Autonomous-Vehicles",
        help="Data directory",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to ood_coc_all.json (default: data_dir/reasoning/ood_coc/ood_coc_all.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir/reasoning/ood_coc_vis)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to visualize (0 = all)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = script_dir / data_dir

    results_json = args.results_json or str(
        data_dir / "reasoning" / "ood_coc" / "ood_coc_all.json"
    )
    output_dir = Path(args.output_dir or str(data_dir / "reasoning" / "ood_coc_vis"))

    print(f"Results JSON: {results_json}")
    print(f"Output dir:   {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_json) as f:
        data = json.load(f)

    models = data.get("models", {})
    model_0 = models.get("model_0", "model_0")
    model_1 = models.get("model_1", "model_1")

    key_0 = f"predicted_coc_{model_0}"
    key_1 = f"predicted_coc_{model_1}"

    results = data["results"]
    both_ok = [
        r for r in results
        if r.get(key_0) and r.get(key_1)
    ]
    print(f"Events with both models OK: {len(both_ok)}/{len(results)}")

    if args.max_samples > 0:
        both_ok = both_ok[: args.max_samples]
        print(f"Limiting to {len(both_ok)} samples")

    for i, r in enumerate(tqdm(both_ok, desc="Rendering")):
        coc_0 = _extract_cot(r[key_0])
        coc_1 = _extract_cot(r[key_1])
        annotated = r.get("annotated_coc", "(none)")

        img = _render_single(
            clip_id=r["clip_id"],
            chunk=r["chunk"],
            timestamp_us=r["event_start_timestamp"],
            annotated_coc=annotated,
            coc_0=coc_0,
            coc_1=coc_1,
            model_name_0=model_0,
            model_name_1=model_1,
            data_dir=str(data_dir),
        )

        if img is None:
            continue

        out_name = f"{r['clip_id'][:8]}_{r['event_start_timestamp']}.jpg"
        img.save(output_dir / out_name, quality=92)

    print(f"\nDone. Images saved to {output_dir}")


if __name__ == "__main__":
    main()

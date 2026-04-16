#!/usr/bin/env python3
"""
Visualize clustered factor samples.

This script:
1. Reads coc_factor_cluster_results.json
2. Randomly selects clusters and samples from each cluster
3. Decodes frames from video based on clip_id and frame_index
4. Draws the factor text on each frame
5. Saves 4 frames per cluster in a single figure

Usage Examples:
    # 1. Visualize 10 random clusters
    python tools/coc_analysis/11_visualize_cluster_samples.py --num-clusters 10

    # 2. Visualize specific clusters
    python tools/coc_analysis/11_visualize_cluster_samples.py --clusters 0,1,2,5

    # 3. Customize samples per cluster
    python tools/coc_analysis/11_visualize_cluster_samples.py --num-clusters 5 --samples-per-cluster 4
"""

import json
import os
import zipfile
import io
import random
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import physical_ai_av.video as video
from dotenv import load_dotenv


def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int,
                               timestamp_us: int, camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
    """
    Extract a frame from video at the specified timestamp.
    """
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


def get_chunk_number_for_clip(data_dir: str, clip_id: str, camera_feature: str = "camera_front_wide_120fov") -> int:
    """
    Find which chunk file contains the given clip_id.
    Returns -1 if not found.
    """
    camera_dir = os.path.join(data_dir, "camera", camera_feature)
    if not os.path.exists(camera_dir):
        return -1
    
    zip_files = [f for f in os.listdir(camera_dir) if f.endswith('.zip')]
    
    for zip_file in sorted(zip_files):
        chunk_path = os.path.join(camera_dir, zip_file)
        try:
            with zipfile.ZipFile(chunk_path, 'r') as zf:
                namelist = zf.namelist()
                matching = [n for n in namelist if n.startswith(clip_id + ".")]
                if matching:
                    chunk_num = int(zip_file.split('_')[-1].split('.')[0])
                    return chunk_num
        except Exception:
            continue
    
    return -1


def draw_text_on_image(image: np.ndarray, text: str, position: tuple = (10, 30),
                       font_scale: float = 0.6, thickness: int = 2,
                       text_color: tuple = (255, 255, 255),
                       bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    Draw text with background on image.
    """
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

    # Draw each line
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
            img_bgr,
            line,
            (x, y_pos),
            font,
            font_scale,
            text_color_bgr,
            thickness
        )

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def visualize_cluster(cluster_id: str, cluster_samples: list, data_dir: str, 
                      output_dir: str, samples_per_cluster: int = 4):
    """
    Visualize samples from a single cluster.
    """
    # Randomly select samples
    if len(cluster_samples) < samples_per_cluster:
        selected_samples = cluster_samples
    else:
        selected_samples = random.sample(cluster_samples, samples_per_cluster)
    
    fig, axes = plt.subplots(1, samples_per_cluster, figsize=(samples_per_cluster * 5, 5))
    if samples_per_cluster == 1:
        axes = [axes]
    
    success_count = 0
    
    for i, sample in enumerate(selected_samples):
        clip_id = sample['clip_id']
        frame_index = int(sample['frame_index'])
        factor_text = sample['text']
        
        try:
            # Find chunk number for this clip
            chunk_number = get_chunk_number_for_clip(data_dir, clip_id)
            if chunk_number == -1:
                print(f"    Warning: Could not find chunk for clip {clip_id}")
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"Chunk not found\n{clip_id[:8]}...", 
                            ha='center', va='center', fontsize=10)
                continue
            
            # Decode frame (frame_index/10 * 1_000_000 us)
            timestamp_us = int(frame_index / 10 * 1_000_000)
            image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
            
            # Draw factor text on image
            image_with_text = draw_text_on_image(
                image,
                f"Factor: {factor_text}",
                position=(20, 50),
                font_scale=0.5,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=(0, 100, 0)
            )
            
            axes[i].imshow(image_with_text)
            axes[i].set_title(f"sample_idx={sample['sample_idx']}\nframe={frame_index}", 
                             fontsize=9)
            axes[i].axis('off')
            success_count += 1
            
        except Exception as e:
            print(f"    Error processing sample {sample['sample_idx']}: {e}")
            axes[i].axis('off')
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                        ha='center', va='center', fontsize=8)
    
    # Set main title
    plt.suptitle(f"Cluster {cluster_id} ({len(cluster_samples)} samples)", 
                fontsize=14, fontweight='bold')
    
    # Save figure
    output_filename = f"cluster_{cluster_id}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return success_count


def main():
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", 
                         "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles")
    
    parser = argparse.ArgumentParser(description="Visualize factor cluster samples")
    parser.add_argument("--input-file", type=str, 
                        default=os.path.join(data_dir, "labels", "coc_train", 
                                           "coc_factor_cluster_results.json"),
                        help="Path to cluster results JSON")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(data_dir, "labels", "coc_train", 
                                           "vis_cluster"),
                        help="Output directory for visualizations")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Base data directory")
    parser.add_argument("--num-clusters", type=int, default=10,
                        help="Number of random clusters to visualize")
    parser.add_argument("--clusters", type=str, default=None,
                        help="Specific clusters to visualize (comma-separated IDs)")
    parser.add_argument("--samples-per-cluster", type=int, default=4,
                        help="Number of samples to show per cluster")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load cluster results
    print(f"Loading cluster results from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        cluster_results = json.load(f)
    
    print(f"Total clusters: {len(cluster_results)}")
    print(f"Total samples: {sum(len(v) for v in cluster_results.values())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Determine which clusters to visualize
    if args.clusters:
        target_clusters = [c.strip() for c in args.clusters.split(',')]
    else:
        # Random selection
        cluster_ids = list(cluster_results.keys())
        target_clusters = random.sample(cluster_ids, 
                                       min(args.num_clusters, len(cluster_ids)))
    
    print(f"\nVisualizing {len(target_clusters)} clusters: {sorted(target_clusters, key=lambda x: int(x))}")
    
    # Process each cluster
    success_total = 0
    for cluster_id in tqdm(target_clusters, desc="Clusters"):
        cluster_samples = cluster_results[cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_samples)} samples")
        
        success_count = visualize_cluster(
            cluster_id, 
            cluster_samples, 
            args.data_dir, 
            args.output_dir,
            samples_per_cluster=args.samples_per_cluster
        )
        success_total += success_count
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Successfully processed: {success_total} samples")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

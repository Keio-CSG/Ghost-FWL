import argparse
import re
import time
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from src.utils import upsample_histogram_direction_zero_padding
from src.visualize.fwl_lib.pointcloud_wrapper import PointcloudWrapper
from src.visualize.vis_pcd import (
    count_and_print_classes,
    create_annotation_map,
    crop_annotation_volume,
    generate_pointcloud_with_open3d,
    load_annotation_volume,
    visualize_open3d,
)


def parse_prediction_filename(pred_path: str) -> dict:
    """Parse scene, hist, and rest from a prediction filename.

    Example: scene001_hist002_20250909..._000049_prediction_voxel.b2
    -> scene=scene001, hist=hist002, rest=20250909..._000049
    """
    name = Path(pred_path).name
    m = re.match(r"^(.+?)_(hist\d+)_(.+)_prediction_voxel\.b2$", name)
    if not m:
        raise ValueError(f"Cannot parse prediction filename: {name}")
    return {"scene": m.group(1), "hist": m.group(2), "rest": m.group(3)}


def resolve_data_file(ghost_dataset: str, info: dict) -> Optional[str]:
    """Resolve data file: {ghost_dataset}/{scene}/data/{hist}/{rest}_voxel.b2"""
    p = Path(ghost_dataset) / info["scene"] / "data" / info["hist"] / f"{info['rest']}_voxel.b2"
    return str(p) if p.exists() else None


def resolve_annotation_file(ghost_dataset: str, info: dict, expand: bool) -> Optional[str]:
    """Resolve annotation file under ghost_dataset.

    Auto-detects annotation_v{N}[_expand] directories for the scene.
    Uses the highest version number found.

    Path: {ghost_dataset}/{scene}/annotation_v{N}[_expand]/{hist}/{rest}_annotation_voxel.b2
    """
    scene_dir = Path(ghost_dataset) / info["scene"]
    if not scene_dir.exists():
        return None

    suffix = "_expand" if expand else ""
    pattern = re.compile(r"^annotation_v(\d+)" + re.escape(suffix) + r"$")

    best_version = -1
    best_path = None
    for d in scene_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        p = d / info["hist"] / f"{info['rest']}_annotation_voxel.b2"
        if p.exists() and int(m.group(1)) > best_version:
            best_version = int(m.group(1))
            best_path = str(p)

    return best_path


def process_file(
    pred_path: str,
    ghost_dataset: str,
    output_dir: str,
    config: dict,
) -> None:
    """Process a single prediction file."""
    info = parse_prediction_filename(pred_path)
    print(f"\n[INFO] scene={info['scene']}, hist={info['hist']}")

    # Resolve data file
    data_path = resolve_data_file(ghost_dataset, info)
    if not data_path:
        print(f"[ERROR] Data file not found for {pred_path}")
        return
    print(f"[INFO] data: {data_path}")

    # Resolve annotation file (auto-detect highest version)
    expand = config.get("expand", False)
    ann_path = resolve_annotation_file(ghost_dataset, info, expand)
    if ann_path:
        print(f"[INFO] annotation: {ann_path}")

    # Load histogram data
    bin_offset = config.get("bin_offset", 0)
    pcw = PointcloudWrapper(data_path, mode="SPD", spd_mode=15, histo_offset=bin_offset)

    x_range = config.get("x_range")
    y_range = config.get("y_range")
    if x_range and y_range:
        pcw.crop_xy(x_range[0], x_range[1], y_range[0], y_range[1])

    x_min = int(pcw.df["//Pixel_X"].min())
    x_max = int(pcw.df["//Pixel_X"].max())
    y_min = int(pcw.df["Pixel_Y"].min())
    y_max = int(pcw.df["Pixel_Y"].max())
    data_width = x_max - x_min + 1
    data_height = y_max - y_min + 1

    # --- Process prediction ---
    pred_volume = load_annotation_volume(pred_path)
    original_depth = config.get("original_depth")
    if config.get("upsample") and original_depth and pred_volume.shape[2] != original_depth:
        print(f"[INFO] Upsampling prediction Z: {pred_volume.shape[2]} -> {original_depth}")
        pred_volume = upsample_histogram_direction_zero_padding(pred_volume, original_depth)

    pred_map = create_annotation_map(
        pred_volume,
        data_width=data_width,
        data_height=data_height,
        x_offset=config.get("x_offset"),
        y_offset=config.get("y_offset"),
        x_min=x_min,
        y_min=y_min,
    )
    count_and_print_classes(pred_map)

    pcd = generate_pointcloud_with_open3d(
        pcw,
        pred_map,
        show_noise=config.get("show_noise", False),
        show_undefined=config.get("show_undefined", False),
        remove_ghost_neighbors_radius=config.get("remove_ghost_neighbors_radius"),
        ghost_removal_mode=config.get("ghost_removal_mode", "none"),
        color_mode=config.get("color_mode", "label"),
    )

    pred_stem = Path(pred_path).stem
    out_path = Path(output_dir) / f"{pred_stem}.pcd"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_open3d(pcd, str(out_path))

    # --- Process GT annotation ---
    if ann_path:
        gt_volume = load_annotation_volume(ann_path)
        if x_range and y_range:
            gt_volume = crop_annotation_volume(gt_volume, x_range, y_range, histo_offset=bin_offset)

        gt_map = create_annotation_map(
            gt_volume,
            data_width=data_width,
            data_height=data_height,
            x_offset=config.get("x_offset"),
            y_offset=config.get("y_offset"),
            x_min=x_min,
            y_min=y_min,
        )
        count_and_print_classes(gt_map)

        gt_pcd = generate_pointcloud_with_open3d(
            pcw,
            gt_map,
            show_noise=config.get("show_noise", False),
            show_undefined=config.get("show_undefined", False),
            ghost_removal_mode="none",
            color_mode=config.get("color_mode", "label"),
        )

        gt_out = Path(output_dir) / f"{pred_stem}_gt.pcd"
        visualize_open3d(gt_pcd, str(gt_out))


def main() -> None:
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Batch visualize prediction point clouds.")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--pred_dir", type=str, help="Directory with *_prediction_voxel.b2 files.")
    parser.add_argument("--ghost_dataset", type=str, help="Root of ghost_dataset.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    args, _ = parser.parse_known_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides config
    if args.pred_dir:
        config["pred_dir"] = args.pred_dir
    if args.ghost_dataset:
        config["ghost_dataset"] = args.ghost_dataset
    if args.output_dir:
        config["output_dir"] = args.output_dir

    pred_dir = config.get("pred_dir")
    ghost_dataset = config.get("ghost_dataset")
    output_dir = config.get("output_dir", "./output")

    if not pred_dir or not ghost_dataset:
        parser.error("pred_dir and ghost_dataset are required (via CLI or config).")

    # Find prediction files
    pred_files = sorted(Path(pred_dir).glob("*_prediction_voxel.b2"))
    print(f"[INFO] Found {len(pred_files)} prediction files in {pred_dir}")

    processed = 0
    failed = 0
    for pred_file in tqdm(pred_files):
        try:
            process_file(str(pred_file), ghost_dataset, output_dir, config)
            processed += 1
        except Exception as e:
            print(f"[ERROR] {pred_file.name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    elapsed = time.perf_counter() - start_time
    print(f"\n[INFO] Done: {processed} processed, {failed} failed, {elapsed:.1f}s")


if __name__ == "__main__":
    main()

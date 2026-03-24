import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

# Reuse single-file evaluator
from src.visualize.evaluate_pcd import evaluate_ghost_removal


@dataclass
class PairResult:
    file_stem: str
    gt_path: Path
    pred_path: Path
    num_gt_ghost: int
    num_removed_ghost: int
    num_remaining_ghost: int
    removal_rate: float
    num_gt_object: int
    num_remaining_object: int
    object_residual_rate: float
    num_gt_glass: int
    num_remaining_glass: int
    glass_residual_rate: float
    num_gt_object_glass: int
    num_remaining_object_glass: int
    object_glass_residual_rate: float
    scene_name: str


def find_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Find (gt, pred) PCD pairs under the given directory (non-recursive).
    A GT file ends with '_gt.pcd' and its pair is the same name without the '_gt' suffix.
    """
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory does not exist or is not a directory: {root}")

    gt_files = sorted(root.glob("*_gt.pcd"))
    pairs: List[Tuple[Path, Path]] = []
    for gt in gt_files:
        pred = gt.with_name(gt.name.replace("_gt.pcd", ".pcd"))
        if pred.exists():
            pairs.append((gt, pred))
    return pairs


def extract_scene_name(filename: str) -> str:
    """
    Extract scene name from a filename such as:
    - scene001_20250901153350_... -> 'scene001'

    If the pattern does not match, fall back to the filename without extensions/suffixes before timestamp.
    """
    name = filename
    if name.endswith("_gt.pcd"):
        name = name[: -len("_gt.pcd")]
    elif name.endswith(".pcd"):
        name = name[: -len(".pcd")]

    # Normalize to the base scene key like "scene<NNN>"
    # e.g., "scene001_hist002_20250901..." -> "scene001"
    m_base = re.match(r"^(scene\d+)", name)
    if m_base:
        return m_base.group(1)

    # Fallback: use the remaining leading part until first timestamp-like block
    m = re.match(r"([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_", name)
    if m:
        return m.group(1)
    return name


def run_batch(
    root: Path,
    distance_threshold: float,
) -> List[PairResult]:
    """
    Run evaluation for all (gt, pred) pairs and return detailed results.
    """
    pairs = find_pairs(root)
    results: List[PairResult] = []
    for gt_path, pred_path in tqdm(pairs):
        single = evaluate_ghost_removal(
            str(gt_path),
            str(pred_path),
            distance_threshold=distance_threshold,
        )
        scene_name = extract_scene_name(gt_path.name)
        results.append(
            PairResult(
                file_stem=gt_path.stem.replace("_gt", ""),
                gt_path=gt_path,
                pred_path=pred_path,
                num_gt_ghost=single["num_gt_ghost"],
                num_removed_ghost=single["num_removed_ghost"],
                num_remaining_ghost=single["num_remaining_ghost"],
                removal_rate=single["removal_rate"],
                num_gt_object=single["num_gt_object"],
                num_remaining_object=single["num_remaining_object"],
                object_residual_rate=single["object_residual_rate"],
                num_gt_glass=single["num_gt_glass"],
                num_remaining_glass=single["num_remaining_glass"],
                glass_residual_rate=single["glass_residual_rate"],
                num_gt_object_glass=single["num_gt_object_glass"],
                num_remaining_object_glass=single["num_remaining_object_glass"],
                object_glass_residual_rate=single["object_glass_residual_rate"],
                scene_name=scene_name,
            )
        )
    return results


def summarize_by_scene(results: List[PairResult]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results per scene.
    Returns mapping:
      scene -> {
        'num_files': int,
        'avg_removal_rate': float,
        'sum_gt_ghost': int,
        'sum_removed_ghost': int,
        'sum_remaining_ghost': int,
        'avg_object_residual_rate': float,
        'avg_glass_residual_rate': float,
        'avg_object_glass_residual_rate': float,
        'sum_gt_object': int,
        'sum_remaining_object': int,
        'sum_gt_glass': int,
        'sum_remaining_glass': int,
        'sum_gt_object_glass': int,
        'sum_remaining_object_glass': int,
      }
    """
    buckets: Dict[str, List[PairResult]] = defaultdict(list)
    for r in results:
        buckets[r.scene_name].append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for bld, items in buckets.items():
        num_files = len(items)
        avg_rate = sum(x.removal_rate for x in items) / num_files if num_files else 0.0
        sum_gt = sum(x.num_gt_ghost for x in items)
        sum_removed = sum(x.num_removed_ghost for x in items)
        sum_remaining = sum(x.num_remaining_ghost for x in items)
        avg_object_residual = (
            sum(x.object_residual_rate for x in items) / num_files if num_files else 0.0
        )
        avg_glass_residual = (
            sum(x.glass_residual_rate for x in items) / num_files if num_files else 0.0
        )
        avg_object_glass_residual = (
            sum(x.object_glass_residual_rate for x in items) / num_files if num_files else 0.0
        )
        sum_gt_object = sum(x.num_gt_object for x in items)
        sum_remaining_object = sum(x.num_remaining_object for x in items)
        sum_gt_glass = sum(x.num_gt_glass for x in items)
        sum_remaining_glass = sum(x.num_remaining_glass for x in items)
        sum_gt_object_glass = sum(x.num_gt_object_glass for x in items)
        sum_remaining_object_glass = sum(x.num_remaining_object_glass for x in items)
        summary[bld] = {
            "num_files": float(num_files),
            "avg_removal_rate": avg_rate,
            "sum_gt_ghost": float(sum_gt),
            "sum_removed_ghost": float(sum_removed),
            "sum_remaining_ghost": float(sum_remaining),
            "avg_object_residual_rate": avg_object_residual,
            "avg_glass_residual_rate": avg_glass_residual,
            "avg_object_glass_residual_rate": avg_object_glass_residual,
            "sum_gt_object": float(sum_gt_object),
            "sum_remaining_object": float(sum_remaining_object),
            "sum_gt_glass": float(sum_gt_glass),
            "sum_remaining_glass": float(sum_remaining_glass),
            "sum_gt_object_glass": float(sum_gt_object_glass),
            "sum_remaining_object_glass": float(sum_remaining_object_glass),
        }
    return summary


def print_per_file(results: List[PairResult]) -> None:
    """
    Print per-file summary lines.
    """
    print("\n" + "=" * 80)
    print("Per-file Results")
    print("=" * 80)
    header = (
        f"{'Scene':<20} {'File':<40} "
        f"{'GhostRem(%)':>12} {'GT Ghost':>10} {'Removed':>10} {'Remain':>10} "
        f"{'ObjRes(%)':>12} {'GlassRes(%)':>12} {'Obj+GlassRes(%)':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scene_name:<20} {r.file_stem:<40} "
            f"{r.removal_rate * 100:>12.2f} {r.num_gt_ghost:>10d} {r.num_removed_ghost:>10d} {r.num_remaining_ghost:>10d} "
            f"{r.object_residual_rate * 100:>12.2f} {r.glass_residual_rate * 100:>12.2f} "
            f"{r.object_glass_residual_rate * 100:>16.2f}"
        )


def print_by_scene(summary: Dict[str, Dict[str, float]]) -> None:
    """
    Print aggregated results per scene and overall average.
    """
    print("\n" + "=" * 80)
    print("Per-scene Aggregation")
    print("=" * 80)
    header = (
        f"{'Scene':<20} {'Files':>6} "
        f"{'GhostRem(%)':>12} {'ObjRes(%)':>12} {'GlassRes(%)':>12} {'Obj+GlassRes(%)':>16} "
        f"{'GT Ghost':>10} {'Removed':>10} {'Remain':>10} "
        f"{'GT Obj':>10} {'Remain Obj':>12} "
        f"{'GT Glass':>10} {'Remain Glass':>13}"
    )
    print(header)
    print("-" * len(header))
    scenes = sorted(summary.keys())
    ghost_rates: List[float] = []
    obj_res_rates: List[float] = []
    glass_res_rates: List[float] = []
    obj_glass_res_rates: List[float] = []
    total_gt_ghost = 0.0
    total_removed_ghost = 0.0
    total_remaining_ghost = 0.0
    total_gt_object = 0.0
    total_remaining_object = 0.0
    total_gt_glass = 0.0
    total_remaining_glass = 0.0
    for b in scenes:
        s = summary[b]
        ghost_rates.append(s["avg_removal_rate"])
        obj_res_rates.append(s["avg_object_residual_rate"])
        glass_res_rates.append(s["avg_glass_residual_rate"])
        obj_glass_res_rates.append(s["avg_object_glass_residual_rate"])
        total_gt_ghost += s["sum_gt_ghost"]
        total_removed_ghost += s["sum_removed_ghost"]
        total_remaining_ghost += s["sum_remaining_ghost"]
        total_gt_object += s["sum_gt_object"]
        total_remaining_object += s["sum_remaining_object"]
        total_gt_glass += s["sum_gt_glass"]
        total_remaining_glass += s["sum_remaining_glass"]
        print(
            f"{b:<20} "
            f"{int(s['num_files']):>6d} "
            f"{s['avg_removal_rate'] * 100:>12.2f} "
            f"{s['avg_object_residual_rate'] * 100:>12.2f} "
            f"{s['avg_glass_residual_rate'] * 100:>12.2f} "
            f"{s['avg_object_glass_residual_rate'] * 100:>16.2f} "
            f"{int(s['sum_gt_ghost']):>10d} "
            f"{int(s['sum_removed_ghost']):>10d} "
            f"{int(s['sum_remaining_ghost']):>10d} "
            f"{int(s['sum_gt_object']):>10d} "
            f"{int(s['sum_remaining_object']):>12d} "
            f"{int(s['sum_gt_glass']):>10d} "
            f"{int(s['sum_remaining_glass']):>13d}"
        )
    if ghost_rates:
        overall_avg = sum(ghost_rates) / len(ghost_rates)
        overall_obj_res = sum(obj_res_rates) / len(obj_res_rates)
        overall_glass_res = sum(glass_res_rates) / len(glass_res_rates)
        overall_obj_glass_res = sum(obj_glass_res_rates) / len(obj_glass_res_rates)
    else:
        overall_avg = overall_obj_res = overall_glass_res = overall_obj_glass_res = 0.0
    print("-" * len(header))
    print(
        f"{'OVERALL AVG':<20} {'':>6} "
        f"{overall_avg * 100:>12.2f} "
        f"{overall_obj_res * 100:>12.2f} "
        f"{overall_glass_res * 100:>12.2f} "
        f"{overall_obj_glass_res * 100:>16.2f} "
        f"{int(total_gt_ghost):>10d} "
        f"{int(total_removed_ghost):>10d} "
        f"{int(total_remaining_ghost):>10d} "
        f"{int(total_gt_object):>10d} "
        f"{int(total_remaining_object):>12d} "
        f"{int(total_gt_glass):>10d} "
        f"{int(total_remaining_glass):>13d}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate ghost removal over a directory containing PCD pairs."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to YAML config.")
    parser.add_argument("--root", type=str, help="Root directory containing PCD pairs.")
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for matching points (in meters). Default: 0.01",
    )
    args, _ = parser.parse_known_args()
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.root:
        if config.get("root"):
            args.root = config["root"]
        else:
            parser.error("the following arguments are required: --root (or root in config)")

    return args


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    distance_threshold = float(args.distance_threshold)

    print(f"[INFO] Scanning directory: {root}")
    pairs = find_pairs(root)
    print(f"[INFO] Found {len(pairs)} pairs")
    if not pairs:
        return

    results = run_batch(root, distance_threshold)
    print_per_file(results)

    summary = summarize_by_scene(results)
    print_by_scene(summary)


if __name__ == "__main__":
    main()

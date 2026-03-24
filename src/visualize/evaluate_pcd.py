import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def load_pointcloud(pcd_path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a PCD file.

    Args:
        pcd_path: Path to the PCD file

    Returns:
        Open3D point cloud object
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {pcd_path}")
    return pcd


def get_class_masks(pcd: o3d.geometry.PointCloud) -> dict:
    """
    Return boolean masks for each semantic class based on color.
    """
    if not pcd.has_colors():
        raise ValueError("Point cloud must have color information to identify classes")

    colors = np.asarray(pcd.colors)

    masks = {
        "OBJECT": (colors[:, 0] < 0.1) & (colors[:, 1] > 0.9) & (colors[:, 2] < 0.1),
        "GLASS": (colors[:, 0] < 0.1) & (colors[:, 1] < 0.1) & (colors[:, 2] > 0.9),
        "GHOST": (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1),
        "NOISE": (
            (colors[:, 0] > 0.4)
            & (colors[:, 0] < 0.6)
            & (colors[:, 1] > 0.4)
            & (colors[:, 1] < 0.6)
            & (colors[:, 2] > 0.4)
            & (colors[:, 2] < 0.6)
        ),
    }
    return masks


def count_points_by_class(pcd: o3d.geometry.PointCloud) -> dict:
    """
    Count points for each class based on color.

    Args:
        pcd: Open3D point cloud with colors

    Returns:
        Dictionary with class names as keys and counts as values
    """
    masks = get_class_masks(pcd)

    counts = {
        "OBJECT": int(np.sum(masks["OBJECT"])),
        "GLASS": int(np.sum(masks["GLASS"])),
        "GHOST": int(np.sum(masks["GHOST"])),
        "NOISE": int(np.sum(masks["NOISE"])),
    }

    return counts


def extract_ghost_points(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Extract ghost points from a point cloud based on color.

    Args:
        pcd: Open3D point cloud with colors

    Returns:
        Array of ghost point coordinates (N, 3)
    """
    if not pcd.has_colors():
        raise ValueError("Point cloud must have color information to identify ghost points")

    points = np.asarray(pcd.points)
    masks = get_class_masks(pcd)

    ghost_points = points[masks["GHOST"]]
    return ghost_points


def count_ghost_points(pcd: o3d.geometry.PointCloud) -> int:
    """
    Count the number of ghost points in a point cloud.

    Args:
        pcd: Open3D point cloud with colors

    Returns:
        Number of ghost points
    """
    ghost_points = extract_ghost_points(pcd)
    return len(ghost_points)


def evaluate_ghost_removal(
    gt_pcd_path: str,
    removed_pcd_path: str,
    distance_threshold: float = 0.01,
) -> dict:
    """
    Evaluate ghost removal rate by comparing GT point cloud and ghost-removed point cloud.

    Args:
        gt_pcd_path: Path to GT point cloud PCD file
        removed_pcd_path: Path to ghost-removed point cloud PCD file
        distance_threshold: Distance threshold for matching points (in meters)

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"[INFO] Loading GT point cloud: {gt_pcd_path}")
    gt_pcd = load_pointcloud(gt_pcd_path)

    print(f"[INFO] Loading ghost-removed point cloud: {removed_pcd_path}")
    removed_pcd = load_pointcloud(removed_pcd_path)

    # Count points by class in GT
    print("[INFO] Counting points by class in GT point cloud...")
    gt_class_counts = count_points_by_class(gt_pcd)
    gt_masks = get_class_masks(gt_pcd)
    gt_points = np.asarray(gt_pcd.points)
    print(f"\n{'=' * 60}")
    print("GT Point Cloud Class Counts")
    print(f"{'=' * 60}")
    print(f"OBJECT:  {gt_class_counts['OBJECT']}")
    print(f"GLASS:   {gt_class_counts['GLASS']}")
    print(f"GHOST:   {gt_class_counts['GHOST']}")
    print(f"NOISE: {gt_class_counts['NOISE']}")
    print(f"{'=' * 60}\n")

    # Extract GT points per class
    gt_ghost_points = gt_points[gt_masks["GHOST"]]
    num_gt_ghost = len(gt_ghost_points)

    if num_gt_ghost == 0:
        print("[WARNING] No ghost points found in GT point cloud")

    removed_points = np.asarray(removed_pcd.points)

    def count_remaining(points: np.ndarray) -> int:
        if points.size == 0:
            return 0
        if distance_threshold == 0:
            # For exact match, use set-based lookup without KDTree
            # Convert points to tuples for set membership
            removed_set = {tuple(p) for p in removed_points}
            count = sum(1 for point in points if tuple(point) in removed_set)
            return count
        else:
            # Use KDTree for distance-based matching
            removed_kdtree = o3d.geometry.KDTreeFlann(removed_pcd)
            count = 0
            for point in points:
                [k, _, _] = removed_kdtree.search_radius_vector_3d(point, distance_threshold)
                if k > 0:
                    count += 1
            return count

    remaining_ghost = count_remaining(gt_ghost_points)
    removed_count = num_gt_ghost - remaining_ghost
    removal_rate = removed_count / num_gt_ghost if num_gt_ghost > 0 else 0.0

    gt_object_points = gt_points[gt_masks["OBJECT"]]
    gt_glass_points = gt_points[gt_masks["GLASS"]]

    num_gt_object = len(gt_object_points)
    num_gt_glass = len(gt_glass_points)

    remaining_object = count_remaining(gt_object_points)
    remaining_glass = count_remaining(gt_glass_points)

    object_residual_rate = remaining_object / num_gt_object if num_gt_object > 0 else 0.0
    glass_residual_rate = remaining_glass / num_gt_glass if num_gt_glass > 0 else 0.0

    num_gt_object_glass = num_gt_object + num_gt_glass
    remaining_object_glass = remaining_object + remaining_glass
    object_glass_residual_rate = (
        remaining_object_glass / num_gt_object_glass if num_gt_object_glass > 0 else 0.0
    )

    print(f"\n{'=' * 60}")
    print("Ghost Removal Evaluation Results")
    print(f"{'=' * 60}")
    print(f"GT ghost points:           {num_gt_ghost}")
    print(f"Removed ghost points:      {removed_count}")
    print(f"Remaining ghost points:   {remaining_ghost}")
    print(f"Removal rate:             {removal_rate * 100:.2f}%")
    print(f"OBJECT residual rate:     {object_residual_rate * 100:.2f}%")
    print(f"GLASS residual rate:      {glass_residual_rate * 100:.2f}%")
    print(f"OBJECT+GLASS residual:    {object_glass_residual_rate * 100:.2f}%")
    print(f"{'=' * 60}")

    return {
        "num_gt_ghost": num_gt_ghost,
        "num_removed_ghost": removed_count,
        "num_remaining_ghost": remaining_ghost,
        "removal_rate": removal_rate,
        "num_gt_object": num_gt_object,
        "num_remaining_object": remaining_object,
        "object_residual_rate": object_residual_rate,
        "num_gt_glass": num_gt_glass,
        "num_remaining_glass": remaining_glass,
        "glass_residual_rate": glass_residual_rate,
        "num_gt_object_glass": num_gt_object_glass,
        "num_remaining_object_glass": remaining_object_glass,
        "object_glass_residual_rate": object_glass_residual_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ghost removal rate by comparing GT and ghost-removed point clouds."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--gt_pcd",
        type=str,
        help="Path to GT point cloud PCD file",
    )
    parser.add_argument(
        "--removed_pcd",
        type=str,
        help="Path to ghost-removed point cloud PCD file",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for matching points (in meters). Default: 0.01",
    )

    args, unknown = parser.parse_known_args()

    # Load config file
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Set defaults from YAML, but allow CLI to override
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Check for required arguments after loading config
    if not args.gt_pcd:
        if config.get("gt_pcd"):
            args.gt_pcd = config["gt_pcd"]
        else:
            parser.error(
                "the following arguments are required: --gt_pcd (or gt_pcd in config file)"
            )

    if not args.removed_pcd:
        if config.get("removed_pcd"):
            args.removed_pcd = config["removed_pcd"]
        else:
            parser.error(
                "the following arguments are required: --removed_pcd (or removed_pcd in config file)"
            )

    # Validate input files
    if not Path(args.gt_pcd).exists():
        parser.error(f"GT point cloud file does not exist: {args.gt_pcd}")

    if not Path(args.removed_pcd).exists():
        parser.error(f"Ghost-removed point cloud file does not exist: {args.removed_pcd}")

    # Evaluate
    results = evaluate_ghost_removal(
        args.gt_pcd,
        args.removed_pcd,
        distance_threshold=args.distance_threshold,
    )

    # Print summary
    print(f"\n[SUMMARY] Ghost removal rate: {results['removal_rate'] * 100:.2f}%")


if __name__ == "__main__":
    main()

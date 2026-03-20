import argparse
import os
from collections import Counter

import blosc2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from scipy.signal import find_peaks

from src.utils import upsample_histogram_direction_zero_padding
from src.visualize.fwl_lib import AnnotationLabel, PointcloudWrapper


def load_annotation_volume(b2_path: str) -> np.ndarray:
    """
    Loads the raw 3D annotation volume from a .b2 file.
    """
    with open(b2_path, "rb") as f:
        return blosc2.unpack_array2(f.read())


def create_annotation_map(
    annotation_volume: np.ndarray,
    data_width: int,
    data_height: int,
    x_offset: int | None = None,
    y_offset: int | None = None,
    x_min: int = 0,
    y_min: int = 0,
) -> dict:
    """
    Creates a sparse annotation map from a dense annotation volume, applying offsets.
    """
    ann_w, ann_h, ann_d = annotation_volume.shape

    # Default to centering if offsets are not provided
    if x_offset is None:
        x_offset = 0
    if y_offset is None:
        y_offset = (data_height - ann_h) // 2

    print(
        f"[INFO] Placing annotation volume of shape {annotation_volume.shape} at offset (x={x_offset}, y={y_offset})"
    )

    # Add
    x_max = x_min + data_width - 1
    y_max = y_min + data_height - 1

    annotation_map = {}
    for x_ann in range(ann_w):
        for y_ann in range(ann_h):
            # Map annotation coordinates to data coordinates
            y_data = y_ann + y_offset
            x_data = x_ann + x_offset

            # Skip if pixel is outside the data dimensions
            if not (x_min <= x_data <= x_max and y_min <= y_data <= y_max):  # fix
                continue

            annotations_for_pixel = {}
            for bin_ann in range(ann_d):
                label = annotation_volume[x_ann, y_ann, bin_ann]
                if label != AnnotationLabel.NOISE:
                    # Apply bin offset
                    # bin_data = bin_ann + bin_offset # Delete
                    annotations_for_pixel[bin_ann] = label

            if annotations_for_pixel:
                annotation_map[(x_data, y_data)] = annotations_for_pixel

    return annotation_map


def generate_pointcloud_with_open3d(
    pcw: PointcloudWrapper,
    annotation_map: dict,
    show_noise: bool = False,
    show_undefined: bool = False,
    remove_ghost_neighbors_radius: float | None = None,
    ghost_removal_mode: str = "none",
    color_mode: str = "label",
) -> o3d.geometry.PointCloud:
    """
    Generates a colored point cloud from histogram data and an annotation map.
    """
    df = pcw.df
    histograms = pcw.histograms

    points = []
    labels = []

    for idx, hist in enumerate(histograms):
        row = df.iloc[idx]
        x = int(row["//Pixel_X"])
        y = int(row["Pixel_Y"])

        # Add
        max_peak_height = np.max(hist)
        peaks, _ = find_peaks(hist, height=max_peak_height * 0.1, width=3)
        ann = annotation_map.get((x, y), {})

        for peak in peaks:
            peak_int = int(peak)
            label = ann.get(peak_int, AnnotationLabel.NOISE)

            if ghost_removal_mode == "skip_points" and label == AnnotationLabel.GHOST:
                continue

            if label == AnnotationLabel.NOISE and not show_noise:
                continue

            if label == AnnotationLabel.UNDEFINED and not show_undefined:
                continue

            distance_result = pcw.get_distance_from_bin(histograms[idx], peak_int)
            distance = distance_result.distance_m
            point = pcw._calc_single_point(row, distance)

            points.append(point)
            labels.append(label)

    points = np.array(points)
    labels = np.array(labels)

    # Remove neighbors of ghost points if radius is specified
    if (
        ghost_removal_mode == "remove_neighbors"
        and remove_ghost_neighbors_radius is not None
        and remove_ghost_neighbors_radius > 0
    ):
        print(
            f"[INFO] Removing points within a {remove_ghost_neighbors_radius}m radius of ghost points."
        )
        if np.any(labels == AnnotationLabel.GHOST):
            # Build KDTree from all points
            pcd_full = o3d.geometry.PointCloud()
            pcd_full.points = o3d.utility.Vector3dVector(points)
            kdtree = o3d.geometry.KDTreeFlann(pcd_full)

            ghost_indices = np.where(labels == AnnotationLabel.GHOST)[0]
            indices_to_remove = set()

            for ghost_idx in ghost_indices:
                # Find neighbors within the radius for each ghost point
                [k, idx_neighbors, _] = kdtree.search_radius_vector_3d(
                    pcd_full.points[ghost_idx], remove_ghost_neighbors_radius
                )
                indices_to_remove.update(idx_neighbors)

            if indices_to_remove:
                print(
                    f"[INFO] Found {len(ghost_indices)} ghost points. Removing {len(indices_to_remove)} points (ghosts and their neighbors)."
                )
                # Create a boolean mask to keep points that are not in the removal set
                keep_mask = np.ones(len(points), dtype=bool)
                keep_mask[list(indices_to_remove)] = False

                # Apply the mask
                points = points[keep_mask]
                labels = labels[keep_mask]
        else:
            print("[INFO] No ghost points found, skipping removal.")

    annotation_pcd = o3d.geometry.PointCloud()
    annotation_pcd.points = o3d.utility.Vector3dVector(points)

    if color_mode == "label":
        # Count points per class
        label_counts = Counter(labels)

        print("=== Point Cloud Class Counts ===")
        print(f"OBJECT  (Green): {label_counts.get(AnnotationLabel.OBJECT, 0)}")
        print(f"GLASS   (Blue) : {label_counts.get(AnnotationLabel.GLASS, 0)}")
        print(f"GHOST   (Red)  : {label_counts.get(AnnotationLabel.GHOST, 0)}")
        print(f"UNDEFINED     (White): {label_counts.get(AnnotationLabel.UNDEFINED, 0)}")
        if show_noise:
            print(f"NOISE (Gray) : {label_counts.get(AnnotationLabel.NOISE, 0)}")

        # Create colored point cloud by label
        color_map = {
            AnnotationLabel.OBJECT: [0.0, 1.0, 0.0],  # Green
            AnnotationLabel.GLASS: [0.0, 0.0, 1.0],  # Blue
            AnnotationLabel.GHOST: [1.0, 0.0, 0.0],  # Red
            AnnotationLabel.NOISE: [0.5, 0.5, 0.5],  # Gray
            AnnotationLabel.UNDEFINED: [1.0, 1.0, 1.0],  # White
        }

        annotation_colors = np.array([color_map.get(label, [0, 0, 0]) for label in labels])
        annotation_pcd.colors = o3d.utility.Vector3dVector(annotation_colors)

    elif color_mode == "distance":
        print("[INFO] Coloring points by log-scaled distance from origin.")
        if len(points) > 0:
            distances = np.linalg.norm(points, axis=1)
            # Apply logarithmic scaling to emphasize closer points
            log_distances = np.log1p(distances)

            max_log_dist = np.max(log_distances)
            min_log_dist = np.min(log_distances)

            if max_log_dist > min_log_dist:
                normalized_distances = (log_distances - min_log_dist) / (
                    max_log_dist - min_log_dist
                )
            else:
                normalized_distances = np.zeros_like(log_distances)

            cmap = plt.get_cmap("viridis")
            colors = cmap(normalized_distances)[:, :3]  # Get RGB from RGBA
            annotation_pcd.colors = o3d.utility.Vector3dVector(colors)

    # if color_mode is 'none', do nothing

    return annotation_pcd


def visualize_open3d(pcd: o3d.geometry.PointCloud, output_path: str = "output.pcd") -> None:
    """Visualizes an Open3D point cloud."""
    # o3d.visualization.draw_geometries([pcd], window_name="B2 Annotation Visualization")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False, compressed=True)
    print(f"[INFO] Saved point cloud to {output_path}")


def count_and_print_classes(annotation_map: dict) -> None:
    """Counts and prints the number of annotations for each class from the map."""
    all_labels = [label for ann in annotation_map.values() for label in ann.values()]
    counts = Counter(all_labels)

    print("=== Annotation Class Counts (from .b2 file) ===")
    print(f"OBJECT:  {counts.get(AnnotationLabel.OBJECT, 0)}")
    print(f"GLASS:   {counts.get(AnnotationLabel.GLASS, 0)}")
    print(f"GHOST:   {counts.get(AnnotationLabel.GHOST, 0)}")
    print(f"NOISE: {counts.get(AnnotationLabel.NOISE, 0)}")
    print(f"UNDEFINED:     {counts.get(AnnotationLabel.UNDEFINED, 0)}")
    print("=================================================")


# Add
def crop_annotation_volume(
    annotation_volume: np.ndarray, x_range: tuple, y_range: tuple, histo_offset: int = 0
) -> np.ndarray:
    """
    Crops an annotation volume to the specified range.

    Args:
        annotation_volume: The full annotation volume
        x_range: (x_min, x_max) tuple
        y_range: (y_min, y_max) tuple

    Returns:
        Cropped annotation volume
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Crop the annotation volume
    ann_w, ann_h, ann_d = annotation_volume.shape

    # Make sure the range is within bounds
    x_min_clamped = max(0, x_min)
    x_max_clamped = min(ann_w, x_max)
    y_min_clamped = max(0, y_min)
    y_max_clamped = min(ann_h, y_max)

    cropped = annotation_volume[
        x_min_clamped:x_max_clamped, y_min_clamped:y_max_clamped, histo_offset:
    ]

    print(
        f"[INFO] Cropped annotation volume from shape {annotation_volume.shape} to {cropped.shape}"
    )
    print(
        f"[INFO] Crop range: x=[{x_min_clamped}:{x_max_clamped}], y=[{y_min_clamped}:{y_max_clamped}]"
    )

    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize point cloud from histogram data and a .b2 annotation file."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML config file.")
    # The other arguments are defined here so we can check for them, but they will be overridden by the config file
    parser.add_argument(
        "-d", "--data", help="Path to the histogram data file (e.g., .csv, .npz, .b2)"
    )
    parser.add_argument("-a", "--annotation", help="Path to the .b2 annotation file.")
    parser.add_argument(
        "-s", "--show_noise", action="store_true", help="Show NOISE class in point cloud."
    )
    parser.add_argument(
        "--show_undefined", action="store_true", help="Show UNDEFINED class in point cloud."
    )
    parser.add_argument(
        "--x_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Range of x-coordinates to include.",
    )
    parser.add_argument(
        "--y_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Range of y-coordinates to include.",
    )
    parser.add_argument(
        "--x_offset",
        type=int,
        help="Manual X offset for annotation placement. Defaults to centering.",
    )
    parser.add_argument(
        "--y_offset",
        type=int,
        help="Manual Y offset for annotation placement. Defaults to centering.",
    )
    parser.add_argument(
        "--bin_offset", type=int, help="Offset for annotation bin placement. Defaults to 0."
    )
    parser.add_argument(
        "--remove_ghost_neighbors_radius",
        type=float,
        help="Radius around ghost points to remove neighbors.",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["label", "distance", "none", "label_cmap"],
        default="label",
        help="Point cloud colorization mode.",
    )
    # Add
    parser.add_argument(
        "--ghost_removal_mode",
        type=str,
        choices=["none", "skip_points", "remove_neighbors"],
        default="none",
        help="Ghost removal method: 'none' (no removal), 'skip_points' (skip ghost-labeled peaks), 'remove_neighbors' (remove ghosts and neighbors within radius).",
    )
    parser.add_argument(
        "--original_depth",
        type=int,
        help="Original input depth size (Z direction) for annotation. If provided with target_depth, peak bin positions will be converted to original scale for visualization.",
    )
    parser.add_argument(
        "--target_depth",
        type=int,
        help="Target input depth size (Z direction) for annotation. If provided with original_depth, peak bin positions will be converted to original scale for visualization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output point cloud file.",
    )
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Whether to use the GT annotation file.",
    )
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Whether to upsample the annotation volume.",
    )

    args, unknown = parser.parse_known_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Set defaults from YAML, but allow CLI to override
    parser.set_defaults(**config)
    args = parser.parse_args()

    # Check for required arguments after loading config
    if not args.data or not args.annotation:
        parser.error(
            "the following arguments are required: -d/--data, -a/--annotation (must be provided via CLI or config file)"
        )

    # Load raw annotation volume
    print("Loading .b2 annotation file...")
    annotation_volume = load_annotation_volume(args.annotation)
    print(f"[INFO] Loaded annotation volume with shape: {annotation_volume.shape}")

    if args.gt:
        annotation_volume = crop_annotation_volume(
            annotation_volume, args.x_range, args.y_range, args.bin_offset
        )
    # Create PointcloudWrapper to get data dimensions
    print("Loading histogram data...")
    pcw = PointcloudWrapper(args.data, mode="SPD", spd_mode=15, histo_offset=args.bin_offset)

    # Crop the data if ranges are specified
    if args.x_range and args.y_range:
        x_min, x_max = args.x_range
        y_min, y_max = args.y_range
        print(f"Cropping data to x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
        pcw.crop_xy(x_min, x_max, y_min, y_max)

    # Add
    x_min = int(pcw.df["//Pixel_X"].min())
    x_max = int(pcw.df["//Pixel_X"].max())
    y_min = int(pcw.df["Pixel_Y"].min())
    y_max = int(pcw.df["Pixel_Y"].max())
    data_width = x_max - x_min + 1
    data_height = y_max - y_min + 1
    print(f"[INFO] Histogram data dimensions: width={data_width}, height={data_height}")

    # Create the sparse annotation map, applying offsets
    print("Creating annotation map with offsets...")
    if not args.gt and args.upsample:
        print(
            f"[INFO] Upsampling annotation volume from shape {annotation_volume.shape} to {args.original_depth}"
        )
        annotation_volume = upsample_histogram_direction_zero_padding(
            annotation_volume, args.original_depth
        )

    annotation_map = create_annotation_map(
        annotation_volume,
        data_width=data_width,
        data_height=data_height,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        x_min=x_min,
        y_min=y_min,
    )

    # Count classes from the generated map
    count_and_print_classes(annotation_map)

    # Generate point cloud & visualize
    print("Generating point cloud...")
    pcd = generate_pointcloud_with_open3d(
        pcw,
        annotation_map,
        show_noise=args.show_noise,
        show_undefined=args.show_undefined,
        remove_ghost_neighbors_radius=args.remove_ghost_neighbors_radius,
        ghost_removal_mode=args.ghost_removal_mode,
        color_mode=args.color_mode,
    )

    # Build output path from annotation filename
    ann_basename = os.path.splitext(os.path.basename(args.annotation))[0]
    if args.gt:
        output_path = os.path.join(args.output_dir, f"{ann_basename}_gt.pcd")
    else:
        output_path = os.path.join(args.output_dir, f"{ann_basename}.pcd")

    print("Visualizing point cloud...")
    visualize_open3d(pcd, output_path)


if __name__ == "__main__":
    main()

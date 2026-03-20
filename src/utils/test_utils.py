import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from src.config.constants import CLASS_COLORS, LABEL_MAP


def calculate_iou_score(
    prediction: np.ndarray, target: np.ndarray, num_classes: int, ignore_labels: list[int]
) -> Dict[str, float]:
    """Calculate IoU (Intersection over Union) for each class and average (excluding ignored classes)"""
    iou_scores = {}
    valid_class_scores = []

    for class_id in range(num_classes):
        # Skip ignored labels
        if class_id in ignore_labels:
            continue

        pred_class = (prediction == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)

        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class) - intersection

        if union == 0:
            iou = 1.0 if np.sum(target_class) == 0 else 0.0
        else:
            iou = intersection / union

        iou_scores[f"iou_class_{class_id}"] = iou
        valid_class_scores.append(iou)

    # Calculate mean IoU (only for non-ignored classes)
    iou_scores["iou_mean"] = np.mean(valid_class_scores) if valid_class_scores else 0.0

    return iou_scores


def calculate_pixel_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calculate pixel-wise accuracy"""
    correct_pixels = np.sum(prediction == target)
    total_pixels = target.size
    return correct_pixels / total_pixels


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def print_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    normalize: str | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Print confusion matrix in a formatted table to standard output.

    Args:
        confusion_matrix: Confusion matrix array (true labels x predicted labels)
        class_names: List of class names for labels
        normalize: Normalization method ('true', 'pred', 'all', or None)
        title: Title for the table
    """
    # Create a copy to avoid modifying the original
    cm = confusion_matrix.copy().astype(float)

    # Normalize confusion matrix if requested
    if normalize == "true":
        # Normalize by true labels (rows)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaN with 0
        title += " (Normalized by True Labels)"
    elif normalize == "pred":
        # Normalize by predicted labels (columns)
        cm = cm / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm)
        title += " (Normalized by Predicted Labels)"
    elif normalize == "all":
        # Normalize by total sum
        cm = cm / cm.sum()
        title += " (Normalized)"

    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print(f"{'=' * 80}")

    # Prepare header
    header = "True\\Pred"
    col_width = max(8, max(len(name) for name in class_names))
    header = f"{header:>{col_width}}"

    for name in class_names:
        header += f" {name:>{col_width}}"

    print(header)
    print("-" * len(header))

    # Print each row
    for i, true_class in enumerate(class_names):
        row = f"{true_class:>{col_width}}"
        for j in range(len(class_names)):
            if normalize:
                value = f"{cm[i, j]:.3f}"
            else:
                value = f"{int(cm[i, j])}"
            row += f" {value:>{col_width}}"
        print(row)

    print("-" * len(header))
    print()


def plot_temporal_histogram(
    predictions: np.ndarray,
    annotations: np.ndarray,
    raw_input: np.ndarray,
    x: int,
    y: int,
    return_fig: bool = False,
) -> plt.Figure | None:
    """
    Plot temporal histogram for a specific x,y coordinate showing predictions and ground truth labels on top of raw input histogram.

    Args:
        predictions: Prediction array of shape (D, H, W) where D is time dimension
        annotations: Annotation array of shape (D, H, W) where D is time dimension
        raw_input: Raw input voxel data of shape (D, H, W) where D is time dimension
        x: X coordinate (width dimension)
        y: Y coordinate (height dimension)
        return_fig: Whether to return the figure object
    """
    # Extract time series for the specified x,y coordinate
    pred_timeseries = predictions[:, y, x]  # Shape: (D,)
    ann_timeseries = annotations[:, y, x]  # Shape: (D,)
    raw_timeseries = raw_input[:, y, x]  # Shape: (D,)

    # Create time axis
    time_steps = np.arange(len(pred_timeseries))

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot raw input as bar chart (background)
    bars = ax.bar(
        time_steps,
        raw_timeseries,
        alpha=0.6,
        color="lightgray",
        label="Raw Input Histogram",
        width=0.8,
        zorder=1,
        edgecolor="gray",
        linewidth=0.5,
    )

    max_value = np.max(raw_timeseries)
    peak_indices, _ = find_peaks(raw_timeseries, height=max_value * 0.1, width=3)

    # Filter non-zero prediction labels for plotting
    pred_nonzero_mask = pred_timeseries != 0
    pred_nonzero_times = time_steps[pred_nonzero_mask]
    pred_nonzero_values = pred_timeseries[pred_nonzero_mask]
    pred_nonzero_raw = raw_timeseries[pred_nonzero_mask]

    # Filter non-zero ground truth labels for plotting
    gt_nonzero_mask = ann_timeseries != 0
    gt_nonzero_times = time_steps[gt_nonzero_mask]
    gt_nonzero_values = ann_timeseries[gt_nonzero_mask]
    gt_nonzero_raw = raw_timeseries[gt_nonzero_mask]

    # Plot prediction labels on top of histogram bars (only non-zero labels)
    pred_scatters = {}
    if len(pred_nonzero_times) > 0:
        pred_y_positions = pred_nonzero_raw + 0.02 * np.max(
            raw_timeseries
        )  # Slightly above bar tops

        # Group by label value for different colors
        for label_val in np.unique(pred_nonzero_values):
            label_mask = pred_nonzero_values == label_val
            label_times = pred_nonzero_times[label_mask]
            label_y_pos = pred_y_positions[label_mask]

            color = CLASS_COLORS.get(label_val, "#000000")  # Default to black if label not found
            label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")

            pred_scatters[label_val] = ax.scatter(
                label_times,
                label_y_pos,
                s=120,
                color=color,
                marker="o",  # Circle for predictions
                label=f"Pred: {label_name}",
                edgecolors="black",
                linewidth=1.5,
                zorder=3,
                alpha=0.9,
            )

    # Plot ground truth labels on top of histogram bars (only non-zero labels)
    gt_scatters = {}
    if len(gt_nonzero_times) > 0:
        gt_y_positions = gt_nonzero_raw + 0.05 * np.max(
            raw_timeseries
        )  # Slightly higher than predictions

        # Group by label value for different colors
        for label_val in np.unique(gt_nonzero_values):
            label_mask = gt_nonzero_values == label_val
            label_times = gt_nonzero_times[label_mask]
            label_y_pos = gt_y_positions[label_mask]

            color = CLASS_COLORS.get(label_val, "#000000")  # Default to black if label not found
            label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")

            gt_scatters[label_val] = ax.scatter(
                label_times,
                label_y_pos,
                s=120,
                color=color,
                marker="s",  # Square for ground truth
                label=f"GT: {label_name}",
                edgecolors="black",
                linewidth=1.5,
                zorder=4,
                alpha=0.9,
            )

    # Set labels and title
    ax.set_xlabel("Time Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Raw Input Intensity", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Temporal Histogram with Labels at Position ({x}, {y})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set grid
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

    # Set y-limits
    max_raw = np.max(raw_timeseries) if np.max(raw_timeseries) > 0 else 1
    ax.set_ylim(0, max_raw * 1.1)  # Add small space for markers above bars

    # Draw vertical red lines at detected peak positions
    peak_lines = None
    if len(peak_indices) > 0:
        peak_lines = ax.vlines(
            peak_indices,
            ymin=0,
            ymax=max_raw * 1.05,
            colors="red",
            linestyles="-",
            linewidth=1.5,
            label="Detected Peaks",
            zorder=2,
        )

    # Set x-axis ticks
    ax.tick_params(axis="both", labelsize=12)

    # Create legend
    legend_handles = [bars]
    legend_labels = ["Raw Input Histogram"]

    # Add prediction scatters to legend
    for label_val, scatter in pred_scatters.items():
        legend_handles.append(scatter)
        label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")
        legend_labels.append(f"Pred: {label_name}")

    # Add ground truth scatters to legend
    for label_val, scatter in gt_scatters.items():
        legend_handles.append(scatter)
        label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")
        legend_labels.append(f"GT: {label_name}")

    # Add peak lines to legend
    if peak_lines is not None:
        legend_handles.append(peak_lines)
        legend_labels.append("Detected Peaks")

    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=12, framealpha=0.9)

    # Add subtle background color for better readability
    ax.set_facecolor("#fafafa")

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.close()
        return None


def plot_temporal_histogram_custom(
    predictions: np.ndarray,
    annotations: np.ndarray,
    raw_input: np.ndarray,
    x: int,
    y: int,
    return_fig: bool = False,
    width: int | None = None,
    prominence: float | None = None,
    threshold: float | None = None,
    distance: int | None = None,
) -> plt.Figure | None:
    """
    Plot temporal histogram for a specific x,y coordinate showing predictions and ground truth labels on top of raw input histogram.

    Args:
        predictions: Prediction array of shape (D, H, W) where D is time dimension
        annotations: Annotation array of shape (D, H, W) where D is time dimension
        raw_input: Raw input voxel data of shape (D, H, W) where D is time dimension
        x: X coordinate (width dimension)
        y: Y coordinate (height dimension)
        return_fig: Whether to return the figure object
    """
    # Extract time series for the specified x,y coordinate
    pred_timeseries = predictions[:, y, x]  # Shape: (D,)
    ann_timeseries = annotations[:, y, x]  # Shape: (D,)
    raw_timeseries = raw_input[:, y, x]  # Shape: (D,)

    # Create time axis
    time_steps = np.arange(len(pred_timeseries))

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot raw input as bar chart (background)
    bars = ax.bar(
        time_steps,
        raw_timeseries,
        alpha=0.6,
        color="lightgray",
        label="Raw Input Histogram",
        width=0.8,
        zorder=1,
        edgecolor="gray",
        linewidth=0.5,
    )

    max_value = np.max(raw_timeseries)
    peak_indices, _ = find_peaks(
        raw_timeseries,
        height=max_value * 0.2,
        width=width,
        distance=distance,
    )

    # Filter non-zero prediction labels for plotting
    pred_nonzero_mask = pred_timeseries != 0
    pred_nonzero_times = time_steps[pred_nonzero_mask]
    pred_nonzero_values = pred_timeseries[pred_nonzero_mask]
    pred_nonzero_raw = raw_timeseries[pred_nonzero_mask]

    # Filter non-zero ground truth labels for plotting
    gt_nonzero_mask = ann_timeseries != 0
    gt_nonzero_times = time_steps[gt_nonzero_mask]
    gt_nonzero_values = ann_timeseries[gt_nonzero_mask]
    gt_nonzero_raw = raw_timeseries[gt_nonzero_mask]

    # Plot prediction labels on top of histogram bars (only non-zero labels)
    pred_scatters = {}
    if len(pred_nonzero_times) > 0:
        pred_y_positions = pred_nonzero_raw + 0.02 * np.max(
            raw_timeseries
        )  # Slightly above bar tops

        # Group by label value for different colors
        for label_val in np.unique(pred_nonzero_values):
            label_mask = pred_nonzero_values == label_val
            label_times = pred_nonzero_times[label_mask]
            label_y_pos = pred_y_positions[label_mask]

            color = CLASS_COLORS.get(label_val, "#000000")  # Default to black if label not found
            label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")

            pred_scatters[label_val] = ax.scatter(
                label_times,
                label_y_pos,
                s=120,
                color=color,
                marker="o",  # Circle for predictions
                label=f"Pred: {label_name}",
                edgecolors="black",
                linewidth=1.5,
                zorder=3,
                alpha=0.9,
            )

    # Plot ground truth labels on top of histogram bars (only non-zero labels)
    gt_scatters = {}
    if len(gt_nonzero_times) > 0:
        gt_y_positions = gt_nonzero_raw + 0.05 * np.max(
            raw_timeseries
        )  # Slightly higher than predictions

        # Group by label value for different colors
        for label_val in np.unique(gt_nonzero_values):
            label_mask = gt_nonzero_values == label_val
            label_times = gt_nonzero_times[label_mask]
            label_y_pos = gt_y_positions[label_mask]

            color = CLASS_COLORS.get(label_val, "#000000")  # Default to black if label not found
            label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")

            gt_scatters[label_val] = ax.scatter(
                label_times,
                label_y_pos,
                s=120,
                color=color,
                marker="s",  # Square for ground truth
                label=f"GT: {label_name}",
                edgecolors="black",
                linewidth=1.5,
                zorder=4,
                alpha=0.9,
            )

    # Set labels and title
    ax.set_xlabel("Time Step", fontsize=14, fontweight="bold")
    ax.set_ylabel("Raw Input Intensity", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Temporal Histogram with Labels at Position ({x}, {y})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set grid
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

    # Set y-limits
    max_raw = np.max(raw_timeseries) if np.max(raw_timeseries) > 0 else 1
    ax.set_ylim(0, max_raw * 1.1)  # Add small space for markers above bars

    # Draw vertical red lines at detected peak positions
    peak_lines = None
    if len(peak_indices) > 0:
        peak_lines = ax.vlines(
            peak_indices,
            ymin=0,
            ymax=max_raw * 1.05,
            colors="red",
            linestyles="-",
            linewidth=1.5,
            label="Detected Peaks",
            zorder=2,
        )

    # Set x-axis ticks
    ax.tick_params(axis="both", labelsize=12)

    # Create legend
    legend_handles = [bars]
    legend_labels = ["Raw Input Histogram"]

    # Add prediction scatters to legend
    for label_val, scatter in pred_scatters.items():
        legend_handles.append(scatter)
        label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")
        legend_labels.append(f"Pred: {label_name}")

    # Add ground truth scatters to legend
    for label_val, scatter in gt_scatters.items():
        legend_handles.append(scatter)
        label_name = LABEL_MAP.get(label_val, f"Class_{label_val}")
        legend_labels.append(f"GT: {label_name}")

    # Add peak lines to legend
    if peak_lines is not None:
        legend_handles.append(peak_lines)
        legend_labels.append("Detected Peaks")

    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=12, framealpha=0.9)

    # Add subtle background color for better readability
    ax.set_facecolor("#fafafa")

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.close()
        return None


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: str,
    normalize: str | None = "true",  # 'true', 'pred', 'all', or None
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (12, 10),
    return_fig: bool = False,
) -> plt.Figure | None:
    """
    Plot and save confusion matrix as an image.

    Args:
        confusion_matrix: Confusion matrix array (true labels x predicted labels)
        class_names: List of class names for labels
        save_path: Path to save the image
        normalize: Normalization method ('true', 'pred', 'all', or None)
        title: Title for the plot
        figsize: Figure size (width, height)
    """
    # Create a copy to avoid modifying the original
    cm = confusion_matrix.copy().astype(float)

    # Normalize confusion matrix if requested
    if normalize == "true":
        # Normalize by true labels (rows)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaN with 0
        title += " (Normalized by True Labels)"
    elif normalize == "pred":
        # Normalize by predicted labels (columns)
        cm = cm / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm)
        title += " (Normalized by Predicted Labels)"
    elif normalize == "all":
        # Normalize by total sum
        cm = cm / cm.sum()
        title += " (Normalized)"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap using matplotlib imshow
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Add colorbar
    cbar_label = "Proportion" if normalize else "Count"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f"{cm[i, j]:.2%}"
            else:
                text = f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Set labels and title
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    # plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    # print(f"Confusion matrix saved to: {save_path}")

    if return_fig:
        return fig
    else:
        plt.close()  # Close the figure to free memory
        return None


def select_random_point(predictions: np.ndarray, annotations: np.ndarray) -> tuple[int, int]:
    """
    Select a random x,y coordinate from the voxel data where both prediction and annotation have valid data.

    Args:
        predictions: Prediction array of shape (D, H, W)
        annotations: Annotation array of shape (D, H, W)

    Returns:
        Tuple of (x, y) coordinates
    """
    D, H, W = predictions.shape

    # Find coordinates where both predictions and annotations have non-zero values at least once
    valid_coords = []
    for y in range(H):
        for x in range(W):
            pred_series = predictions[:, y, x]
            ann_series = annotations[:, y, x]
            # Check if there's any variation in the data (not all zeros or same value)
            if (np.any(pred_series != 0) or np.any(ann_series != 0)) and len(
                np.unique(np.concatenate([pred_series, ann_series]))
            ) > 1:
                valid_coords.append((x, y))

    if not valid_coords:
        # Fallback to center point if no valid coordinates found
        return W // 2, H // 2

    # Select random coordinate from valid ones
    idx = np.random.randint(0, len(valid_coords))
    return valid_coords[idx]

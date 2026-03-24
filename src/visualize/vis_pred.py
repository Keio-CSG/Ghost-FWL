import argparse
import os
import pathlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.signal import find_peaks
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import TestConfig, TrainingConfig, load_config_from_yaml
from src.config.constants import CLASS_COLORS, LABEL_MAP
from src.data import FWLDataset
from src.utils import get_model, plot_temporal_histogram, select_random_point

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Create class color mapping for visualization
def get_class_color_mapping() -> Dict[int, str]:
    """Get color mapping for classes"""
    return {
        label: CLASS_COLORS.get(label, "#808080")  # default gray for noise labels
        for label in LABEL_MAP.keys()
    }


def setup_matplotlib_backend() -> None:
    """Setup matplotlib backend for different environments"""
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("GTK3Agg")
            except Exception:
                print(
                    "Warning: No interactive matplotlib backend available. Using non-interactive backend."
                )
                matplotlib.use("Agg")


def detect_peaks_in_voxel(raw_input: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Detect peaks in 3D voxel data.

    Args:
        raw_input: Raw input voxel data of shape (X, Y, Z)
        min_height: Minimum height threshold for peak detection
        min_distance: Minimum distance between peaks

    Returns:
        List of (x, y, z) coordinates of detected peaks in (X, Y, Z) format
    """
    peaks = []
    X, Y, Z = raw_input.shape

    # Find peaks in each temporal series (along Z dimension)
    for x in range(X):
        for y in range(Y):
            time_series = raw_input[x, y, :]

            # Find peaks in the time series
            max_value = np.max(time_series)
            peak_indices, _ = find_peaks(time_series, height=max_value * 0.1, width=3)

            # Add peaks to the list
            for peak_z in peak_indices:
                peaks.append((x, y, peak_z))

    return peaks


def evaluate_peaks_at_positions(
    predictions: np.ndarray,
    annotations: np.ndarray,
    peaks: list[tuple[int, int, int]],
    ignore_labels: list[int],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Evaluate predictions and annotations at peak positions.

    Args:
        predictions: Prediction array of shape (X, Y, Z)
        annotations: Annotation array of shape (X, Y, Z)
        peaks: List of (x, y, z) coordinates of detected peaks
        ignore_labels: List of labels to ignore in evaluation
        num_classes: Number of classes

    Returns:
        Dictionary containing peak evaluation metrics
    """
    if not peaks:
        return {
            "peak_accuracy": 0.0,
            "peak_total_count": 0,
            "peak_correct_count": 0,
            "peak_confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64),
            "peak_class_distribution": {},
            "peak_prediction_distribution": {},
            "peak_precision": np.zeros(num_classes),
            "peak_recall": np.zeros(num_classes),
            "peak_f1": np.zeros(num_classes),
            "peak_macro_precision": 0.0,
            "peak_macro_recall": 0.0,
            "peak_macro_f1": 0.0,
        }

    # Extract predictions and annotations at peak positions
    peak_predictions = []
    peak_annotations = []

    for x, y, z in peaks:
        pred_label = predictions[x, y, z]
        ann_label = annotations[x, y, z]

        # Skip ignored labels
        if ann_label not in ignore_labels:
            peak_predictions.append(pred_label)
            peak_annotations.append(ann_label)

    if not peak_predictions:
        return {
            "peak_accuracy": 0.0,
            "peak_total_count": 0,
            "peak_correct_count": 0,
            "peak_confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64),
            "peak_class_distribution": {},
            "peak_prediction_distribution": {},
            "peak_precision": np.zeros(num_classes),
            "peak_recall": np.zeros(num_classes),
            "peak_f1": np.zeros(num_classes),
            "peak_macro_precision": 0.0,
            "peak_macro_recall": 0.0,
            "peak_macro_f1": 0.0,
        }

    peak_predictions = np.array(peak_predictions)
    peak_annotations = np.array(peak_annotations)

    # Calculate peak accuracy
    correct_predictions = np.sum(peak_predictions == peak_annotations)
    peak_accuracy = correct_predictions / len(peak_predictions)

    # Create confusion matrix for peaks
    peak_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, ann in zip(peak_predictions, peak_annotations):
        if 0 <= pred < num_classes and 0 <= ann < num_classes:
            peak_confusion_matrix[ann, pred] += 1

    # Calculate class distributions
    unique_ann, ann_counts = np.unique(peak_annotations, return_counts=True)
    peak_class_distribution = {
        int(label): int(count) for label, count in zip(unique_ann, ann_counts)
    }

    unique_pred, pred_counts = np.unique(peak_predictions, return_counts=True)
    peak_prediction_distribution = {
        int(label): int(count) for label, count in zip(unique_pred, pred_counts)
    }

    # Calculate precision, recall, F1 from confusion matrix
    valid_classes = [i for i in range(num_classes) if i not in ignore_labels]

    # Initialize arrays
    peak_precision = np.zeros(num_classes)
    peak_recall = np.zeros(num_classes)
    peak_f1 = np.zeros(num_classes)

    # Calculate TP, FP, FN for each class
    for class_id in valid_classes:
        tp = peak_confusion_matrix[class_id, class_id]
        fp = np.sum(peak_confusion_matrix[:, class_id]) - tp
        fn = np.sum(peak_confusion_matrix[class_id, :]) - tp

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        peak_precision[class_id] = precision
        peak_recall[class_id] = recall
        peak_f1[class_id] = f1

    # Calculate macro averages (only for valid classes)
    peak_macro_precision = (
        np.mean([peak_precision[i] for i in valid_classes]) if valid_classes else 0.0
    )
    peak_macro_recall = np.mean([peak_recall[i] for i in valid_classes]) if valid_classes else 0.0
    peak_macro_f1 = np.mean([peak_f1[i] for i in valid_classes]) if valid_classes else 0.0

    return {
        "peak_accuracy": peak_accuracy,
        "peak_total_count": len(peak_predictions),
        "peak_correct_count": int(correct_predictions),
        "peak_confusion_matrix": peak_confusion_matrix,
        "peak_class_distribution": peak_class_distribution,
        "peak_prediction_distribution": peak_prediction_distribution,
        "peak_precision": peak_precision,
        "peak_recall": peak_recall,
        "peak_f1": peak_f1,
        "peak_macro_precision": peak_macro_precision,
        "peak_macro_recall": peak_macro_recall,
        "peak_macro_f1": peak_macro_f1,
    }


def inference_single_sample(
    model: torch.nn.Module,
    device: torch.device,
    voxel_grid: np.ndarray,
    annotation: np.ndarray | None = None,
    config: Optional[TestConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a single voxel grid sample

    Args:
        model: Trained model
        device: Device to run inference on
        voxel_grid: Input voxel grid (X, Y, Z) format
        annotation: Ground truth annotation (optional, for comparison)
        config: Configuration object (optional, for threshold-based prediction)

    Returns:
        Dictionary containing prediction, confidence, and optionally ground truth
    """
    model.eval()

    with torch.no_grad():
        # Convert voxel grid to tensor and add batch dimension
        # From (X, Y, Z) to (1, 1, Z, Y, X) for UNet3D
        voxel_tensor = torch.from_numpy(voxel_grid).float()
        voxel_tensor = voxel_tensor.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        voxel_tensor = voxel_tensor.to(device)

        # Forward pass
        outputs = model(voxel_tensor)  # (1, num_classes, D, H, W)

        # Get prediction probabilities
        probabilities = torch.softmax(outputs, dim=1)  # (1, num_classes, D, H, W)

        # Apply threshold-based prediction if config is provided and enabled
        if (
            config
            and hasattr(config, "use_threshold_prediction")
            and config.use_threshold_prediction
        ):
            # Get the maximum probability and corresponding class for each voxel
            max_probs, argmax_predictions = torch.max(probabilities, dim=1)  # (1, D, H, W)

            # Create predictions with threshold filtering
            predictions = torch.where(
                max_probs >= config.prediction_threshold,
                argmax_predictions,
                torch.zeros_like(argmax_predictions),  # Assign noise (0) if below threshold
            )
        else:
            # Original argmax-based prediction
            predictions = torch.argmax(outputs, dim=1)  # (1, D, H, W)

        # Get confidence (max probability)
        confidence = torch.max(probabilities, dim=1)[0]  # (1, D, H, W)

        # Convert back to original format: (D, H, W) -> (W, H, D) = (X, Y, Z)
        prediction_np = predictions.squeeze(0).permute(2, 1, 0).cpu().numpy()
        confidence_np = confidence.squeeze(0).permute(2, 1, 0).cpu().numpy()

    result = {
        "prediction": prediction_np,
        "confidence": confidence_np,
    }

    if annotation is not None:
        result["ground_truth"] = annotation

    return result


def calculate_metrics(gt: np.ndarray, pred: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    """
    Calculate evaluation metrics

    Args:
        gt: Ground truth array
        pred: Prediction array
        num_classes: Number of classes

    Returns:
        Dictionary containing metrics
    """
    # Flatten arrays
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    # Filter out ignore label (0)
    mask = gt_flat != 0
    gt_filtered = gt_flat[mask]
    pred_filtered = pred_flat[mask]

    if len(gt_filtered) == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    # Calculate metrics
    accuracy = accuracy_score(gt_filtered, pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_filtered, pred_filtered, average="macro", zero_division=1
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def print_detailed_metrics(gt: np.ndarray, pred: np.ndarray, label_map: Dict[int, str]) -> None:
    """
    Print detailed classification metrics

    Args:
        gt: Ground truth array
        pred: Prediction array
        label_map: Mapping from class indices to names
    """
    # Flatten arrays
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    # Filter out ignore label (0)
    mask = gt_flat != 0
    gt_filtered = gt_flat[mask]
    pred_filtered = pred_flat[mask]

    if len(gt_filtered) == 0:
        print("No valid labels for evaluation")
        return

    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate([gt_filtered, pred_filtered]))
    target_names = [label_map.get(label, f"class_{label}") for label in unique_labels]

    print("\n" + "=" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 50)

    # Overall accuracy
    accuracy = accuracy_score(gt_filtered, pred_filtered)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Per-class metrics
    print("\nClassification Report:")
    print(
        classification_report(
            gt_filtered,
            pred_filtered,
            labels=unique_labels,
            target_names=target_names,
            digits=4,
            zero_division=1,
        )
    )

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(gt_filtered, pred_filtered, labels=unique_labels)
    print("Predicted ->")
    print("Actual \\/ " + "".join([f"{name:>8}" for name in target_names]))
    for i, actual_name in enumerate(target_names):
        print(f"{actual_name:>8}" + "".join([f"{cm[i, j]:>8}" for j in range(len(target_names))]))


class UNet3DPeakVisualizationTool:
    """
    Peak-based visualization tool for UNet3D inference results
    """

    def __init__(
        self,
        config_path: str,
        frame_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the peak visualization tool

        Args:
            config_path: Path to configuration file
            frame_id: Optional specific frame ID to display
            min_height: Minimum height threshold for peak detection
            min_distance: Minimum distance between peaks
        """
        # Load configuration
        self.config = load_config_from_yaml(config_path)
        if not isinstance(self.config, (TrainingConfig, TestConfig)):
            raise ValueError(f"config is not TrainingConfig or TestConfig: {self.config}")

        # Setup device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Display threshold configuration
        if (
            hasattr(self.config, "use_threshold_prediction")
            and self.config.use_threshold_prediction
        ):
            print(
                f"Threshold-based prediction enabled: threshold = {self.config.prediction_threshold}"
            )
            print("Predictions below threshold will be classified as 'noise' (label 0)")
        else:
            print("Using standard argmax-based prediction")

        # Load model
        self.model = get_model(self.config).to(self.device)
        print(f"Loading checkpoint from: {self.config.checkpoint_path}")
        if os.path.exists(self.config.checkpoint_path):
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        # Setup dataset from config
        if hasattr(self.config, "test_voxel_dirs") and hasattr(self.config, "test_annotation_dirs"):
            voxel_dirs = self.config.test_voxel_dirs
            annotation_dirs = self.config.test_annotation_dirs
        elif hasattr(self.config, "voxel_dirs") and hasattr(self.config, "annotation_dirs"):
            voxel_dirs = self.config.voxel_dirs
            annotation_dirs = self.config.annotation_dirs
        else:
            raise ValueError(
                "Config must contain voxel_dirs and annotation_dirs or test_voxel_dirs and test_annotation_dirs"
            )

        self.dataset = FWLDataset(
            voxel_dirs=voxel_dirs,
            annotation_dirs=annotation_dirs,
            target_size=self.config.target_size,
            downsample_z=self.config.downsample_z,
            divide=1,
            y_crop_top=self.config.y_crop_top,
            y_crop_bottom=self.config.y_crop_bottom,
            z_crop_front=self.config.z_crop_front,
            z_crop_back=self.config.z_crop_back,
        )

        print(f"Found {len(self.dataset)} samples")

        # Initialize frame management
        if frame_id:
            self.current_index = self._find_frame_index(frame_id)
            if self.current_index == -1:
                print(f"Frame ID '{frame_id}' not found. Starting from index 0.")
                self.current_index = 0
        else:
            self.current_index = 0

        # Initialize matplotlib
        setup_matplotlib_backend()
        self.fig = None
        self.axes = None

        # Cache for inference results and peaks
        self._inference_cache = {}
        self._peak_cache = {}

    def _find_frame_index(self, frame_id: str) -> int:
        """Find the index of a specific frame ID"""
        for i in range(len(self.dataset)):
            sample_info = self.dataset.get_sample_info(i)
            if sample_info["frame_id"] == frame_id:
                return i
        return -1

    def _get_inference_result(self, index: int) -> Dict[str, np.ndarray]:
        """Get inference result for a specific index (with caching)"""
        if index not in self._inference_cache:
            sample = self.dataset[index]
            result = inference_single_sample(
                self.model, self.device, sample["voxel_grid"], sample["annotation"], self.config
            )
            # Store raw input voxel data for peak detection and temporal histogram
            result["raw_input"] = sample["voxel_grid"]
            self._inference_cache[index] = result
        return self._inference_cache[index]

    def _get_peaks_and_evaluation(
        self, index: int
    ) -> Tuple[List[Tuple[int, int, int]], Dict[str, Any]]:
        """Get peaks and evaluation results for a specific index (with caching)"""
        if index not in self._peak_cache:
            result = self._get_inference_result(index)

            # Detect peaks in raw input
            peaks = detect_peaks_in_voxel(result["raw_input"])

            # Evaluate peaks if ground truth is available
            peak_evaluation = {}
            if "ground_truth" in result:
                # Get number of classes from config or infer from data
                num_classes = getattr(self.config, "num_classes", 4)
                ignore_labels = getattr(self.config, "ignore_visualize_labels", [])

                peak_evaluation = evaluate_peaks_at_positions(
                    predictions=result["prediction"],
                    annotations=result["ground_truth"],
                    peaks=peaks,
                    ignore_labels=ignore_labels,
                    num_classes=num_classes,
                )

            self._peak_cache[index] = (peaks, peak_evaluation)

        return self._peak_cache[index]

    def _create_peak_voxel_data(
        self, voxel_data: np.ndarray, peaks: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Create voxel data containing only peak positions"""
        peak_voxel_data = np.zeros_like(voxel_data)

        for x, y, z in peaks:
            if (
                0 <= x < voxel_data.shape[0]
                and 0 <= y < voxel_data.shape[1]
                and 0 <= z < voxel_data.shape[2]
            ):
                peak_voxel_data[x, y, z] = voxel_data[x, y, z]

        return peak_voxel_data

    def _plot_peak_voxel_3d(
        self,
        ax: Axes3D,
        voxel_data: np.ndarray,
        peaks: List[Tuple[int, int, int]],
        title: str,
        max_voxels: int = 5000,
    ) -> None:
        """
        Plot 3D voxel data showing only peak positions with class colors

        Args:
            ax: 3D axis to plot on
            voxel_data: Voxel data to plot
            peaks: List of peak positions
            title: Title for the plot
            max_voxels: Maximum number of voxels to display
        """
        ax.clear()

        # Set title first after clearing
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Get voxel grid dimensions
        max_x, max_y, max_z = voxel_data.shape

        # Draw boundary box wireframe
        self._draw_boundary_wireframe(ax, 0, 0, 0, max_x, max_y, max_z)

        if not peaks:
            print(f"  {title}: No peaks found")
            self._set_axis_properties(ax, max_x, max_y, max_z)
            return

        # Create peak-only voxel data
        peak_voxel_data = self._create_peak_voxel_data(voxel_data, peaks)

        # Get peak positions and values (exclude noise class 0)
        non_noise_mask = peak_voxel_data > 0
        occupied_positions = np.argwhere(non_noise_mask)

        if len(occupied_positions) > 0:
            occupied_values = peak_voxel_data[peak_voxel_data > 0]

            # Limit voxels for performance
            if len(occupied_positions) > max_voxels:
                indices = np.random.choice(len(occupied_positions), max_voxels, replace=False)
                occupied_positions = occupied_positions[indices]
                occupied_values = occupied_values[indices]

            # Extract coordinates with remapping: X->X, Y->Z, Histogram->Y
            x_coords = occupied_positions[:, 0]
            y_coords = occupied_positions[:, 1]
            z_coords = occupied_positions[:, 2]

            # Remap coordinates for display: (x, y, z) -> (x, z, y)
            display_x = x_coords
            display_y = z_coords  # Histogram bins become Y
            display_z = y_coords  # Y coordinates become Z (will be inverted)

            # Use class-specific colors for categorical data (excluding noise)
            # Filter out any noise class values that might have slipped through
            non_noise_mask = occupied_values > 0
            if np.any(non_noise_mask):
                filtered_positions = occupied_positions[non_noise_mask]
                filtered_values = occupied_values[non_noise_mask]

                # Update display coordinates
                display_x = filtered_positions[:, 0]
                display_y = filtered_positions[:, 2]  # Z coord
                display_z = filtered_positions[:, 1]  # Y coord

                colors = [CLASS_COLORS.get(int(val), "#808080") for val in filtered_values]
                scatter = ax.scatter(
                    display_x,
                    display_y,
                    display_z,
                    c=colors,
                    alpha=0.6,
                    s=8,
                )

                # Add legend for class colors
                self._add_class_legend(ax, filtered_values)
                self._set_axis_properties(ax, max_x, max_y, max_z)
                return scatter

        self._set_axis_properties(ax, max_x, max_y, max_z)
        return None

    def _set_axis_properties(self, ax: Axes3D, max_x: int, max_y: int, max_z: int) -> None:
        """Set axis labels and limits"""
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Histogram Bin")
        ax.set_zlabel("Y Coordinate")
        ax.set_xlim((-50, max_x + 50))
        ax.set_ylim((0, max_z))
        ax.set_zlim((max_y + 50, -50))
        ax.view_init(elev=20, azim=-60)

    def _add_class_legend(self, ax: Axes3D, class_values: np.ndarray) -> None:
        """
        Add legend for class colors

        Args:
            ax: 3D axis to add legend to
            class_values: Array of class values present in the data
        """
        # Get unique classes in the current data (exclude noise class 0)
        unique_classes = np.unique(class_values.astype(int))
        unique_classes = unique_classes[unique_classes > 0]  # Remove noise class (0)

        # Create legend elements for non-noise classes present in data
        legend_elements = []
        for class_id in unique_classes:
            if class_id in LABEL_MAP and class_id > 0:  # Exclude noise class
                color = CLASS_COLORS.get(class_id, "#808080")
                label = LABEL_MAP[class_id]
                legend_elements.append(Patch(facecolor=color, label=f"{class_id}: {label}"))

        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    def _draw_boundary_wireframe(
        self, ax: Axes3D, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int
    ) -> None:
        """Draw wireframe edges of the boundary box"""
        # Define the 8 vertices with coordinate remapping
        vertices = np.array(
            [
                [x_min, z_min, y_max],  # 0: left-front-top
                [x_max, z_min, y_max],  # 1: right-front-top
                [x_max, z_min, y_min],  # 2: right-front-bottom
                [x_min, z_min, y_min],  # 3: left-front-bottom
                [x_min, z_max, y_max],  # 4: left-back-top
                [x_max, z_max, y_max],  # 5: right-back-top
                [x_max, z_max, y_min],  # 6: right-back-bottom
                [x_min, z_max, y_min],  # 7: left-back-bottom
            ]
        )

        # Define the 12 edges
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # Front face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # Back face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # Connections
        ]

        # Draw edges
        for edge in edges:
            start, end = edge
            ax.plot3D(
                [vertices[start][0], vertices[end][0]],
                [vertices[start][1], vertices[end][1]],
                [vertices[start][2], vertices[end][2]],
                "k-",
                linewidth=1.0,
                alpha=0.3,
            )

    def _display_current_frame(self) -> None:
        """Display the current frame with peak-based inference results"""
        if len(self.dataset) == 0:
            print("No data to display")
            return

        # Get current sample info
        sample_info = self.dataset.get_sample_info(self.current_index)
        frame_id = sample_info["frame_id"]

        print(
            f"\nDisplaying frame: {sample_info['scene_id']}/{sample_info['hist_id']}/{frame_id} ({self.current_index + 1}/{len(self.dataset)})"
        )

        # Get inference results and peaks
        result = self._get_inference_result(self.current_index)
        peaks, peak_evaluation = self._get_peaks_and_evaluation(self.current_index)

        print(f"Detected {len(peaks)} peaks in raw input data")

        # Print per-class metrics at peak positions
        if "ground_truth" in result and peak_evaluation:
            num_classes = getattr(self.config, "num_classes", 4)
            ignore_labels = getattr(self.config, "ignore_visualize_labels", [])
            valid_classes = [i for i in range(num_classes) if i not in ignore_labels]

            print(f"\n=== Per-class Peak Metrics ===")
            for class_id in valid_classes:
                class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
                precision = peak_evaluation["peak_precision"][class_id]
                recall = peak_evaluation["peak_recall"][class_id]
                f1 = peak_evaluation["peak_f1"][class_id]
                print(
                    f"  {class_name} (ID: {class_id}): P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
                )

        # Create or update plot
        if self.fig is None:
            self._setup_plot()

        # Plot prediction peaks with class colors and legend
        try:
            scatter1 = self._plot_peak_voxel_3d(
                self.axes[0],
                result["prediction"],
                peaks,
                f"Peak Predictions",
            )
        except Exception as e:
            print(f"Warning: Could not plot prediction peaks: {e}")
            scatter1 = None

        # Plot ground truth peaks with class colors and legend if available
        scatter2 = None
        if "ground_truth" in result:
            try:
                scatter2 = self._plot_peak_voxel_3d(
                    self.axes[1],
                    result["ground_truth"],
                    peaks,
                    f"Peak Ground Truth",
                )
            except Exception as e:
                print(f"Warning: Could not plot ground truth peaks: {e}")
                scatter2 = None

        # Update display
        try:
            plt.draw()
        except Exception as e:
            print(f"Warning: Could not update display: {e}")

    def _setup_plot(self) -> None:
        """Setup the matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(14, 8))

        # Adjust layout to make room for titles
        self.fig.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)

        # Create 2 subplots side by side
        self.axes = [
            self.fig.add_subplot(121, projection="3d"),  # Peak Predictions
            self.fig.add_subplot(122, projection="3d"),  # Peak Ground Truth
        ]

        # Setup buttons
        ax_prev = plt.axes((0.1, 0.02, 0.1, 0.04))
        ax_next = plt.axes((0.21, 0.02, 0.1, 0.04))
        ax_temporal = plt.axes((0.32, 0.02, 0.15, 0.04))
        ax_peak_info = plt.axes((0.48, 0.02, 0.15, 0.04))

        self.btn_prev = Button(ax_prev, "← Previous")
        self.btn_next = Button(ax_next, "Next →")
        self.btn_temporal = Button(ax_temporal, "Temporal Histogram")
        self.btn_peak_info = Button(ax_peak_info, "Peak Info")

        self.btn_prev.on_clicked(self._prev_frame)
        self.btn_next.on_clicked(self._next_frame)
        self.btn_temporal.on_clicked(self._show_temporal_histogram)
        self.btn_peak_info.on_clicked(self._show_peak_info)

    def _next_frame(self, event: Any) -> None:
        """Move to next frame"""
        if self.current_index < len(self.dataset) - 1:
            self.current_index += 1
            self._display_current_frame()
        else:
            print("Already at the last frame")

    def _prev_frame(self, event: Any) -> None:
        """Move to previous frame"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current_frame()
        else:
            print("Already at the first frame")

    def _show_peak_info(self, event: Any) -> None:
        """Show detailed peak information for current frame"""
        if len(self.dataset) == 0:
            print("No data available")
            return

        # Get current sample info
        sample_info = self.dataset.get_sample_info(self.current_index)
        frame_id = sample_info["frame_id"]

        # Get peaks and evaluation
        peaks, peak_evaluation = self._get_peaks_and_evaluation(self.current_index)

        print(f"\n=== Peak Info for Frame: {frame_id} ({len(peaks)} peaks) ===")

        if peak_evaluation:
            num_classes = getattr(self.config, "num_classes", 4)
            ignore_labels = getattr(self.config, "ignore_visualize_labels", [])
            valid_classes = [i for i in range(num_classes) if i not in ignore_labels]

            for class_id in valid_classes:
                class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
                precision = peak_evaluation["peak_precision"][class_id]
                recall = peak_evaluation["peak_recall"][class_id]
                f1 = peak_evaluation["peak_f1"][class_id]
                print(
                    f"  {class_name} (ID: {class_id}): P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
                )

    def _show_temporal_histogram(self, event: Any) -> None:
        """Show temporal histogram for current frame at a peak location"""
        if len(self.dataset) == 0:
            print("No data available")
            return

        # Get current inference result and peaks
        result = self._get_inference_result(self.current_index)
        peaks, _ = self._get_peaks_and_evaluation(self.current_index)

        if "ground_truth" not in result:
            print("No ground truth annotation available for temporal analysis")
            return

        if not peaks:
            print("No peaks detected in current frame for temporal analysis")
            return

        # Get current sample info for display
        sample_info = self.dataset.get_sample_info(self.current_index)
        frame_id = sample_info["frame_id"]

        print(
            f"\nShowing temporal histogram for frame: {sample_info['scene_id']}/{sample_info['hist_id']}/{frame_id}"
        )

        # Select a random peak for temporal analysis
        selected_peak = peaks[np.random.randint(len(peaks))]
        sample_x, sample_y, sample_z = selected_peak

        print(
            f"Selected peak at coordinate ({sample_x}, {sample_y}, {sample_z}) for temporal analysis"
        )

        # Get raw input data - need to convert from (X, Y, Z) to (D, H, W) format for plot function
        raw_input = result["raw_input"]
        # Convert from (X, Y, Z) to (Z, Y, X) = (D, H, W) for temporal plot
        raw_input_dhw = raw_input.transpose(2, 1, 0)

        # Convert prediction and annotation to (D, H, W) format
        prediction_dhw = result["prediction"].transpose(2, 1, 0)
        annotation_dhw = result["ground_truth"].transpose(2, 1, 0)

        # Convert coordinates from (X,Y,Z) format to (D,H,W) format
        # Original: (X,Y,Z) -> Transposed: (D,H,W) = (Z,Y,X)
        # So coordinate mapping: x_original -> x_dhw, y_original -> y_dhw
        # In (D,H,W) format: x corresponds to W dimension, y corresponds to H dimension
        coord_x_dhw = sample_x  # X coordinate stays the same for W dimension
        coord_y_dhw = sample_y  # Y coordinate stays the same for H dimension

        try:
            # Create temporal histogram plot
            temporal_fig = plot_temporal_histogram(
                predictions=prediction_dhw,
                annotations=annotation_dhw,
                raw_input=raw_input_dhw,
                x=coord_x_dhw,
                y=coord_y_dhw,
                return_fig=True,
            )

            if temporal_fig:
                plt.show()
                print(
                    f"Temporal histogram displayed for peak coordinate ({sample_x}, {sample_y}, {sample_z}) [original (X,Y,Z) format]"
                )
            else:
                print("Failed to create temporal histogram")

        except Exception as e:
            print(f"Error creating temporal histogram: {e}")
            import traceback

            traceback.print_exc()

    def run(self) -> None:
        """Run the peak visualization tool"""
        print("Starting UNet3D Peak Visualization Tool")
        print("This tool shows only the voxels at detected peak positions")
        print("Use the Previous/Next buttons to navigate between frames")
        print(
            "Use the 'Temporal Histogram' button to show temporal analysis at a random peak location"
        )
        print("Use the 'Peak Info' button to show detailed peak information")
        print("Press Ctrl+C to exit")

        try:
            self._display_current_frame()
            plt.show()
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="UNet3D Peak-based Inference Visualization Tool")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--frame_id", type=str, help="Optional specific frame ID to display")

    args = parser.parse_args()

    # Validate paths
    if not pathlib.Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Create and run peak visualization tool
    tool = UNet3DPeakVisualizationTool(
        config_path=args.config,
        frame_id=args.frame_id,
    )

    tool.run()


if __name__ == "__main__":
    main()

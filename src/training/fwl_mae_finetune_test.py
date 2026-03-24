import argparse
import os
from typing import Any, Dict

import numpy as np
import polars as pl
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TestConfig, load_config_from_yaml
from src.config.config import TrainingConfig
from src.config.constants import LABEL_MAP
from src.data import FWLDataset, voxel_collate_fn
from src.utils import get_model, set_seed, set_wandb
from src.utils.log import log_info


def detect_peaks_in_voxel(raw_input: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Detect peaks in 3D voxel data.

    Args:
        raw_input: Raw input voxel data of shape (D, H, W)

    Returns:
        List of (d, y, x) coordinates of detected peaks
    """
    from scipy.signal import find_peaks

    peaks = []
    D, H, W = raw_input.shape

    # Find peaks in each temporal series (along depth dimension)
    for y in range(H):
        for x in range(W):
            time_series = raw_input[:, y, x]

            # Find peaks in the time series
            max_value = np.max(time_series)
            peak_indices, _ = find_peaks(time_series, height=max_value * 0.1, width=3)

            # Add peaks to the list
            for peak_d in peak_indices:
                peaks.append((peak_d, y, x))

    return peaks


def evaluate_peaks(
    predictions: np.ndarray,
    annotations: np.ndarray,
    peaks: list[tuple[int, int, int]],
    ignore_labels: list[int],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Evaluate predictions and annotations at peak positions.

    Args:
        predictions: Prediction array of shape (D, H, W)
        annotations: Annotation array of shape (D, H, W)
        peaks: List of (d, y, x) coordinates of detected peaks
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

    for d, y, x in peaks:
        pred_label = predictions[d, y, x]
        ann_label = annotations[d, y, x]

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


def test_model_voxel_mae_finetune(
    config: TestConfig,
    model: torch.nn.Module,
    device: torch.device,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    """Test VoxelMAE finetune model and calculate peak-based evaluation metrics"""
    model.eval()

    num_classes = 0

    # For peak evaluation
    overall_peak_metrics = {
        "peak_accuracy": 0.0,
        "peak_total_count": 0,
        "peak_correct_count": 0,
        "peak_confusion_matrix": None,
        "peak_class_distribution": {},
        "peak_prediction_distribution": {},
        "peak_precision": None,
        "peak_recall": None,
        "peak_f1": None,
        "peak_macro_precision": 0.0,
        "peak_macro_recall": 0.0,
        "peak_macro_f1": 0.0,
    }

    scene_id_peak_data = {}  # Dictionary to store per-scene_id peak data

    ignore_labels = torch.tensor(config.ignore_visualize_labels, device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(
                test_loader,
                total=len(test_loader),
                desc="Testing VoxelMAE Finetune",
            )
        ):
            # Extract voxel data and annotations (already in 3D UNet format)
            voxel_grids = batch["voxel_grids"]  # (B, C=1, D, H, W)
            annotations = batch["annotations"]  # (B, D, H, W)
            scene_ids = batch.get("scene_ids", [])  # List of scene_ids for the batch

            # Move to device (channel dimension already added in collate_fn)
            voxel_grids = voxel_grids.float().to(device)  # (B, C=1, D, H, W)
            annotations = annotations.long().to(device)  # (B, D, H, W)

            # Forward pass
            outputs = model(voxel_grids)  # (B, num_classes, D, H, W)

            if num_classes == 0:
                num_classes = outputs.shape[1]

            # Set ignore labels to very low values to prevent them from being predicted
            if ignore_labels.numel() > 0:
                outputs[:, ignore_labels] = -1e9

            # Apply threshold-based prediction if enabled
            if config.use_threshold_prediction:
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs, dim=1)  # (B, num_classes, D, H, W)

                # Get the maximum probability and corresponding class for each voxel
                max_probs, argmax_predictions = torch.max(probabilities, dim=1)  # (B, D, H, W)

                # Create predictions with threshold filtering
                predictions = torch.where(
                    max_probs >= config.prediction_threshold,
                    argmax_predictions,
                    torch.zeros_like(argmax_predictions),  # Assign noise (0) if below threshold
                )
            else:
                # Original argmax-based prediction
                predictions = torch.argmax(outputs, dim=1)  # (B, D, H, W)

            # Convert to numpy for metric calculation
            predictions_np = predictions.cpu().numpy()
            annotations_np = annotations.cpu().numpy()

            # Process each sample in the batch
            for i in range(predictions_np.shape[0]):
                pred = predictions_np[i]
                target = annotations_np[i]
                scene_id = scene_ids[i] if i < len(scene_ids) else "unknown"
                raw_input_sample = voxel_grids[i, 0].cpu().numpy()  # Shape: (D, H, W)

                # Initialize scene_id peak data if not exists
                if scene_id not in scene_id_peak_data:
                    scene_id_peak_data[scene_id] = {
                        "peak_accuracy": 0.0,
                        "peak_total_count": 0,
                        "peak_correct_count": 0,
                        "peak_confusion_matrix": np.zeros(
                            (num_classes, num_classes), dtype=np.int64
                        ),
                        "peak_class_distribution": {},
                        "peak_prediction_distribution": {},
                        "peak_precision": np.zeros(num_classes),
                        "peak_recall": np.zeros(num_classes),
                        "peak_f1": np.zeros(num_classes),
                        "peak_macro_precision": 0.0,
                        "peak_macro_recall": 0.0,
                        "peak_macro_f1": 0.0,
                    }

                # Peak evaluation for this sample
                peaks = detect_peaks_in_voxel(raw_input_sample)
                if peaks:
                    # Peak evaluation with precision, recall, F1 calculation
                    peak_eval = evaluate_peaks(
                        predictions=pred,
                        annotations=target,
                        peaks=peaks,
                        ignore_labels=config.ignore_visualize_labels,
                        num_classes=num_classes,
                    )

                    # Update overall peak metrics
                    overall_peak_metrics["peak_total_count"] += peak_eval["peak_total_count"]
                    overall_peak_metrics["peak_correct_count"] += peak_eval["peak_correct_count"]

                    # Initialize overall peak confusion matrix if needed
                    if overall_peak_metrics["peak_confusion_matrix"] is None:
                        overall_peak_metrics["peak_confusion_matrix"] = np.zeros(
                            (num_classes, num_classes), dtype=np.int64
                        )

                    # Update overall peak confusion matrix
                    overall_peak_metrics["peak_confusion_matrix"] += peak_eval[
                        "peak_confusion_matrix"
                    ]

                    # Update overall peak class distributions
                    for label, count in peak_eval["peak_class_distribution"].items():
                        overall_peak_metrics["peak_class_distribution"][label] = (
                            overall_peak_metrics["peak_class_distribution"].get(label, 0) + count
                        )

                    for label, count in peak_eval["peak_prediction_distribution"].items():
                        overall_peak_metrics["peak_prediction_distribution"][label] = (
                            overall_peak_metrics["peak_prediction_distribution"].get(label, 0)
                            + count
                        )

                    # Update scene_id specific peak metrics
                    scene_id_peak_data[scene_id]["peak_total_count"] += peak_eval[
                        "peak_total_count"
                    ]
                    scene_id_peak_data[scene_id]["peak_correct_count"] += peak_eval[
                        "peak_correct_count"
                    ]
                    scene_id_peak_data[scene_id]["peak_confusion_matrix"] += peak_eval[
                        "peak_confusion_matrix"
                    ]

                    # Update scene_id peak class distributions
                    for label, count in peak_eval["peak_class_distribution"].items():
                        scene_id_peak_data[scene_id]["peak_class_distribution"][label] = (
                            scene_id_peak_data[scene_id]["peak_class_distribution"].get(label, 0)
                            + count
                        )

                    for label, count in peak_eval["peak_prediction_distribution"].items():
                        scene_id_peak_data[scene_id]["peak_prediction_distribution"][label] = (
                            scene_id_peak_data[scene_id]["peak_prediction_distribution"].get(
                                label, 0
                            )
                            + count
                        )

            # Clear GPU memory after each batch
            del outputs, predictions, voxel_grids, annotations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Calculate final peak metrics
    valid_classes = [i for i in range(num_classes) if i not in config.ignore_visualize_labels]

    if overall_peak_metrics["peak_total_count"] > 0:
        overall_peak_metrics["peak_accuracy"] = (
            overall_peak_metrics["peak_correct_count"] / overall_peak_metrics["peak_total_count"]
        )

        # Calculate overall precision, recall, F1 from confusion matrix
        overall_peak_confusion_matrix = overall_peak_metrics["peak_confusion_matrix"]

        # Initialize arrays
        overall_peak_precision = np.zeros(num_classes)
        overall_peak_recall = np.zeros(num_classes)
        overall_peak_f1 = np.zeros(num_classes)

        # Calculate TP, FP, FN for each class
        for class_id in valid_classes:
            tp = overall_peak_confusion_matrix[class_id, class_id]
            fp = np.sum(overall_peak_confusion_matrix[:, class_id]) - tp
            fn = np.sum(overall_peak_confusion_matrix[class_id, :]) - tp

            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            overall_peak_precision[class_id] = precision
            overall_peak_recall[class_id] = recall
            overall_peak_f1[class_id] = f1

        # Calculate macro averages (only for valid classes)
        overall_peak_metrics["peak_precision"] = overall_peak_precision
        overall_peak_metrics["peak_recall"] = overall_peak_recall
        overall_peak_metrics["peak_f1"] = overall_peak_f1
        overall_peak_metrics["peak_macro_precision"] = (
            np.mean([overall_peak_precision[i] for i in valid_classes]) if valid_classes else 0.0
        )
        overall_peak_metrics["peak_macro_recall"] = (
            np.mean([overall_peak_recall[i] for i in valid_classes]) if valid_classes else 0.0
        )
        overall_peak_metrics["peak_macro_f1"] = (
            np.mean([overall_peak_f1[i] for i in valid_classes]) if valid_classes else 0.0
        )

    # Calculate per-scene_id peak metrics
    scene_id_peak_metrics = {}
    for scene_id, data in scene_id_peak_data.items():
        if data["peak_total_count"] > 0:
            peak_accuracy = data["peak_correct_count"] / data["peak_total_count"]

            # Calculate precision, recall, F1 from confusion matrix for this scene_id
            scene_peak_confusion_matrix = data["peak_confusion_matrix"]

            # Initialize arrays
            scene_peak_precision = np.zeros(num_classes)
            scene_peak_recall = np.zeros(num_classes)
            scene_peak_f1 = np.zeros(num_classes)

            # Calculate TP, FP, FN for each class
            for class_id in valid_classes:
                tp = scene_peak_confusion_matrix[class_id, class_id]
                fp = np.sum(scene_peak_confusion_matrix[:, class_id]) - tp
                fn = np.sum(scene_peak_confusion_matrix[class_id, :]) - tp

                # Calculate precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                scene_peak_precision[class_id] = precision
                scene_peak_recall[class_id] = recall
                scene_peak_f1[class_id] = f1

            # Calculate macro averages (only for valid classes)
            peak_macro_precision = (
                np.mean([scene_peak_precision[i] for i in valid_classes]) if valid_classes else 0.0
            )
            peak_macro_recall = (
                np.mean([scene_peak_recall[i] for i in valid_classes]) if valid_classes else 0.0
            )
            peak_macro_f1 = (
                np.mean([scene_peak_f1[i] for i in valid_classes]) if valid_classes else 0.0
            )
        else:
            peak_accuracy = 0.0
            scene_peak_precision = np.zeros(num_classes)
            scene_peak_recall = np.zeros(num_classes)
            scene_peak_f1 = np.zeros(num_classes)
            peak_macro_precision = 0.0
            peak_macro_recall = 0.0
            peak_macro_f1 = 0.0

        scene_id_peak_metrics[scene_id] = {
            "peak_accuracy": peak_accuracy,
            "peak_total_count": data["peak_total_count"],
            "peak_correct_count": data["peak_correct_count"],
            "peak_confusion_matrix": data["peak_confusion_matrix"],
            "peak_class_distribution": data["peak_class_distribution"],
            "peak_prediction_distribution": data["peak_prediction_distribution"],
            "peak_precision": scene_peak_precision,
            "peak_recall": scene_peak_recall,
            "peak_f1": scene_peak_f1,
            "peak_macro_precision": peak_macro_precision,
            "peak_macro_recall": peak_macro_recall,
            "peak_macro_f1": peak_macro_f1,
        }

    # Calculate average peak metrics across all scene_ids
    if scene_id_peak_metrics:
        # Filter out scene_ids with no peaks for average calculation
        valid_peak_metrics = {
            k: v for k, v in scene_id_peak_metrics.items() if v["peak_total_count"] > 0
        }

        if valid_peak_metrics:
            avg_peak_accuracy = np.mean(
                [metrics["peak_accuracy"] for metrics in valid_peak_metrics.values()]
            )
            avg_peak_macro_precision = np.mean(
                [metrics["peak_macro_precision"] for metrics in valid_peak_metrics.values()]
            )
            avg_peak_macro_recall = np.mean(
                [metrics["peak_macro_recall"] for metrics in valid_peak_metrics.values()]
            )
            avg_peak_macro_f1 = (
                2
                * avg_peak_macro_precision
                * avg_peak_macro_recall
                / (avg_peak_macro_precision + avg_peak_macro_recall)
                if (avg_peak_macro_precision + avg_peak_macro_recall) > 0
                else 0.0
            )
            # Calculate per-class peak averages across scene_ids
            avg_peak_per_class_precision = np.zeros(num_classes)
            avg_peak_per_class_recall = np.zeros(num_classes)
            avg_peak_per_class_f1 = np.zeros(num_classes)

            for class_id in valid_classes:
                class_peak_precisions = [
                    metrics["peak_precision"][class_id] for metrics in valid_peak_metrics.values()
                ]
                class_peak_recalls = [
                    metrics["peak_recall"][class_id] for metrics in valid_peak_metrics.values()
                ]

                avg_peak_per_class_precision[class_id] = np.mean(class_peak_precisions)
                avg_peak_per_class_recall[class_id] = np.mean(class_peak_recalls)
                p = avg_peak_per_class_precision[class_id]
                r = avg_peak_per_class_recall[class_id]
                avg_peak_per_class_f1[class_id] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            scene_id_peak_average_metrics = {
                "peak_accuracy": avg_peak_accuracy,
                "peak_total_count": sum(
                    metrics["peak_total_count"] for metrics in scene_id_peak_metrics.values()
                ),
                "peak_correct_count": sum(
                    metrics["peak_correct_count"] for metrics in scene_id_peak_metrics.values()
                ),
                "peak_macro_precision": avg_peak_macro_precision,
                "peak_macro_recall": avg_peak_macro_recall,
                "peak_macro_f1": avg_peak_macro_f1,
                "peak_precision": avg_peak_per_class_precision,
                "peak_recall": avg_peak_per_class_recall,
                "peak_f1": avg_peak_per_class_f1,
            }
        else:
            scene_id_peak_average_metrics = {
                "peak_accuracy": 0.0,
                "peak_total_count": 0,
                "peak_correct_count": 0,
                "peak_macro_precision": 0.0,
                "peak_macro_recall": 0.0,
                "peak_macro_f1": 0.0,
                "peak_precision": np.zeros(num_classes),
                "peak_recall": np.zeros(num_classes),
                "peak_f1": np.zeros(num_classes),
            }
    else:
        scene_id_peak_average_metrics = {
            "peak_accuracy": 0.0,
            "peak_total_count": 0,
            "peak_correct_count": 0,
            "peak_macro_precision": 0.0,
            "peak_macro_recall": 0.0,
            "peak_macro_f1": 0.0,
            "peak_precision": np.zeros(num_classes),
            "peak_recall": np.zeros(num_classes),
            "peak_f1": np.zeros(num_classes),
        }

    # Configure Polars display settings for better table formatting
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(2000)

    # === Log Table 1: Average Peak Metrics Across All Scene_IDs ===
    if scene_id_peak_average_metrics["peak_total_count"] > 0:
        avg_peak_class_metrics = {
            "Class": [],
            "Peak_Precision": [],
            "Peak_Recall": [],
            "Peak_F1": [],
            "Peak_GT_Count": [],
            "Peak_Pred_Count": [],
        }

        avg_peak_precision_arr = scene_id_peak_average_metrics["peak_precision"]
        avg_peak_recall_arr = scene_id_peak_average_metrics["peak_recall"]
        avg_peak_f1_arr = scene_id_peak_average_metrics["peak_f1"]

        for class_id in valid_classes:
            class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
            avg_peak_class_metrics["Class"].append(class_name)
            avg_peak_class_metrics["Peak_Precision"].append(float(avg_peak_precision_arr[class_id]))
            avg_peak_class_metrics["Peak_Recall"].append(float(avg_peak_recall_arr[class_id]))
            avg_peak_class_metrics["Peak_F1"].append(float(avg_peak_f1_arr[class_id]))
            avg_peak_class_metrics["Peak_GT_Count"].append(
                int(
                    sum(
                        metrics["peak_class_distribution"].get(class_id, 0)
                        for metrics in scene_id_peak_metrics.values()
                    )
                )
            )
            avg_peak_class_metrics["Peak_Pred_Count"].append(
                int(
                    sum(
                        metrics["peak_prediction_distribution"].get(class_id, 0)
                        for metrics in scene_id_peak_metrics.values()
                    )
                )
            )

        # Add macro averages
        avg_peak_class_metrics["Class"].append("Macro avg")
        avg_peak_class_metrics["Peak_Precision"].append(
            float(scene_id_peak_average_metrics["peak_macro_precision"])
        )
        avg_peak_class_metrics["Peak_Recall"].append(
            float(scene_id_peak_average_metrics["peak_macro_recall"])
        )
        avg_peak_class_metrics["Peak_F1"].append(
            float(scene_id_peak_average_metrics["peak_macro_f1"])
        )
        avg_peak_class_metrics["Peak_GT_Count"].append(None)
        avg_peak_class_metrics["Peak_Pred_Count"].append(None)

        df_avg_peak_class = pl.DataFrame(
            avg_peak_class_metrics,
            schema={
                "Class": pl.Utf8,
                "Peak_Precision": pl.Float64,
                "Peak_Recall": pl.Float64,
                "Peak_F1": pl.Float64,
                "Peak_GT_Count": pl.Int64,
                "Peak_Pred_Count": pl.Int64,
            },
        )

        df_avg_peak_class_display = df_avg_peak_class.with_columns(
            [
                pl.col("Peak_Precision").round(6),
                pl.col("Peak_Recall").round(6),
                pl.col("Peak_F1").round(6),
                pl.col("Peak_GT_Count").cast(pl.Utf8).str.replace("null", "null"),
                pl.col("Peak_Pred_Count").cast(pl.Utf8).str.replace("null", "null"),
            ]
        )

        log_info(f"\nAverage Peak Metrics Across All Scene_IDs\n {df_avg_peak_class_display}")

    # === Log Table 2: Detailed Peak Metrics per Scene_ID ===
    for scene_id, peak_metrics in scene_id_peak_metrics.items():
        if peak_metrics["peak_total_count"] > 0:
            scene_peak_class_metrics = {
                "Class": [],
                "Peak_Precision": [],
                "Peak_Recall": [],
                "Peak_F1": [],
                "Peak_GT_Count": [],
                "Peak_Pred_Count": [],
            }

            peak_precision_arr = peak_metrics["peak_precision"]
            peak_recall_arr = peak_metrics["peak_recall"]
            peak_f1_arr = peak_metrics["peak_f1"]
            peak_class_dist = peak_metrics["peak_class_distribution"]
            peak_pred_dist = peak_metrics["peak_prediction_distribution"]

            for class_id in valid_classes:
                class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
                scene_peak_class_metrics["Class"].append(class_name)
                scene_peak_class_metrics["Peak_Precision"].append(
                    float(peak_precision_arr[class_id])
                )
                scene_peak_class_metrics["Peak_Recall"].append(float(peak_recall_arr[class_id]))
                scene_peak_class_metrics["Peak_F1"].append(float(peak_f1_arr[class_id]))
                scene_peak_class_metrics["Peak_GT_Count"].append(
                    int(peak_class_dist.get(class_id, 0))
                )
                scene_peak_class_metrics["Peak_Pred_Count"].append(
                    int(peak_pred_dist.get(class_id, 0))
                )

            # Add macro averages
            scene_peak_class_metrics["Class"].append("Macro avg")
            scene_peak_class_metrics["Peak_Precision"].append(
                float(peak_metrics["peak_macro_precision"])
            )
            scene_peak_class_metrics["Peak_Recall"].append(float(peak_metrics["peak_macro_recall"]))
            scene_peak_class_metrics["Peak_F1"].append(float(peak_metrics["peak_macro_f1"]))
            scene_peak_class_metrics["Peak_GT_Count"].append(None)
            scene_peak_class_metrics["Peak_Pred_Count"].append(None)

            df_scene_peak_class = pl.DataFrame(
                scene_peak_class_metrics,
                schema={
                    "Class": pl.Utf8,
                    "Peak_Precision": pl.Float64,
                    "Peak_Recall": pl.Float64,
                    "Peak_F1": pl.Float64,
                    "Peak_GT_Count": pl.Int64,
                    "Peak_Pred_Count": pl.Int64,
                },
            )

            df_scene_peak_class_display = df_scene_peak_class.with_columns(
                [
                    pl.col("Peak_Precision").round(6),
                    pl.col("Peak_Recall").round(6),
                    pl.col("Peak_F1").round(6),
                    pl.col("Peak_GT_Count").cast(pl.Utf8).str.replace("null", "null"),
                    pl.col("Peak_Pred_Count").cast(pl.Utf8).str.replace("null", "null"),
                ]
            )

            log_info(
                f"\nDetailed Peak Metrics for Scene ID: {scene_id}\n {df_scene_peak_class_display}"
            )

    # Log peak metrics to WandB if logging is enabled
    if config.is_log:
        wandb_metrics = {
            "test_peak_accuracy": overall_peak_metrics["peak_accuracy"],
            "test_peak_total_count": overall_peak_metrics["peak_total_count"],
            "test_peak_correct_count": overall_peak_metrics["peak_correct_count"],
            "test_peak_macro_precision": overall_peak_metrics["peak_macro_precision"],
            "test_peak_macro_recall": overall_peak_metrics["peak_macro_recall"],
            "test_peak_macro_f1": overall_peak_metrics["peak_macro_f1"],
        }

        # Add per-class peak metrics
        if overall_peak_metrics["peak_precision"] is not None:
            for class_id in valid_classes:
                class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
                wandb_metrics[f"test_peak_precision_{class_name}"] = float(
                    overall_peak_metrics["peak_precision"][class_id]
                )
                wandb_metrics[f"test_peak_recall_{class_name}"] = float(
                    overall_peak_metrics["peak_recall"][class_id]
                )
                wandb_metrics[f"test_peak_f1_{class_name}"] = float(
                    overall_peak_metrics["peak_f1"][class_id]
                )

        # Add peak class distribution metrics
        for label, count in overall_peak_metrics["peak_class_distribution"].items():
            class_name = LABEL_MAP.get(label, f"Class_{label}")
            wandb_metrics[f"test_peak_gt_count_{class_name}"] = count

        for label, count in overall_peak_metrics["peak_prediction_distribution"].items():
            class_name = LABEL_MAP.get(label, f"Class_{label}")
            wandb_metrics[f"test_peak_pred_count_{class_name}"] = count

        # Add average peak metrics across all scene_ids
        wandb_metrics["test_avg_peak_accuracy"] = scene_id_peak_average_metrics["peak_accuracy"]
        wandb_metrics["test_avg_peak_macro_precision"] = scene_id_peak_average_metrics[
            "peak_macro_precision"
        ]
        wandb_metrics["test_avg_peak_macro_recall"] = scene_id_peak_average_metrics[
            "peak_macro_recall"
        ]
        wandb_metrics["test_avg_peak_macro_f1"] = scene_id_peak_average_metrics["peak_macro_f1"]

        # Add average per-class peak metrics across all scene_ids
        for class_id in valid_classes:
            class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
            wandb_metrics[f"test_avg_peak_precision_{class_name}"] = float(
                scene_id_peak_average_metrics["peak_precision"][class_id]
            )
            wandb_metrics[f"test_avg_peak_recall_{class_name}"] = float(
                scene_id_peak_average_metrics["peak_recall"][class_id]
            )
            wandb_metrics[f"test_avg_peak_f1_{class_name}"] = float(
                scene_id_peak_average_metrics["peak_f1"][class_id]
            )

        # Add per-scene_id peak metrics
        for scene_id, peak_metrics in scene_id_peak_metrics.items():
            prefix = f"test_scene_{scene_id}"
            wandb_metrics[f"{prefix}_peak_accuracy"] = peak_metrics["peak_accuracy"]
            wandb_metrics[f"{prefix}_peak_total_count"] = peak_metrics["peak_total_count"]
            wandb_metrics[f"{prefix}_peak_correct_count"] = peak_metrics["peak_correct_count"]
            wandb_metrics[f"{prefix}_peak_macro_precision"] = peak_metrics["peak_macro_precision"]
            wandb_metrics[f"{prefix}_peak_macro_recall"] = peak_metrics["peak_macro_recall"]
            wandb_metrics[f"{prefix}_peak_macro_f1"] = peak_metrics["peak_macro_f1"]

            # Add per-class peak metrics for each scene_id
            for class_id in valid_classes:
                class_name = LABEL_MAP.get(class_id, f"Class_{class_id}")
                wandb_metrics[f"{prefix}_peak_precision_{class_name}"] = float(
                    peak_metrics["peak_precision"][class_id]
                )
                wandb_metrics[f"{prefix}_peak_recall_{class_name}"] = float(
                    peak_metrics["peak_recall"][class_id]
                )
                wandb_metrics[f"{prefix}_peak_f1_{class_name}"] = float(
                    peak_metrics["peak_f1"][class_id]
                )

            # Add peak class distribution for each scene_id
            for label, count in peak_metrics["peak_class_distribution"].items():
                class_name = LABEL_MAP.get(label, f"Class_{label}")
                wandb_metrics[f"{prefix}_peak_gt_count_{class_name}"] = count

            for label, count in peak_metrics["peak_prediction_distribution"].items():
                class_name = LABEL_MAP.get(label, f"Class_{label}")
                wandb_metrics[f"{prefix}_peak_pred_count_{class_name}"] = count

        wandb.log(wandb_metrics)
        log_info("Test metrics logged to WandB")

    # Return summary metrics
    results = {
        "peak_evaluation": overall_peak_metrics,
        "scene_id_peak_metrics": scene_id_peak_metrics,
        "scene_id_peak_average_metrics": scene_id_peak_average_metrics,
    }

    return results


def test_fwl_mae_finetune(config_path: str) -> Dict[str, Any]:
    """Main test function for FWLMAE finetune"""
    # Load configuration
    config: TrainingConfig | TestConfig = load_config_from_yaml(config_path)
    if not isinstance(config, TestConfig):
        raise ValueError(f"config is not TestConfig: {config}")

    log_info(f"Test Configuration:")
    log_info(f"  Model checkpoint: {config.checkpoint_path}")
    log_info(f"  Batch size: {config.batch_size}")
    log_info(f"  Device: {config.device}")
    log_info(
        f"  Ignored labels: {config.ignore_visualize_labels} ({[LABEL_MAP.get(i, f'Class_{i}') for i in config.ignore_visualize_labels]})"
    )
    log_info(f"  Use threshold prediction: {config.use_threshold_prediction}")
    if config.use_threshold_prediction:
        log_info(f"  Prediction threshold: {config.prediction_threshold}")
        log_info("  Predictions below threshold will be classified as 'noise' (label 0)")

    # Set seed for reproducibility
    set_seed(config.seed)

    # Set up wandb if logging is enabled
    if config.is_log:
        set_wandb(config)

    # Device setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    # Load model
    model = get_model(config).to(device)

    # Load checkpoint
    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")

    log_info(f"Loading checkpoint from: {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    log_info(f"Total model parameters: {total_params:,}")

    # Create test dataset
    test_dataset = FWLDataset(
        voxel_dirs=config.test_voxel_dirs,
        annotation_dirs=config.test_annotation_dirs,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        divide=config.divide,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=voxel_collate_fn,
    )

    log_info(f"Test dataset size: {len(test_dataset)}")
    log_info(f"Test batches: {len(test_loader)}")
    log_info(
        f"Memory optimization: batch_size={config.batch_size}, num_workers={config.num_workers}"
    )

    # Run testing
    results = test_model_voxel_mae_finetune(
        config=config,
        model=model,
        device=device,
        test_loader=test_loader,
    )

    return results


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test FWLMAE finetune model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to test configuration YAML file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    results = test_fwl_mae_finetune(args.config)

    # Print final summary
    log_info("=" * 50)
    log_info("FINAL TEST RESULTS SUMMARY (Peak Evaluation)")
    log_info("=" * 50)

    peak_eval = results["peak_evaluation"]
    log_info(f"Peak Accuracy: {peak_eval['peak_accuracy']:.4f}")
    log_info(f"Total Peaks Detected: {peak_eval['peak_total_count']}")
    log_info(f"Correctly Predicted Peaks: {peak_eval['peak_correct_count']}")
    log_info(f"Peak Macro Precision: {peak_eval['peak_macro_precision']:.4f}")
    log_info(f"Peak Macro Recall: {peak_eval['peak_macro_recall']:.4f}")
    log_info(f"Peak Macro F1-Score: {peak_eval['peak_macro_f1']:.4f}")

    avg_peak_metrics = results["scene_id_peak_average_metrics"]
    log_info(f"\nAverage Peak Evaluation Across All Scene_IDs:")
    log_info(f"  Average Peak Accuracy: {avg_peak_metrics['peak_accuracy']:.4f}")
    log_info(f"  Average Peak Macro Precision: {avg_peak_metrics['peak_macro_precision']:.4f}")
    log_info(f"  Average Peak Macro Recall: {avg_peak_metrics['peak_macro_recall']:.4f}")
    log_info(f"  Average Peak Macro F1-Score: {avg_peak_metrics['peak_macro_f1']:.4f}")

    log_info("=" * 50)


if __name__ == "__main__":
    main()

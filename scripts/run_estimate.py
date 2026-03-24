import argparse
import os
import pathlib
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import TestConfig, load_config_from_yaml
from src.utils import (
    downsample_histogram_direction,
    get_model,
    load_blosc2,
    save_blosc2,
    set_seed,
    upsample_histogram_direction_zero_padding,
)
from src.utils.log import log_info


class SimpleFWLDataset:
    """Simple dataset for inference on voxel files"""

    def __init__(
        self,
        voxel_dirs: List[str],
        downsample_z: Optional[int] = None,
        y_crop_top: int = 0,
        y_crop_bottom: int = 0,
        z_crop_front: int = 0,
        z_crop_back: int = 0,
        voxel_pattern: str = "*_voxel.b2",
    ) -> None:
        self.voxel_dirs = [pathlib.Path(d) for d in voxel_dirs]
        self.downsample_z = downsample_z
        self.y_crop_top = y_crop_top
        self.y_crop_bottom = y_crop_bottom
        self.z_crop_front = z_crop_front
        self.z_crop_back = z_crop_back
        # Get all voxel files recursively from voxel_dir and subdirectories
        all_files = []
        for voxel_dir in self.voxel_dirs:
            files = list(voxel_dir.glob(voxel_pattern))
            all_files.extend(files)
        self.voxel_files = sorted(all_files)

    def __len__(self) -> int:
        return len(self.voxel_files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        voxel_file = self.voxel_files[index]
        frame_id = voxel_file.stem.replace("_voxel", "")

        # Load voxel grid
        voxel_grid = load_blosc2(str(voxel_file))

        # Apply preprocessing
        voxel_grid = self._apply_y_crop(voxel_grid)
        voxel_grid = self._apply_z_crop(voxel_grid)
        original_shape = voxel_grid.shape

        # Store shape after cropping but before downsampling
        cropped_shape = voxel_grid.shape

        if self.downsample_z is not None:
            voxel_grid = downsample_histogram_direction(voxel_grid, self.downsample_z)

        return {
            "frame_id": frame_id,
            "voxel_file": voxel_file,
            "voxel_grid": voxel_grid,
            "original_shape": original_shape,
            "cropped_shape": cropped_shape,
        }

    def _apply_y_crop(self, voxel_grid: np.ndarray) -> np.ndarray:
        """Apply Y axis cropping"""
        if self.y_crop_top == 0 and self.y_crop_bottom == 0:
            return voxel_grid

        y_size = voxel_grid.shape[1]
        y_start = self.y_crop_bottom
        y_end = y_size - self.y_crop_top

        if y_start >= y_end:
            raise ValueError(
                f"Y cropping parameters too large: {self.y_crop_bottom}, {self.y_crop_top}"
            )

        return voxel_grid[:, y_start:y_end, :]

    def _apply_z_crop(self, voxel_grid: np.ndarray) -> np.ndarray:
        """Apply Z axis cropping"""
        if self.z_crop_front == 0:
            return voxel_grid

        z_size = voxel_grid.shape[2]
        z_start = self.z_crop_front
        z_end = z_size - self.z_crop_back

        if z_start >= z_end:
            raise ValueError(
                f"Z cropping parameter too large: {self.z_crop_front}, {self.z_crop_back}"
            )

        return voxel_grid[:, :, z_start:z_end]


class SlidingWindowInference:
    """Sliding window inference for fixed-size model input"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        target_size: List[int],  # [X, Y, Z] - model input size
        config: TestConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.target_size = target_size  # [X, Y, Z]
        self.config = config
        self.model.eval()

    def _get_window_positions(self, data_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Calculate sliding window positions for X, Y directions only"""
        positions = []
        x_size, y_size, z_size = data_shape
        win_x, win_y, win_z = self.target_size

        # Calculate number of windows needed in X, Y directions only
        num_windows_x = (x_size + win_x - 1) // win_x  # Ceiling division
        num_windows_y = (y_size + win_y - 1) // win_y

        for i in range(num_windows_x):
            for j in range(num_windows_y):
                start_x = i * win_x
                start_y = j * win_y
                start_z = 0  # Always start from z=0

                # Ensure we don't go beyond data boundaries
                start_x = min(start_x, x_size - win_x) if x_size >= win_x else 0
                start_y = min(start_y, y_size - win_y) if y_size >= win_y else 0

                positions.append((start_x, start_y, start_z))

        return positions

    def _extract_window(self, data: np.ndarray, position: Tuple[int, int, int]) -> np.ndarray:
        """Extract window from data, padding if necessary"""
        start_x, start_y, start_z = position
        win_x, win_y, win_z = self.target_size

        end_x = start_x + win_x
        end_y = start_y + win_y
        # Use full Z dimension (already downsampled)
        end_z = data.shape[2]

        # Extract available data
        actual_end_x = min(end_x, data.shape[0])
        actual_end_y = min(end_y, data.shape[1])

        window = data[start_x:actual_end_x, start_y:actual_end_y, :]
        # Pad X, Y if necessary, but keep Z dimension as is
        target_shape = (win_x, win_y, data.shape[2])
        if window.shape != target_shape:
            padded_window = np.zeros(target_shape, dtype=window.dtype)
            padded_window[: window.shape[0], : window.shape[1], :] = window
            window = padded_window

        return window

    def _inference_single_window(self, window: np.ndarray) -> np.ndarray:
        """Run inference on a single window"""
        with torch.no_grad():
            # Convert to tensor: (X, Y, Z) -> (1, 1, Z, Y, X)
            window_tensor = torch.from_numpy(window).float()
            window_tensor = window_tensor.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
            window_tensor = window_tensor.to(self.device)

            # Forward pass
            outputs = self.model(window_tensor)  # (1, num_classes, Z, Y, X)
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)

            if (
                hasattr(self.config, "use_threshold_prediction")
                and self.config.use_threshold_prediction
            ):
                max_probs, predictions = torch.max(probabilities, dim=1)
                predictions = torch.where(
                    max_probs >= self.config.prediction_threshold,
                    predictions,
                    torch.full_like(predictions, -1),
                )
            else:
                predictions = torch.argmax(outputs, dim=1)

            # Convert back: (1, Z, Y, X) -> (X, Y, Z)
            prediction_np = predictions.squeeze(0).permute(2, 1, 0).cpu().numpy()

            return prediction_np

    def predict(self, voxel_grid: np.ndarray) -> np.ndarray:
        """Run sliding window inference on full voxel grid"""
        data_shape = voxel_grid.shape
        positions = self._get_window_positions(data_shape)
        # Initialize output
        output_prediction = np.zeros(data_shape, dtype=np.int32)

        # Process each window
        for pos in positions:
            window = self._extract_window(voxel_grid, pos)
            window_pred = self._inference_single_window(window)

            start_x, start_y, start_z = pos
            win_x, win_y, win_z = self.target_size

            # Calculate actual region to place prediction (X, Y only)
            end_x = min(start_x + win_x, data_shape[0])
            end_y = min(start_y + win_y, data_shape[1])

            # Place prediction in output (simple overwrite for now)
            pred_x = end_x - start_x
            pred_y = end_y - start_y

            # Use full Z dimension
            output_prediction[start_x:end_x, start_y:end_y, :] = window_pred[:pred_x, :pred_y, :]

        return output_prediction


def upsampling_prediction(
    prediction: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Upsample prediction back to original voxel coordinate system
    Z dimension is upsampled to match original_shape

    Args:
        prediction: Prediction array (already processed)
        original_shape: Original voxel shape
        y_crop_top: Y crop from top
        y_crop_bottom: Y crop from bottom
        z_crop_front: Z crop from front
        alpha: Weighting parameter for upsampling (higher values = more emphasis on front)

    Returns:
        Prediction in original coordinate system with upsampled z dimension
    """
    output_pred = np.zeros(original_shape, dtype=prediction.dtype)
    target_z_size = original_shape[2]

    # Upsample the prediction in z direction
    upsampled_prediction = upsample_histogram_direction_zero_padding(prediction, target_z_size)

    output_pred = upsampled_prediction

    return output_pred


def run_estimation(config_path: str) -> None:
    """
    Main estimation function that processes all voxel files from the first test_voxel_dir

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config_from_yaml(config_path)
    if not isinstance(config, TestConfig):
        raise ValueError(f"config is not TestConfig: {config}")

    # Set up environment
    set_seed(config.seed)
    pprint(config)

    # Device setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = get_model(config).to(device)
    log_info(f"Model: {model.__class__.__name__}")

    # Load checkpoint
    if not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")

    log_info(f"Loading checkpoint from: {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    log_info(f"Total parameters: {total_params:,}")

    # Get voxel files from the first test_voxel_dir using FullRegionVoxelDataset
    if not config.test_voxel_dirs:
        raise ValueError("test_voxel_dirs is empty in config")

    # Create dataset with only the first voxel directory
    dataset = SimpleFWLDataset(
        voxel_dirs=config.test_voxel_dirs,
        downsample_z=config.downsample_z,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
        voxel_pattern="*.b2",
    )

    log_info(f"Found {len(dataset)} voxel files")

    if len(dataset) == 0:
        log_info("No voxel files found. Exiting.")
        return

    # Create output directory
    if config.output_dir:
        output_dir = pathlib.Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    else:
        raise ValueError("output_dir must be specified in config")

    # Create sliding window inference
    sliding_window = SlidingWindowInference(
        model=model,
        device=device,
        target_size=config.target_size,  # [X, Y, Z]
        config=config,
    )

    # Process each sample from dataset
    for i in tqdm(range(len(dataset)), desc="Processing voxel files"):
        try:
            # Get sample from dataset (preprocessing already applied)
            sample = dataset[i]
            frame_id = sample["frame_id"]
            voxel_grid = sample["voxel_grid"]  # Already preprocessed
            original_shape = sample["original_shape"]
            cropped_shape = sample["cropped_shape"]

            log_info(f"Processing {frame_id}: shape {voxel_grid.shape}")

            # Run sliding window inference
            prediction = sliding_window.predict(voxel_grid)

            # Upsample prediction back to original coordinate system
            restored_prediction = upsampling_prediction(
                prediction=prediction,
                original_shape=original_shape,
            )

            voxel_file = str(sample["voxel_file"])

            voxel_path = pathlib.Path(voxel_file)
            parent_dir = voxel_path.parent
            grandparent_dir = parent_dir.parent
            grandgrandparent_dir = grandparent_dir.parent
            root_dir_name = grandgrandparent_dir.name
            sub_dir_name = parent_dir.name

            prediction_path = (
                output_dir / f"{root_dir_name}_{sub_dir_name}_{frame_id}_prediction_voxel.b2"
            )

            if not restored_prediction.flags["C_CONTIGUOUS"]:
                restored_prediction = np.ascontiguousarray(restored_prediction)

            save_blosc2(str(prediction_path), restored_prediction)

            log_info(f"Saved results for {frame_id}: {restored_prediction.shape}")

        except Exception as e:
            log_info(f"Error processing sample {i}: {e}")
            continue

    log_info(f"Estimation completed! Results saved to: {output_dir}")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run inference on voxel files using trained UNet3D model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to test configuration YAML file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    run_estimation(args.config)


if __name__ == "__main__":
    main()

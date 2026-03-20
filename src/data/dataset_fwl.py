import pathlib
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import LABEL_MAP
from src.utils import (
    downsample_histogram_direction,
    load_blosc2,
    random_crop_voxel_grid_with_coords,
)
from src.utils.log import log_info


class FWLDataset(Dataset):
    def __init__(
        self,
        voxel_dirs: Optional[List[str]] = None,
        annotation_dirs: Optional[List[str]] = None,
        voxel_pattern: str = "*_voxel.b2",
        annotation_pattern: str = "*_annotation_voxel.b2",
        target_size: Optional[list[int]] = None,
        downsample_z: Optional[int] = None,
        divide: int = 1,
        ignore_label: int = 0,
        y_crop_top: int = 0,
        y_crop_bottom: int = 0,
        z_crop_front: int = 0,
        z_crop_back: int = 0,
    ) -> None:
        """
        Initialize the FWL dataset.

        Args:
            voxel_dirs: List of directories containing voxel grid files
            annotation_dirs: List of directories containing annotation files
            voxel_pattern: File pattern for voxel files (default: "*_voxel.npy")
            annotation_pattern: File pattern for annotation files (default: "*_annotation_voxel.npy")
            target_size: Optional target size (x, y, z) for resizing voxel grids
            downsample_z: Optional target z size for histogram direction downsampling
            divide: Divide factor for limiting dataset size (for debugging)
            ignore_label: Label to ignore during training (default: 0 for noise)
            y_crop_top: Number of voxels to crop from the top of Y axis (default: 0)
            y_crop_bottom: Number of voxels to crop from the bottom of Y axis (default: 0)
            z_crop_front: Number of histogram bins to crop from the front of Z axis (default: 0)
        """
        # Handle directory arguments - support both single and multiple directories
        if voxel_dirs is not None and annotation_dirs is not None:
            self.voxel_dirs = [pathlib.Path(d) for d in voxel_dirs]
            self.annotation_dirs = [pathlib.Path(d) for d in annotation_dirs]
        else:
            raise ValueError("Must specify either (voxel_dirs, annotation_dirs)")

        if len(self.voxel_dirs) != len(self.annotation_dirs):
            raise ValueError(
                f"Number of voxel directories must match number of annotation directories: {len(self.voxel_dirs)} != {len(self.annotation_dirs)}"
            )

        self.voxel_pattern = voxel_pattern
        self.annotation_pattern = annotation_pattern
        self.target_size = tuple(target_size) if target_size else None
        if self.target_size and len(self.target_size) != 3:
            raise ValueError("target_size must be a tuple of three elements (x, y, z)")
        self.downsample_z = downsample_z
        self.divide = divide
        self.ignore_label = ignore_label
        self.y_crop_top = y_crop_top
        self.y_crop_bottom = y_crop_bottom
        self.z_crop_front = z_crop_front
        self.z_crop_back = z_crop_back

        # Label information
        self.label_names = LABEL_MAP
        self.num_classes = len(self.label_names)

        self.voxel_files = self._get_voxel_files()
        self.annotation_files = self._get_annotation_files()

        if self.divide > 1:
            # Calculate target size
            target_size = len(self.voxel_files) // self.divide

            # Create indices and randomly sample
            indices = list(range(len(self.voxel_files)))
            sampled_indices = random.sample(indices, target_size)
            sampled_indices.sort()  # Keep sorted order for consistency

            # Apply sampling to both file lists
            self.voxel_files = [self.voxel_files[i] for i in sampled_indices]
            self.annotation_files = [self.annotation_files[i] for i in sampled_indices]

        log_info(f"Found {len(self.voxel_files)} voxel files")
        log_info(f"Found {len(self.annotation_files)} annotation files")
        if self.target_size:
            log_info(f"Target size for resizing: {self.target_size}")
        if self.downsample_z:
            log_info(f"Target z size for histogram downsampling: {self.downsample_z}")
        if self.y_crop_top > 0 or self.y_crop_bottom > 0:
            log_info(f"Y axis cropping: top={self.y_crop_top}, bottom={self.y_crop_bottom}")
        if self.z_crop_front > 0:
            log_info(f"Z axis cropping: front={self.z_crop_front}")
        if self.z_crop_back > 0:
            log_info(f"Z axis cropping: back={self.z_crop_back}")

    def _get_voxel_files(self) -> list[str]:
        """Get all voxel files from all voxel directories."""
        all_files = []
        for voxel_dir in self.voxel_dirs:
            files = list(voxel_dir.glob(self.voxel_pattern))
            all_files.extend(files)
        return sorted(all_files)

    def _get_annotation_files(self) -> list[str]:
        """Get all annotation files from all annotation directories."""
        all_files = []
        for annotation_dir in self.annotation_dirs:
            files = list(annotation_dir.glob(self.annotation_pattern))
            all_files.extend(files)
        return sorted(all_files)

    def _load_voxel_grid(self, file_path: str) -> np.ndarray:
        """Load voxel grid from npy file."""
        return load_blosc2(file_path).copy()

    def _load_annotation_voxel(self, file_path: str) -> np.ndarray:
        """Load annotation voxel from npy file."""
        return load_blosc2(file_path).copy()

    def _apply_y_crop(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Apply Y axis cropping to voxel grid.

        Args:
            voxel_grid: Input voxel grid with shape (X, Y, Z)

        Returns:
            Cropped voxel grid
        """
        if self.y_crop_top == 0 and self.y_crop_bottom == 0:
            return voxel_grid

        y_size = voxel_grid.shape[1]

        # Calculate crop indices
        y_start = self.y_crop_bottom
        y_end = y_size - self.y_crop_top

        # Ensure valid cropping range
        if y_start >= y_end:
            raise ValueError(
                f"Y cropping parameters are too large: y_crop_bottom={self.y_crop_bottom}, y_crop_top={self.y_crop_top}, y_size={y_size}"
            )

        # Apply cropping along Y axis
        cropped_voxel = voxel_grid[:, y_start:y_end, :]

        return cropped_voxel

    def _apply_z_crop(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Apply Z axis (histogram direction) cropping to voxel grid.

        Args:
            voxel_grid: Input voxel grid with shape (X, Y, Z)

        Returns:
            Cropped voxel grid
        """
        if self.z_crop_front == 0 and self.z_crop_back == 0:
            return voxel_grid

        z_size = voxel_grid.shape[2]

        # Calculate crop indices (remove bins from front)
        z_start = self.z_crop_front
        z_end = z_size - self.z_crop_back

        # Ensure valid cropping range
        if z_start >= z_end:
            raise ValueError(
                f"Z cropping parameter is too large: z_crop_front={self.z_crop_front}, z_size={z_size}"
            )

        # Apply cropping along Z axis
        cropped_voxel = voxel_grid[:, :, z_start:z_end]

        return cropped_voxel

    def _determine_crop_coordinates(
        self, voxel_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """
        Determine crop coordinates for consistent cropping between voxel and annotation.

        Args:
            voxel_shape: Shape of the voxel grid (x, y, z)

        Returns:
            tuple[int, int, int]: Start coordinates (x, y, z) for cropping
        """
        if self.target_size is None:
            return (0, 0, 0)

        # If downsampling was applied, adjust target_size accordingly
        if self.downsample_z is not None:
            target_shape = (
                self.target_size[0],
                self.target_size[1],
                self.downsample_z,
            )  # (X, Y, hist)
        else:
            target_shape = self.target_size  # (X, Y, hist)

        # Use the original voxel shape for coordinate calculation
        original_shape = np.array(voxel_shape)
        target_shape = np.array(target_shape)

        # Calculate maximum possible start indices to ensure no overflow
        max_start_x = max(0, original_shape[0] - target_shape[0])
        max_start_y = max(0, original_shape[1] - target_shape[1])
        max_start_z = max(0, original_shape[2] - target_shape[2])

        # Random cropping
        start_x = np.random.randint(0, max_start_x + 1) if max_start_x > 0 else 0
        start_y = np.random.randint(0, max_start_y + 1) if max_start_y > 0 else 0
        start_z = np.random.randint(0, max_start_z + 1) if max_start_z > 0 else 0

        return (start_x, start_y, start_z)

    def _preprocess_voxel(
        self, voxel_grid: np.ndarray, start_coords: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Apply preprocessing to voxel grid.

        Args:
            voxel_grid: Input voxel grid (already downsampled if specified)
            start_coords: Start coordinates for cropping
        Returns:
            Preprocessed voxel grid
        """
        processed_voxel = voxel_grid

        # Apply target size resizing if specified
        if self.target_size is not None:
            # If downsampling was applied, adjust target_size accordingly
            if self.downsample_z is not None:
                target_shape = (
                    self.target_size[0],
                    self.target_size[1],
                    self.downsample_z,
                )  # (X, Y, hist)
            else:
                target_shape = self.target_size  # (X, Y, hist)

            # Use random cropping with consistent coordinates
            processed_voxel, _ = random_crop_voxel_grid_with_coords(
                processed_voxel, target_shape, start_coords
            )

        return processed_voxel

    def _preprocess_annotation(
        self, annotation_voxel: np.ndarray, start_coords: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Apply preprocessing to annotation voxel.

        Args:
            annotation_voxel: Input annotation voxel (already downsampled if specified)
            start_coords: Start coordinates for cropping
        Returns:
            Preprocessed annotation voxel
        """
        processed_annotation = annotation_voxel

        # Apply target size resizing if specified
        if self.target_size is not None:
            # If downsampling was applied, adjust target_size accordingly
            if self.downsample_z is not None:
                target_shape = (
                    self.target_size[0],
                    self.target_size[1],
                    self.downsample_z,
                )  # (X, Y, hist)
            else:
                target_shape = self.target_size  # (X, Y, hist)

            # Use random cropping with consistent coordinates (same as voxel)
            processed_annotation, _ = random_crop_voxel_grid_with_coords(
                processed_annotation, target_shape, start_coords
            )

        return processed_annotation

    def get_label_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Compute label statistics across the entire dataset.

        Returns:
            Dictionary with label statistics
        """
        total_counts = {label: 0 for label in range(self.num_classes)}
        total_voxels = 0

        log_info("Computing label statistics across dataset...")

        for i in range(len(self.annotation_files)):
            annotation_file = self.annotation_files[i]
            annotation_voxel = self._load_annotation_voxel(annotation_file)

            # Apply Y axis cropping first
            annotation_voxel = self._apply_y_crop(annotation_voxel)

            # Apply Z axis cropping
            annotation_voxel = self._apply_z_crop(annotation_voxel)

            # Apply downsampling if specified
            if self.downsample_z is not None:
                annotation_voxel = downsample_histogram_direction(
                    annotation_voxel, self.downsample_z
                )

            # Apply preprocessing
            start_coords = self._determine_crop_coordinates(annotation_voxel.shape)
            annotation_voxel = self._preprocess_annotation(annotation_voxel, start_coords)

            unique_labels, counts = np.unique(annotation_voxel, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label < self.num_classes:
                    total_counts[label] += count
                    total_voxels += count

        # Convert to statistics
        statistics = {}
        for label, count in total_counts.items():
            label_name = self.label_names.get(label, f"label_{label}")
            percentage = (count / total_voxels * 100) if total_voxels > 0 else 0.0
            statistics[label_name] = {"count": count, "percentage": round(percentage, 2)}

        return statistics

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.voxel_files)

    def __getitem__(self, index: int) -> Dict[str, Union[str, np.ndarray, torch.Tensor]]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Dictionary containing:
                - 'frame_id': File stem identifier
                - 'voxel_grid': Voxel grid data
                - 'annotation': Annotation voxel data
        """
        voxel_file = self.voxel_files[index]
        annotation_file = self.annotation_files[index]

        # Extract frame ID from file stem
        frame_id = pathlib.Path(voxel_file).stem.replace("_voxel", "")

        # Extract scene_id from path: get directory name containing 'scene'
        # Example: /path/to/ghost_dataset/scene001/data/hist001 -> scene001
        voxel_path = pathlib.Path(voxel_file)

        # Load data
        voxel_grid = self._load_voxel_grid(voxel_file)  # (X, Y, hist)
        annotation_voxel = self._load_annotation_voxel(annotation_file)  # (X, Y, hist)

        # Apply Y axis cropping first
        voxel_grid = self._apply_y_crop(voxel_grid)
        annotation_voxel = self._apply_y_crop(annotation_voxel)

        # Apply Z axis cropping
        voxel_grid = self._apply_z_crop(voxel_grid)
        annotation_voxel = self._apply_z_crop(annotation_voxel)

        # Apply downsampling if specified
        if self.downsample_z is not None:
            voxel_grid = downsample_histogram_direction(voxel_grid, self.downsample_z)
            annotation_voxel = downsample_histogram_direction(annotation_voxel, self.downsample_z)

        # Determine crop coordinates after downsampling
        start_coords = self._determine_crop_coordinates(voxel_grid.shape)

        # Apply cropping preprocessing
        voxel_grid = self._preprocess_voxel(voxel_grid, start_coords)
        annotation_voxel = self._preprocess_annotation(annotation_voxel, start_coords)

        scene_id = "unknown"
        hist_id = "unknown"
        for parent in voxel_path.parents:
            if "scene" in parent.name:
                scene_id = parent.name
            if "hist" in parent.name:
                hist_id = parent.name

        # Create sample dictionary
        sample = {
            "frame_id": frame_id,
            "scene_id": scene_id,
            "hist_id": hist_id,
            "voxel_grid": voxel_grid,
            "annotation": annotation_voxel,
        }

        return sample

    def get_sample_info(self, index: int) -> Dict[str, Union[str, Tuple[int, ...]]]:
        """
        Get information about a specific sample without loading the full data.

        Args:
            index: Index of the sample

        Returns:
            Dictionary with sample information
        """
        voxel_file = self.voxel_files[index]
        annotation_file = self.annotation_files[index]
        frame_id = pathlib.Path(voxel_file).stem.replace("_voxel", "")
        scene_id = "unknown"
        hist_id = "unknown"
        voxel_path = pathlib.Path(voxel_file)
        for parent in voxel_path.parents:
            if "scene" in parent.name:
                scene_id = parent.name
            if "hist" in parent.name:
                hist_id = parent.name
        return {
            "frame_id": frame_id,
            "scene_id": scene_id,
            "hist_id": hist_id,
            "voxel_file": voxel_file,
            "annotation_file": annotation_file,
        }


# Helper function for creating a custom collate function
def voxel_collate_fn(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
    """
    Custom collate function for batching voxel data.

    Converts voxel data from (X, Y, Z) format to standard 3D UNet format (N, C, D, H, W):
    - Z (histogram direction) -> D (depth)
    - X -> W (width)
    - Y -> H (height)
    - Add channel dimension C=1

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with tensors in (N, C, D, H, W) format
    """
    frame_ids = [sample["frame_id"] for sample in batch]
    scene_ids = [sample["scene_id"] for sample in batch]

    # Stack and convert (X, Y, Z) -> (N, X, Y, Z)
    voxel_grids = torch.stack([torch.from_numpy(sample["voxel_grid"]) for sample in batch]).float()
    annotations = torch.stack([torch.from_numpy(sample["annotation"]) for sample in batch]).long()

    # Permute from (N, X, Y, Z) to (N, Z, Y, X) then add channel: (N, 1, Z, X, Y)
    # This gives us (N, C, D, H, W) where Z=D, X=W, Y=H
    voxel_grids = voxel_grids.permute(0, 3, 2, 1).unsqueeze(1)  # (N, 1, D, H, W)
    annotations = annotations.permute(0, 3, 2, 1)  # (N, D, H, W)

    return {
        "frame_ids": frame_ids,
        "scene_ids": scene_ids,
        "voxel_grids": voxel_grids,  # (N, C=1, D, H, W)
        "annotations": annotations,  # (N, D, H, W)
    }

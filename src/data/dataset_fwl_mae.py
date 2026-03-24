import pathlib
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.data import FWLDataset
from src.utils import (
    create_voxel_mask,
    downsample_histogram_direction,
    load_blosc2,
    log_info,
    random_crop_voxel_grid_with_coords,
)


class FWLMAEPDataset(FWLDataset):
    def __init__(
        self,
        voxel_dirs: list[str],
        peak_dirs: list[str],
        voxel_pattern: str = "*_voxel.b2",
        peak_pattern: str = "*_peak.npy",
        target_size: Optional[list[int]] = None,
        downsample_z: Optional[int] = None,
        max_peaks: int = 4,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0,
        divide: int = 1,
        patch_size: tuple[int, int, int] = (4, 4, 128),
        y_crop_top: int = 0,
        y_crop_bottom: int = 0,
        z_crop_front: int = 0,
        z_crop_back: int = 0,
    ) -> None:
        # Initialize directories
        self.voxel_dirs = [pathlib.Path(d) for d in voxel_dirs]
        self.peak_dirs = [pathlib.Path(d) for d in peak_dirs]
        self.voxel_pattern = voxel_pattern
        self.peak_pattern = peak_pattern
        self.target_size = tuple(target_size) if target_size else None
        if self.target_size and len(self.target_size) != 3:
            raise ValueError("target_size must be a tuple of three elements (x, y, z)")
        self.downsample_z = downsample_z
        self.max_peaks = max_peaks
        self.divide = divide
        self.y_crop_top = y_crop_top
        self.y_crop_bottom = y_crop_bottom
        self.z_crop_front = z_crop_front
        self.z_crop_back = z_crop_back
        # Masked autoencoder parameters
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.patch_size = patch_size

        # Load voxel and peak file pairs
        self.voxel_files, self.peak_files = self._get_file_pairs()

        if self.divide > 1:
            target_size = len(self.voxel_files) // self.divide
            indices = list(range(len(self.voxel_files)))
            sampled_indices = random.sample(indices, target_size)
            sampled_indices.sort()  # Keep sorted order for consistency
            self.voxel_files = [self.voxel_files[i] for i in sampled_indices]
            self.peak_files = [self.peak_files[i] for i in sampled_indices]

    def _get_file_pairs(self) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
        """Get matching voxel and peak file pairs."""
        # Get all voxel files
        voxel_files: list[pathlib.Path] = []
        for root_dir in self.voxel_dirs:
            if root_dir.is_file():
                if root_dir.match(self.voxel_pattern):
                    voxel_files.append(root_dir)
            elif root_dir.is_dir():
                voxel_files.extend(root_dir.rglob(self.voxel_pattern))

        # Get all peak files
        peak_files: list[pathlib.Path] = []
        for root_dir in self.peak_dirs:
            if root_dir.is_file():
                if root_dir.match(self.peak_pattern):
                    peak_files.append(root_dir)
            elif root_dir.is_dir():
                peak_files.extend(root_dir.rglob(self.peak_pattern))

        # Create mapping from frame_id to files
        voxel_map = {}
        for vf in voxel_files:
            frame_id = vf.stem.replace("_voxel", "")
            voxel_map[frame_id] = vf

        peak_map = {}
        for pf in peak_files:
            frame_id = pf.stem.replace("_peak", "")
            peak_map[frame_id] = pf

        # Find matching pairs
        matched_voxel_files = []
        matched_peak_files = []

        for frame_id in voxel_map:
            if frame_id in peak_map:
                matched_voxel_files.append(voxel_map[frame_id])
                matched_peak_files.append(peak_map[frame_id])

        log_info(f"Found {len(voxel_files)} voxel files")
        log_info(f"Found {len(peak_files)} peak files")
        log_info(f"Matched {len(matched_voxel_files)} voxel-peak pairs")

        return sorted(matched_voxel_files), sorted(matched_peak_files)

    def _load_voxel_grid(self, file_path: Union[str, pathlib.Path]) -> np.ndarray:
        """Load voxel grid from b2 file."""
        return load_blosc2(file_path)

    def _load_peak_data(self, file_path: Union[str, pathlib.Path]) -> np.ndarray:
        """
        Load peak data from npy file efficiently.

        This method reads peak data and returns it as numpy array without
        unnecessary conversions for better performance.

        Args:
            file_path: Path to the peak data npy file

        Returns:
            Raw peak data as numpy array (object dtype)
        """
        file_path = pathlib.Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Peak file not found: {file_path}")

        # Load the peak data array directly without conversion
        try:
            peak_data = np.load(file_path, allow_pickle=True)
            return peak_data
        except Exception as e:
            raise RuntimeError(f"Failed to load peak data from {file_path}: {e}")

    def _convert_peaks_to_tensors_fast(
        self, peak_data: np.ndarray, voxel_shape: tuple
    ) -> Dict[str, np.ndarray]:
        """
        Convert peak data to tensor format efficiently using vectorized operations.

        Args:
            peak_data: Raw peak data array from npy file
            voxel_shape: Shape of the voxel grid (x, y, z)

        Returns:
            Dictionary with peak tensors
        """
        x_size, y_size, z_size = voxel_shape

        # Pre-allocate tensors: (K, X, Y) format (will be permuted later)
        peak_positions = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)
        peak_heights = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)
        peak_widths = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)

        # Process peak data efficiently
        for item in peak_data:
            try:
                x, y, peaks = item
                x, y = int(x), int(y)

                # Bounds check
                if not (0 <= x < x_size and 0 <= y < y_size):
                    continue

                # Process peaks for this voxel
                if peaks:
                    num_peaks = min(len(peaks), self.max_peaks)
                    for k in range(num_peaks):
                        peak = peaks[k]
                        if len(peak) >= 4:
                            pos, intensity, width = float(peak[0]), float(peak[1]), float(peak[2])
                            # Validate peak position
                            if 0 <= pos < z_size:
                                peak_positions[k, x, y] = pos
                                peak_heights[k, x, y] = intensity
                                peak_widths[k, x, y] = width
            except (ValueError, TypeError, IndexError):
                # Skip malformed entries silently for performance
                continue

        return {
            "peak_positions": peak_positions,
            "peak_heights": peak_heights,
            "peak_widths": peak_widths,
        }

    def _determine_crop_coordinates(
        self, voxel_shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """Same as VoxelReconstructionDataset."""
        if self.target_size is None:
            return (0, 0, 0)

        if self.downsample_z is not None:
            target_shape = (
                self.target_size[0],
                self.target_size[1],
                self.downsample_z,
            )
        else:
            target_shape = self.target_size

        original_shape = np.array(voxel_shape)
        target_shape = np.array(target_shape)

        max_start_x = max(0, original_shape[0] - target_shape[0])
        max_start_y = max(0, original_shape[1] - target_shape[1])
        max_start_z = max(0, original_shape[2] - target_shape[2])

        start_x = np.random.randint(0, max_start_x + 1) if max_start_x > 0 else 0
        start_y = np.random.randint(0, max_start_y + 1) if max_start_y > 0 else 0
        start_z = np.random.randint(0, max_start_z + 1) if max_start_z > 0 else 0

        return (start_x, start_y, start_z)

    def _preprocess_voxel(
        self, voxel_grid: np.ndarray, start_coords: tuple | None = None
    ) -> np.ndarray:
        """Same as VoxelReconstructionDataset."""
        processed_voxel = voxel_grid

        if self.target_size is not None:
            if self.downsample_z is not None:
                target_shape = (
                    self.target_size[0],
                    self.target_size[1],
                    self.downsample_z,
                )
            else:
                target_shape = self.target_size

            if start_coords is None:
                start_coords = self._determine_crop_coordinates(processed_voxel.shape)
            processed_voxel, _ = random_crop_voxel_grid_with_coords(
                processed_voxel, target_shape, start_coords
            )

        return processed_voxel

    def _crop_peak_tensors(
        self, peak_tensors: Dict[str, np.ndarray], start_coords: tuple, target_shape: tuple
    ) -> Dict[str, np.ndarray]:
        """Crop peak tensors to match the cropped voxel grid efficiently."""
        start_x, start_y, start_z = start_coords
        target_x, target_y, target_z = target_shape

        # Pre-calculate slice objects for better performance
        x_slice = slice(start_x, start_x + target_x)
        y_slice = slice(start_y, start_y + target_y)

        # Use dict comprehension for efficient cropping
        return {key: tensor[:, x_slice, y_slice] for key, tensor in peak_tensors.items()}

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.voxel_files)

    def __getitem__(self, index: int) -> Dict[str, Union[str, np.ndarray]]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Dictionary containing:
                - 'frame_id': File stem identifier
                - 'masked_voxel': Masked voxel grid (MAE input)
                - 'original_voxel': Original voxel grid (MAE target)
                - 'mask': Boolean mask indicating masked positions
                - 'peak_positions': Peak positions tensor (only at masked locations)
                - 'peak_heights': Peak heights tensor (only at masked locations)
                - 'peak_widths': Peak widths tensor (only at masked locations)
        """
        voxel_file = self.voxel_files[index]
        peak_file = self.peak_files[index]

        # Extract frame ID
        frame_id = voxel_file.stem.replace("_voxel", "")

        # Load voxel data
        voxel_grid = self._load_voxel_grid(voxel_file)

        voxel_grid = self._apply_y_crop(voxel_grid)
        voxel_grid = self._apply_z_crop(voxel_grid)

        if self.downsample_z is not None:
            voxel_grid = downsample_histogram_direction(voxel_grid, self.downsample_z)

        # Load peak data efficiently
        peak_data = self._load_peak_data(peak_file)

        # Convert peaks to tensors before any cropping using fast method
        peak_tensors = self._convert_peaks_to_tensors_fast(peak_data, voxel_grid.shape)

        # Store coordinates for cropping both voxel and peak data
        start_coords = self._determine_crop_coordinates(voxel_grid.shape)

        # Preprocess voxel data (cropping)
        voxel_grid = self._preprocess_voxel(voxel_grid, start_coords)

        # Crop peak tensors to match the voxel grid
        if self.target_size is not None:
            target_shape = self.target_size
            if self.downsample_z is not None:
                target_shape = (target_shape[0], target_shape[1], self.downsample_z)
            peak_tensors = self._crop_peak_tensors(peak_tensors, start_coords, target_shape)

        # Apply masking to create masked autoencoder data
        masked_voxel = None
        original_voxel = voxel_grid  # Default to original voxel_grid
        mask = None

        # Create sample dictionary
        sample = {
            "frame_id": frame_id,
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "original_voxel": original_voxel,  # Target for MAE
            "peak_positions": peak_tensors["peak_positions"],  # (K, X, Y) only at masked locations
            "peak_heights": peak_tensors["peak_heights"],  # (K, X, Y) only at masked locations
            "peak_widths": peak_tensors["peak_widths"],  # (K, X, Y) only at masked locations
        }

        return sample

    def get_sample_info(self, index: int) -> Dict[str, Union[str, Tuple[int, ...]]]:
        """
        Get information about a specific sample without loading heavy data.

        Args:
            index: Index of the sample

        Returns:
            Dictionary with basic sample information
        """
        voxel_file = self.voxel_files[index]
        peak_file = self.peak_files[index]
        frame_id = voxel_file.stem.replace("_voxel", "")

        # Return basic info only for performance
        return {
            "frame_id": frame_id,
            "voxel_file": str(voxel_file),
            "peak_file": str(peak_file),
            "max_peaks": self.max_peaks,
            "voxel_exists": voxel_file.exists(),
            "peak_exists": peak_file.exists(),
        }


# Helper function for creating a custom collate function for MAE + peak prediction
def fwl_mae_collate_fn(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
    """
    Custom collate function for batching masked autoencoder + peak prediction data.

    Converts voxel data from (X, Y, Z) format to standard 3D format (N, C, D, H, W):
    - Z (histogram direction) -> D (depth)
    - X -> W (width)
    - Y -> H (height)
    - Add channel dimension C=1

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with tensors in (N, C, D, H, W) format
    """
    data = {}

    frame_ids = [sample["frame_id"] for sample in batch]
    data["frame_ids"] = frame_ids

    # Stack and convert (X, Y, Z) -> (N, X, Y, Z)
    original_voxels = torch.stack(
        [torch.from_numpy(sample["original_voxel"]).float() for sample in batch]
    )
    original_voxels = original_voxels.permute(0, 3, 2, 1).unsqueeze(1)  # (N, 1, D, H, W)
    data["original_voxels"] = original_voxels

    # Stack peak prediction data
    peak_positions = torch.stack(
        [torch.from_numpy(sample["peak_positions"]).float() for sample in batch]
    )
    peak_positions = peak_positions.permute(0, 1, 3, 2)  # (N, K, H, W)
    data["peak_positions"] = peak_positions

    peak_heights = torch.stack(
        [torch.from_numpy(sample["peak_heights"]).float() for sample in batch]
    )
    peak_heights = peak_heights.permute(0, 1, 3, 2)  # (N, K, H, W)
    data["peak_heights"] = peak_heights

    peak_widths = torch.stack([torch.from_numpy(sample["peak_widths"]).float() for sample in batch])
    peak_widths = peak_widths.permute(0, 1, 3, 2)  # (N, K, H, W)
    data["peak_widths"] = peak_widths

    sample = batch[0]
    masks, num_patches = create_voxel_mask(
        original_voxels, split_xyz=sample["patch_size"], mask_ratio=sample["mask_ratio"]
    )
    data["masks"] = masks
    data["num_patches"] = num_patches

    return data

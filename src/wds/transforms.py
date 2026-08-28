"""Composable preprocessing transforms for the WebDataset (wds) loaders.

They reproduce, step by step, the preprocessing that FWLDataset / FWLMAEPDataset
apply inside __getitem__, so that a sample read from a WebDataset shard ends up
in exactly the same format as one read from the original directory layout:

    (X, Y, Z) ndarray  ->  Y crop  ->  Z crop  ->  Z downsample  ->  random crop

Everything here operates on numpy arrays in (X, Y, Z) order; the conversion to
(N, C, D, H, W) tensors is left to the existing collate functions.
"""

import io
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.utils import (
    downsample_histogram_direction,
    load_blosc2_bytes,
    random_crop_voxel_grid_with_coords,
)

Coords = Tuple[int, int, int]


class YCrop:
    """Crop the Y axis: keep voxel[:, bottom : Y - top, :] (same as FWLDataset._apply_y_crop)."""

    def __init__(self, top: int = 0, bottom: int = 0) -> None:
        self.top = top
        self.bottom = bottom

    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        if self.top == 0 and self.bottom == 0:
            return voxel
        y_size = voxel.shape[1]
        y_start = self.bottom
        y_end = y_size - self.top
        if y_start >= y_end:
            raise ValueError(
                f"Y cropping parameters are too large: y_crop_bottom={self.bottom}, "
                f"y_crop_top={self.top}, y_size={y_size}"
            )
        return voxel[:, y_start:y_end, :]


class ZCrop:
    """Crop the Z (histogram) axis: keep voxel[:, :, front : Z - back]."""

    def __init__(self, front: int = 0, back: int = 0) -> None:
        self.front = front
        self.back = back

    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        if self.front == 0 and self.back == 0:
            return voxel
        z_size = voxel.shape[2]
        z_start = self.front
        z_end = z_size - self.back
        if z_start >= z_end:
            raise ValueError(
                f"Z cropping parameter is too large: z_crop_front={self.front}, "
                f"z_crop_back={self.back}, z_size={z_size}"
            )
        return voxel[:, :, z_start:z_end]


class DownsampleZ:
    """Downsample the histogram axis to `target_z` bins (no-op when target_z is None)."""

    def __init__(self, target_z: Optional[int] = None) -> None:
        self.target_z = target_z

    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        if self.target_z is None:
            return voxel
        return downsample_histogram_direction(voxel, self.target_z)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            voxel = t(voxel)
        return voxel


def resolve_target_shape(
    target_size: Optional[Sequence[int]], downsample_z: Optional[int]
) -> Optional[Tuple[int, int, int]]:
    """Target (X, Y, Z) after downsampling — mirrors the logic in FWLDataset."""
    if target_size is None:
        return None
    if len(target_size) != 3:
        raise ValueError("target_size must be a tuple of three elements (x, y, z)")
    x, y, z = target_size
    if downsample_z is not None:
        z = downsample_z
    return (int(x), int(y), int(z))


class RandomCrop3D:
    """Random crop to a fixed (X, Y, Z) shape, with the crop origin shareable between arrays.

    `sample_coords()` draws the origin exactly like FWLDataset._determine_crop_coordinates
    (np.random.randint per axis, in x, y, z order) so that seeding np.random reproduces
    the legacy datasets' crops bit for bit.
    """

    def __init__(self, target_shape: Optional[Tuple[int, int, int]]) -> None:
        self.target_shape = target_shape

    def sample_coords(self, voxel_shape: Tuple[int, int, int]) -> Coords:
        if self.target_shape is None:
            return (0, 0, 0)
        original = np.array(voxel_shape)
        target = np.array(self.target_shape)
        max_start_x = max(0, original[0] - target[0])
        max_start_y = max(0, original[1] - target[1])
        max_start_z = max(0, original[2] - target[2])
        start_x = np.random.randint(0, max_start_x + 1) if max_start_x > 0 else 0
        start_y = np.random.randint(0, max_start_y + 1) if max_start_y > 0 else 0
        start_z = np.random.randint(0, max_start_z + 1) if max_start_z > 0 else 0
        return (int(start_x), int(start_y), int(start_z))

    def __call__(self, voxel: np.ndarray, start_coords: Coords) -> np.ndarray:
        if self.target_shape is None:
            return voxel
        cropped, _ = random_crop_voxel_grid_with_coords(voxel, self.target_shape, start_coords)
        return cropped


@dataclass
class VoxelPreprocess:
    """The full per-sample voxel pipeline shared by the ghost and mae loaders."""

    target_size: Optional[Sequence[int]] = None
    downsample_z: Optional[int] = None
    y_crop_top: int = 0
    y_crop_bottom: int = 0
    z_crop_front: int = 0
    z_crop_back: int = 0

    def __post_init__(self) -> None:
        self.pre_crop = Compose(
            [
                YCrop(self.y_crop_top, self.y_crop_bottom),
                ZCrop(self.z_crop_front, self.z_crop_back),
                DownsampleZ(self.downsample_z),
            ]
        )
        self.target_shape = resolve_target_shape(self.target_size, self.downsample_z)
        self.random_crop = RandomCrop3D(self.target_shape)

    def prepare(self, voxel: np.ndarray) -> np.ndarray:
        """Crop + downsample (everything before the random crop)."""
        return self.pre_crop(voxel)

    def crop(self, voxel: np.ndarray, start_coords: Coords) -> np.ndarray:
        return self.random_crop(voxel, start_coords)

    def sample_coords(self, voxel_shape: Tuple[int, int, int]) -> Coords:
        return self.random_crop.sample_coords(voxel_shape)


class PeaksToTensors:
    """Convert a raw peaks array (object ndarray of (x, y, [peak, ...])) to (K, X, Y) tensors.

    Same semantics as FWLMAEPDataset._convert_peaks_to_tensors_fast.
    """

    def __init__(self, max_peaks: int = 4) -> None:
        self.max_peaks = max_peaks

    def __call__(
        self, peak_data: np.ndarray, voxel_shape: Tuple[int, int, int]
    ) -> Dict[str, np.ndarray]:
        x_size, y_size, z_size = voxel_shape
        peak_positions = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)
        peak_heights = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)
        peak_widths = np.zeros((self.max_peaks, x_size, y_size), dtype=np.float32)

        for item in peak_data:
            try:
                x, y, peaks = item
                x, y = int(x), int(y)
                if not (0 <= x < x_size and 0 <= y < y_size):
                    continue
                if peaks:
                    num_peaks = min(len(peaks), self.max_peaks)
                    for k in range(num_peaks):
                        peak = peaks[k]
                        if len(peak) >= 4:
                            pos, intensity, width = float(peak[0]), float(peak[1]), float(peak[2])
                            if 0 <= pos < z_size:
                                peak_positions[k, x, y] = pos
                                peak_heights[k, x, y] = intensity
                                peak_widths[k, x, y] = width
            except (ValueError, TypeError, IndexError):
                continue

        return {
            "peak_positions": peak_positions,
            "peak_heights": peak_heights,
            "peak_widths": peak_widths,
        }


def crop_peak_tensors(
    peak_tensors: Dict[str, np.ndarray], start_coords: Coords, target_shape: Tuple[int, int, int]
) -> Dict[str, np.ndarray]:
    """Crop (K, X, Y) peak tensors to the region selected for the voxel grid."""
    start_x, start_y, _ = start_coords
    target_x, target_y, _ = target_shape
    x_slice = slice(start_x, start_x + target_x)
    y_slice = slice(start_y, start_y + target_y)
    return {key: tensor[:, x_slice, y_slice] for key, tensor in peak_tensors.items()}


def decode_b2(data: bytes) -> np.ndarray:
    """bytes of a .b2 tar member -> ndarray (X, Y, Z)."""
    return load_blosc2_bytes(data)


def decode_peaks(data: bytes) -> np.ndarray:
    """bytes of a _peak.npy tar member -> object ndarray."""
    return np.load(io.BytesIO(data), allow_pickle=True)

from typing import Tuple

import numpy as np
import torch


def crop_voxel_grid(
    voxel_grid: np.ndarray,
    start_indices: tuple[int, int, int] = (0, 0, 0),
    end_indices: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Crop the voxel grid by specifying start and end indices for each dimension.

    Args:
        voxel_grid (np.ndarray): The original voxel grid.
        start_indices (tuple[int, int, int]): Starting indices for each dimension (x, y, z).
        end_indices (tuple[int, int, int] | None): Ending indices for each dimension (x, y, z).
                                                   If None, uses the full size from start_indices.

    Returns:
        np.ndarray: The cropped voxel grid.
    """
    if end_indices is None:
        end_indices = voxel_grid.shape

    # Validate indices
    start_x, start_y, start_z = start_indices
    end_x, end_y, end_z = end_indices

    # Ensure indices are within bounds
    start_x = max(0, min(start_x, voxel_grid.shape[0]))
    start_y = max(0, min(start_y, voxel_grid.shape[1]))
    start_z = max(0, min(start_z, voxel_grid.shape[2]))

    end_x = max(start_x, min(end_x, voxel_grid.shape[0]))
    end_y = max(start_y, min(end_y, voxel_grid.shape[1]))
    end_z = max(start_z, min(end_z, voxel_grid.shape[2]))

    return voxel_grid[start_x:end_x, start_y:end_y, start_z:end_z]


def downsample_histogram_direction(voxel_grid: np.ndarray, target_z_size: int) -> np.ndarray:
    """
    Downsample the voxel grid in the histogram (z) direction by sampling at regular intervals.

    Args:
        voxel_grid (np.ndarray): The original voxel grid with shape (x, y, z).
        target_z_size (int): The target size for z dimension.

    Returns:
        np.ndarray: The downsampled voxel grid with shape (x, y, target_z_size).
    """
    x_size, y_size, z_size = voxel_grid.shape

    # Check if target size is larger than original size
    if target_z_size > z_size:
        raise ValueError(
            f"Target z size {target_z_size} cannot be larger than original z size {z_size}"
        )

    # Calculate sampling indices to get target_z_size samples from z_size
    # Use linspace to get evenly distributed indices
    z_indices = np.linspace(0, z_size - 1, target_z_size, dtype=int)

    # Sample the voxel grid at the calculated indices
    downsampled = voxel_grid[:, :, z_indices]

    return downsampled.astype(voxel_grid.dtype)


def upsample_histogram_direction(voxel_grid: np.ndarray, target_z_size: int) -> np.ndarray:
    """
    Upsample the voxel grid in the histogram (z) direction by interpolation.

    Args:
        voxel_grid (np.ndarray): The original voxel grid with shape (x, y, z).
        target_z_size (int): The target size for z dimension.
        order (int): The order of interpolation (0: nearest, 1: linear, 3: cubic).

    Returns:
        np.ndarray: The upsampled voxel grid with shape (x, y, target_z_size).
    """
    x_size, y_size, z_size = voxel_grid.shape

    # Check if target size is smaller than or equal to original size
    if target_z_size <= z_size:
        return voxel_grid.copy()

    # Create interpolation positions
    original_z_positions = np.arange(z_size)
    target_z_positions = np.linspace(0, z_size - 1, target_z_size)

    # Initialize upsampled grid
    upsampled_grid = np.zeros((x_size, y_size, target_z_size), dtype=voxel_grid.dtype)

    # Interpolate along z dimension for each (x, y) position
    for i in range(x_size):
        for j in range(y_size):
            upsampled_grid[i, j, :] = np.interp(
                target_z_positions, original_z_positions, voxel_grid[i, j, :]
            )

    return upsampled_grid


def upsample_histogram_direction_zero_padding(
    voxel_grid: np.ndarray, target_z_size: int
) -> np.ndarray:
    """
    Upsample the voxel grid in the histogram (z) direction by evenly placing original data
    and zero-padding the gaps between them.

    Original data points are evenly distributed across the target z dimension,
    and the positions between them are filled with zeros.

    Args:
        voxel_grid (np.ndarray): The original voxel grid with shape (x, y, z).
        target_z_size (int): The target size for z dimension.

    Returns:
        np.ndarray: The upsampled voxel grid with shape (x, y, target_z_size).
    """
    x_size, y_size, z_size = voxel_grid.shape

    # Check if target size is smaller than or equal to original size
    if target_z_size <= z_size:
        return voxel_grid.copy()

    # Initialize upsampled grid with zeros
    upsampled_grid = np.zeros((x_size, y_size, target_z_size), dtype=voxel_grid.dtype)

    # Calculate evenly spaced positions for original data
    # Place original z_size points evenly across target_z_size positions
    z_positions = np.linspace(0, target_z_size - 1, z_size, dtype=int)

    # Place original data at calculated positions
    upsampled_grid[:, :, z_positions] = voxel_grid

    return upsampled_grid


def random_crop_voxel_grid(voxel_grid: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """
    Randomly crop the voxel grid to the specified target size.

    Args:
        voxel_grid (np.ndarray): The original voxel grid.
        target_size (tuple[int, int, int]): The target size for each dimension (x, y, z).

    Returns:
        np.ndarray: The randomly cropped voxel grid.

    Raises:
        ValueError: If target size is larger than original size.
    """

    original_shape = np.array(voxel_grid.shape)
    target_shape = np.array(target_size)

    # Calculate maximum possible start indices to ensure no overflow
    max_start_x = max(0, original_shape[0] - target_shape[0])
    max_start_y = max(0, original_shape[1] - target_shape[1])
    max_start_z = max(0, original_shape[2] - target_shape[2])

    # Randomly select start indices within safe bounds
    start_x = np.random.randint(0, max_start_x + 1) if max_start_x > 0 else 0
    start_y = np.random.randint(0, max_start_y + 1) if max_start_y > 0 else 0
    start_z = np.random.randint(0, max_start_z + 1) if max_start_z > 0 else 0

    # Calculate end indices ensuring they don't exceed original bounds
    end_x = min(start_x + target_shape[0], original_shape[0])
    end_y = min(start_y + target_shape[1], original_shape[1])
    end_z = min(start_z + target_shape[2], original_shape[2])

    return crop_voxel_grid(voxel_grid, (start_x, start_y, start_z), (end_x, end_y, end_z))


def random_crop_voxel_grid_with_coords(
    voxel_grid: np.ndarray,
    target_size: tuple[int, int, int],
    start_coords: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Randomly crop the voxel grid to the specified target size with optional coordinate specification.

    Args:
        voxel_grid (np.ndarray): The original voxel grid.
        target_size (tuple[int, int, int]): The target size for each dimension (x, y, z).
        start_coords (tuple[int, int, int] | None): Starting coordinates for cropping.
                                                   If None, randomly generated.

    Returns:
        tuple[np.ndarray, tuple[int, int, int]]: The cropped voxel grid and the start coordinates used.

    Raises:
        ValueError: If target size is larger than original size.
    """
    original_shape = np.array(voxel_grid.shape)
    target_shape = np.array(target_size)

    if start_coords is None:
        # Calculate maximum possible start indices to ensure no overflow
        max_start_x = max(0, original_shape[0] - target_shape[0])
        max_start_y = max(0, original_shape[1] - target_shape[1])
        max_start_z = max(0, original_shape[2] - target_shape[2])
        # Randomly select start indices within safe bounds
        start_x = np.random.randint(0, max_start_x + 1) if max_start_x > 0 else 0
        start_y = np.random.randint(0, max_start_y + 1) if max_start_y > 0 else 0
        start_z = np.random.randint(0, max_start_z + 1) if max_start_z > 0 else 0
    else:
        # Use provided coordinates (already validated beforehand)
        start_x, start_y, start_z = start_coords

    # Calculate end indices ensuring they don't exceed original bounds
    end_x = min(start_x + target_shape[0], original_shape[0])
    end_y = min(start_y + target_shape[1], original_shape[1])
    end_z = min(start_z + target_shape[2], original_shape[2])

    cropped_voxel = crop_voxel_grid(voxel_grid, (start_x, start_y, start_z), (end_x, end_y, end_z))
    return cropped_voxel, (start_x, start_y, start_z)


def create_voxel_mask(
    x: torch.Tensor, split_xyz: Tuple[int, int, int] = (8, 8, 128), mask_ratio: float = 0.7
) -> Tuple[torch.Tensor, int]:
    """
    Create a random mask for 3D voxel data by dividing it into (x, y, z) patches.

    Args:
        x (torch.Tensor): Input tensor with shape (B, C, D, H, W)
        split_xyz (tuple): Number of divisions for (x, y, z) axes
        mask_ratio (float): Masking ratio (e.g., 0.7 for 70% masking)

    Returns:
        mask (torch.BoolTensor): Boolean mask with shape (B, num_patches)
        num_patches (int): Total number of patches after division
    """
    B, C, D, H, W = x.shape
    sx, sy, sz = split_xyz

    # Division size for each axis
    x_step = W // sx
    y_step = H // sy
    z_step = D // sz

    # Total number of patches
    num_patches = x_step * y_step * z_step

    # Initialize output mask
    mask = torch.zeros(B, num_patches, dtype=torch.bool)

    # Generate mask for each batch
    num_masked = int(num_patches * mask_ratio)
    for b in range(B):
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[b, masked_indices] = True

    return mask, num_patches

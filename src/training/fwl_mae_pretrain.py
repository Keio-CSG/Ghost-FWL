import argparse
import pathlib
from datetime import datetime
from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TestConfig, TrainingConfig, load_config_from_yaml
from src.data import FWLMAEPDataset, fwl_mae_collate_fn
from src.utils import (
    create_optimizer,
    create_scheduler,
    get_loss_fn,
    get_model,
    log_info,
    set_seed,
    set_wandb,
)


def patchify_voxel(x: torch.Tensor, patch_size: tuple) -> torch.Tensor:
    """Convert voxel to patches

    Args:
        x: Input voxel [B, C, D, H, W]
        patch_size: (patch_d, patch_h, patch_w)

    Returns:
        Patches [B, N, patch_volume]
    """
    B, C, D, H, W = x.shape
    patch_d, patch_h, patch_w = patch_size

    # Calculate number of patches
    num_patches_d = D // patch_d
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w

    # Reshape to patches
    x = x.reshape(B, C, num_patches_d, patch_d, num_patches_h, patch_h, num_patches_w, patch_w)
    x = x.permute(
        0, 2, 4, 6, 1, 3, 5, 7
    )  # [B, num_patches_d, num_patches_h, num_patches_w, C, patch_d, patch_h, patch_w]
    x = x.reshape(B, num_patches_d * num_patches_h * num_patches_w, C * patch_d * patch_h * patch_w)

    return x


def validation_epoch(
    config: TrainingConfig,
    model: torch.nn.Module,
    device: torch.device,
    valid_loader: DataLoader,
    current_epoch: int,
    loss_fn: nn.Module,
) -> None:
    """Validation epoch for FWLMAE Pretrain Model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(valid_loader)

    patch_size = model.patch_size

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(
                valid_loader,
                total=num_batches,
                desc=f"Validation Epoch {current_epoch + 1}/{config.epochs}",
            )
        ):
            original_voxels = batch["original_voxels"]  # (B, C=1, D, H, W) - MAE Target
            masks = batch["masks"]  # (B, N)
            peak_positions = batch[
                "peak_positions"
            ]  # (B, K, H, W) - Peak positions (masked locations)
            peak_heights = batch["peak_heights"]  # (B, K, H, W) - Peak heights (masked locations)
            peak_widths = batch["peak_widths"]  # (B, K, H, W) - Peak widths (masked locations)
            B = original_voxels.shape[0]

            original_voxels = original_voxels.to(device)
            masks = masks.to(device)
            peak_positions = peak_positions.to(device)
            peak_heights = peak_heights.to(device)
            peak_widths = peak_widths.to(device)

            model_output = model(original_voxels, masks)

            # Convert original voxels to patches for target calculation (original VoxelMAE method)
            patch_size_tuple = tuple(patch_size) if isinstance(patch_size, list) else patch_size
            target_patches = patchify_voxel(
                original_voxels, patch_size_tuple
            )  # [B, N, patch_volume]
            target_masked = target_patches[masks]  # [B*N_mask, patch_volume]
            target_masked = target_masked.reshape(
                B, -1, target_patches.shape[-1]
            )  # [B, N_mask, patch_volume]

            # Calculate peak prediction loss
            total_loss_batch, loss_components = loss_fn(
                model_output,
                {
                    "peak_positions": peak_positions,
                    "peak_heights": peak_heights,
                    "peak_widths": peak_widths,
                },
                masks,
                target_patches,
            )

            total_loss += total_loss_batch.item()

            if config.is_log and batch_idx % config.log_interval == 0:
                wandb.log(
                    {
                        "validation_total_loss": total_loss_batch.item(),
                    }
                )
                for comp_name, comp_value in loss_components.items():
                    if isinstance(comp_value, torch.Tensor):
                        wandb.log({f"validation_{comp_name}": comp_value.item()})

    average_loss = total_loss / num_batches

    log_info(f"Validation Loss: {average_loss:.6f}")

    if config.is_log:
        wandb.log({"validation_loss_per_epoch": average_loss})


def train_epoch(
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    train_loader: DataLoader,
    device: torch.device,
    current_epoch: int,
    loss_fn: nn.Module,
) -> None:
    """Training epoch for FWLMAE Pretrain Model"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    patch_size = model.patch_size

    for batch_idx, batch in enumerate(
        tqdm(
            train_loader,
            total=num_batches,
            desc=f"Epoch {current_epoch + 1}/{config.epochs}",
        )
    ):
        original_voxels = batch["original_voxels"]  # (B, C=1, D, H, W) - MAE Target
        masks = batch["masks"]  # (B, N)
        peak_positions = batch["peak_positions"]  # (B, K, H, W) - Peak positions (masked locations)
        peak_heights = batch["peak_heights"]  # (B, K, H, W) - Peak heights (masked locations)
        peak_widths = batch["peak_widths"]  # (B, K, H, W) - Peak widths (masked locations)
        B = original_voxels.shape[0]

        original_voxels = original_voxels.to(device)
        masks = masks.to(device)
        peak_positions = peak_positions.to(device)
        peak_heights = peak_heights.to(device)
        peak_widths = peak_widths.to(device)

        model_output = model(original_voxels, masks)

        # Convert original voxels to patches for target calculation (original VoxelMAE method)
        patch_size_tuple = tuple(patch_size) if isinstance(patch_size, list) else patch_size
        target_patches = patchify_voxel(original_voxels, patch_size_tuple)  # [B, N, patch_volume]
        target_masked = target_patches[masks]  # [B*N_mask, patch_volume]
        target_masked = target_masked.reshape(
            B, -1, target_patches.shape[-1]
        )  # [B, N_mask, patch_volume]

        # Calculate peak prediction loss
        total_loss_batch, loss_components = loss_fn(
            model_output,
            {
                "peak_positions": peak_positions,
                "peak_heights": peak_heights,
                "peak_widths": peak_widths,
            },
            masks,
            target_patches,
        )

        # Combine losses
        total_loss += total_loss_batch.item()

        # Log individual components for monitoring
        if config.is_log and batch_idx % config.log_interval == 0:
            wandb.log(
                {
                    "train_total_loss": total_loss_batch.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            for comp_name, comp_value in loss_components.items():
                if isinstance(comp_value, torch.Tensor):
                    wandb.log({f"train_{comp_name}": comp_value.item()})

        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    average_loss = total_loss / num_batches

    log_info(f"Epoch {current_epoch + 1}/{config.epochs} average loss: {average_loss:.6f}")

    if config.is_log:
        wandb.log({"train_loss_per_epoch": average_loss})


def save_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
) -> None:
    """Save model checkpoint"""
    save_dir = pathlib.Path(config.save_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voxel_mae_epoch_{epoch:03d}_{timestamp}.pth"
    day_dir_str = datetime.now().strftime("%m%d")
    save_dir = save_dir / day_dir_str
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename

    torch.save(checkpoint, filepath)
    log_info(f"Model saved: {filepath}")


def train_fwl_mae_pretrain(config_path: str) -> None:
    """Main training function for FWL-MAE Pretrain Model"""
    # Load configuration
    config: TrainingConfig | TestConfig = load_config_from_yaml(config_path)
    if not isinstance(config, TrainingConfig):
        raise ValueError(f"config is not TrainingConfig: {config}")

    # Set up training environment
    set_seed(config.seed)
    if config.is_log:
        set_wandb(config)
    pprint(config)

    # Device setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    # Model initialization
    model = get_model(config).to(device)
    log_info(f"Model: {model.__class__.__name__}")

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    log_info(f"Total parameters: {total_params:,}")

    # Loss function - Multi-task loss for peak prediction
    loss_fn = get_loss_fn(config)
    log_info(f"Using {config.loss_fn} loss for peak prediction")

    # Optimizer
    optimizer = create_optimizer(config, model.parameters())
    log_info(f"Optimizer: {config.optimizer}")

    # Scheduler
    if config.scheduler:
        scheduler = create_scheduler(optimizer, config)
        log_info(f"Scheduler: {config.scheduler}")
    else:
        scheduler = None

    # Dataset and DataLoader setup
    # Get dataset-specific parameters
    max_peaks = getattr(config, "max_peaks", 4)

    # Expect both voxel and peak directories for peak prediction training
    if not hasattr(config, "train_voxel_dirs") or not config.train_voxel_dirs:
        raise ValueError("train_voxel_dirs must be specified for peak prediction training")

    if not hasattr(config, "train_peak_dirs") or not config.train_peak_dirs:
        raise ValueError("train_peak_dirs must be specified for peak prediction training")

    train_dataset = FWLMAEPDataset(
        voxel_dirs=config.train_voxel_dirs,
        peak_dirs=config.train_peak_dirs,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        max_peaks=max_peaks,
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        divide=config.divide,
        patch_size=tuple(config.patch_size[::-1]),  # type: ignore[arg-type]
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=fwl_mae_collate_fn,
        pin_memory=True,
        persistent_workers=False,  # Disabled to prevent memory leaks
    )

    log_info(f"Training dataset size: {len(train_dataset)}")
    log_info(f"Training batches per epoch: {len(train_loader)}")
    log_info(f"Max peaks per voxel: {max_peaks}")

    # Validation dataset
    if not hasattr(config, "valid_voxel_dirs") or not config.valid_voxel_dirs:
        raise ValueError("valid_voxel_dirs must be specified for peak prediction validation")

    if not hasattr(config, "valid_peak_dirs") or not config.valid_peak_dirs:
        raise ValueError("valid_peak_dirs must be specified for peak prediction validation")

    valid_dataset = FWLMAEPDataset(
        voxel_dirs=config.valid_voxel_dirs,
        peak_dirs=config.valid_peak_dirs,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        max_peaks=max_peaks,
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        divide=config.divide,
        patch_size=tuple(config.patch_size[::-1]),  # type: ignore[arg-type]
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=fwl_mae_collate_fn,
        pin_memory=True,
        persistent_workers=False,  # Disabled to prevent memory leaks
    )

    log_info(f"Validation dataset size: {len(valid_dataset)}")
    log_info(f"Validation batches per epoch: {len(valid_loader)}")

    # Training loop
    for epoch in range(config.epochs):
        log_info(f"Starting epoch {epoch + 1}/{config.epochs}")

        # Training
        train_epoch(config, model, optimizer, scheduler, train_loader, device, epoch, loss_fn)

        # Validation
        validation_epoch(config, model, device, valid_loader, epoch, loss_fn)

        # Save model
        if (epoch + 1) % config.save_model_interval == 0:
            save_model(config, model, optimizer, epoch + 1, 0.0)  # Loss not tracked here

    # Save final model
    save_model(config, model, optimizer, config.epochs, 0.0)
    log_info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FWL-MAE Pretrain Model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to training configuration file"
    )
    args = parser.parse_args()

    # Ensure required imports
    import pathlib

    train_fwl_mae_pretrain(args.config)

import argparse
import os
from datetime import datetime
from pprint import pprint
from typing import Any

import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import TrainingConfig, load_config_from_yaml
from src.config.config import TestConfig
from src.data import FWLDataset, voxel_collate_fn
from src.utils import (
    create_optimizer,
    create_scheduler,
    get_loss_fn,
    get_model,
    set_seed,
    set_wandb,
)
from src.utils.log import log_info


def load_pretrained_ghost_fwl_pretrain(
    model: nn.Module, pretrained_path: str, device: torch.device, freeze_encoder: bool = False
) -> nn.Module:
    """Load  FWLMAEPretrain weights into FWLMAE model

    Args:
        model: FWLMAE finetune model
        pretrained_path: Path to pretrained FWLMAEPretrain checkpoint
        device: Device to load model on
        freeze_encoder: Whether to freeze encoder parameters

    Returns:
        Model with loaded pretrained weights
    """
    log_info(f"Loading pretrained FWLMAEPretrain from: {pretrained_path}")

    # Load pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)

    # Extract model state dict (handle different checkpoint formats)
    if "model_state_dict" in checkpoint:
        pretrained_state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        pretrained_state_dict = checkpoint["model"]
    else:
        pretrained_state_dict = checkpoint

    # Normalize keys (e.g., remove DistributedDataParallel 'module.' prefix)
    normalized_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        normalized_state_dict[k] = v

    # Get current model state dict
    model_state_dict = model.state_dict()

    # Define key mapping from pretrained FWLMAEPretrain to FWLMAE
    # Pretrained VoxelMAE structure: encoder.patch_embed, encoder.blocks, etc.
    # FWLMAE structure: patch_embed, blocks, etc. (direct)
    loaded_keys = []
    skipped_keys = []

    for key, value in normalized_state_dict.items():
        finetune_key = None

        # Map encoder weights
        # Case 1: Plain FWLMAEPretrain checkpoint -> keys like 'encoder.*'
        if key.startswith("encoder."):
            # Remove "encoder." prefix for finetune model
            finetune_key = key[8:]  # Remove "encoder."
        # Case 2: FWLMAEPretrain checkpoint -> keys like 'mae.encoder.*'
        elif key.startswith("mae.encoder."):
            finetune_key = key[len("mae.encoder.") :]

        # Handle positional embedding (might be registered as buffer)
        elif key == "pos_embed":
            finetune_key = "pos_embed"

        # Skip decoder and other components not needed for FWL prediction
        elif key.startswith(
            (
                "decoder.",
                "encoder_to_decoder.",
                "mask_token",
                # Skip non-encoder parts from FWLMAEPretrain checkpoints
                "mae.decoder.",
                "mae.encoder_to_decoder.",
                "mae.mask_token",
                "peak_position_head.",
                "peak_width_head.",
                "peak_height_head.",
            )
        ):
            skipped_keys.append(key)
            continue
        else:
            # Other keys that might be directly compatible
            if key in model_state_dict:
                finetune_key = key

        # Load compatible weights
        if finetune_key and finetune_key in model_state_dict:
            if model_state_dict[finetune_key].shape == value.shape:
                model_state_dict[finetune_key] = value.clone()
                loaded_keys.append(finetune_key)
            else:
                log_info(
                    f"Shape mismatch for {finetune_key}: "
                    f"model {model_state_dict[finetune_key].shape} vs "
                    f"pretrained {value.shape}"
                )
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)

    # Load the updated state dict
    model.load_state_dict(model_state_dict)

    # Freeze encoder parameters if requested
    if freeze_encoder:
        log_info("Freezing encoder parameters")
        frozen_params = 0
        trainable_params = 0

        # Define encoder components that should be frozen
        encoder_components = [
            "patch_embed",  # Patch embedding layer
            "pos_embed",  # Positional embedding
            "blocks",  # Transformer blocks
            "norm",  # Layer normalization (if not part of head)
        ]

        for name, param in model.named_parameters():
            # Check if parameter belongs to encoder components
            is_encoder_param = any(name.startswith(component) for component in encoder_components)

            # Freeze encoder parameters, keep decoder/head trainable
            if is_encoder_param and not name.startswith("head") and not name.startswith("decoder"):
                param.requires_grad = False
                frozen_params += 1
            else:
                param.requires_grad = True
                trainable_params += 1

        log_info(f"Frozen {frozen_params} encoder parameters")
        log_info(f"Keeping {trainable_params} decoder/head parameters trainable")
    else:
        log_info("All parameters remain trainable")

    log_info(f"Loaded {len(loaded_keys)} pretrained weights")
    log_info(f"Skipped {len(skipped_keys)} weights (decoder/classification head/incompatible)")

    if loaded_keys:
        log_info(
            "Loaded keys: " + ", ".join(loaded_keys[:10]) + ("..." if len(loaded_keys) > 10 else "")
        )

    if skipped_keys:
        log_info(
            "Skipped keys: "
            + ", ".join(skipped_keys[:5])
            + ("..." if len(skipped_keys) > 5 else "")
        )

    return model


def validation_epoch(
    config: TrainingConfig,
    model: torch.nn.Module,
    device: torch.device,
    valid_loader: DataLoader,
    current_epoch: int,
    loss_fn: nn.Module,
) -> None:
    """Validation epoch for FWLMAE Finetune Model"""
    model.eval()
    total_loss = 0.0
    num_batches = len(valid_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(
                valid_loader,
                total=num_batches,
                desc=f"Validation Epoch {current_epoch + 1}/{config.epochs}",
            )
        ):
            # Extract voxel data and annotations (already in 3D UNet format)
            voxel_grids = batch["voxel_grids"]  # (B, C=1, D, H, W)
            annotations = batch["annotations"]  # (B, D, H, W)

            # Move to device (channel dimension already added in collate_fn)
            voxel_grids = voxel_grids.float().to(device)  # (B, C=1, D, H, W)
            annotations = annotations.long().to(device)  # (B, D, H, W)

            # Forward pass
            outputs = model(voxel_grids)  # (B, num_classes, D, H, W)

            # Calculate loss
            loss = loss_fn(outputs, annotations)
            total_loss += loss.item()

            if config.is_log and batch_idx % config.log_interval == 0:
                wandb.log({"validation_loss": loss.item()})

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
    """Training epoch for FWLMAE Finetune Model"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(
        tqdm(
            train_loader,
            total=num_batches,
            desc=f"Epoch {current_epoch + 1}/{config.epochs}",
        )
    ):
        # Extract voxel data and annotations (already in 3D UNet format)
        voxel_grids = batch["voxel_grids"]  # (B, C=1, D, H, W)
        annotations = batch["annotations"]  # (B, D, H, W)

        # Move to device (channel dimension already added in collate_fn)
        voxel_grids = voxel_grids.float().to(device)  # (B, C=1, D, H, W)
        annotations = annotations.long().to(device)  # (B, D, H, W)

        # Forward pass
        outputs = model(voxel_grids)  # (B, num_classes, D, H, W)

        # Calculate loss
        loss = loss_fn(outputs, annotations)
        total_loss += loss.item()

        if config.is_log and batch_idx % config.log_interval == 0:
            wandb.log({"loss": loss.item()})

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    # Log metrics
    if config.is_log:
        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

    average_loss = total_loss / num_batches
    log_info(f"Epoch {current_epoch + 1}/{config.epochs} average loss: {average_loss:.6f}")

    if config.is_log:
        wandb.log({"loss_per_epoch": average_loss})

    # Save model checkpoint
    if (current_epoch + 1) % config.save_model_interval == 0:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        day_str = datetime.now().strftime("%m%d")
        save_dir = os.path.join(config.save_model_dir, day_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(
            save_dir,
            f"{config.model_name}_finetune_epoch_{current_epoch + 1}_{time_str}_{average_loss:.5f}.pth",
        )
        torch.save(
            model.state_dict(),
            save_path,
        )
        log_info(f"Model saved at epoch {current_epoch + 1} to {save_path}")


def train_fwl_mae_finetune(config_path: str) -> None:
    """Main training function for FWL-MAE Finetune Model"""
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
    model = get_model(config)
    # Print model parameter information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model parameters before loading pretrained weights:")
    log_info(f"  Total parameters: {total_params:,}")
    log_info(f"  Trainable parameters: {trainable_params:,}")
    model = model.to(device)
    log_info(f"Model: {model.__class__.__name__}")

    # Load pretrained weights if specified in config
    if hasattr(config, "pretrained_model_path") and config.pretrained_model_path:
        if os.path.exists(config.pretrained_model_path):
            model = load_pretrained_ghost_fwl_pretrain(
                model, config.pretrained_model_path, device, config.freeze_encoder
            )
        else:
            log_info(
                f"Warning: Pretrained path {config.pretrained_model_path} does not exist. Training from scratch."
            )

    # Load checkpoint if specified in config
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        log_info(f"Loading checkpoint from: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        log_info(f"Checkpoint loaded from: {config.checkpoint_path}")

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Total parameters: {total_params:,}")
    log_info(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    loss_fn = get_loss_fn(config)

    # Optimizer
    optimizer = create_optimizer(config, model.parameters())

    # Scheduler
    if config.scheduler is not None:
        scheduler = create_scheduler(optimizer, config)
    else:
        scheduler = None

    # Dataset and DataLoader setup
    train_dataset = FWLDataset(
        voxel_dirs=config.train_voxel_dirs,
        annotation_dirs=config.train_annotation_dirs,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        divide=config.divide,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
    )

    # Check if validation directories are provided in config
    if config.valid_voxel_dirs and config.valid_annotation_dirs:
        # Use separate validation dataset
        valid_dataset = FWLDataset(
            voxel_dirs=config.valid_voxel_dirs,
            annotation_dirs=config.valid_annotation_dirs,
            target_size=config.target_size,
            downsample_z=config.downsample_z,
            divide=config.divide,
            y_crop_top=config.y_crop_top,
            y_crop_bottom=config.y_crop_bottom,
            z_crop_front=config.z_crop_front,
            z_crop_back=config.z_crop_back,
        )
        train_sub_dataset = train_dataset
        valid_sub_dataset = valid_dataset
        log_info("Using separate validation dataset from config")
    else:
        # Split training dataset for validation
        indices = list(range(len(train_dataset)))
        train_indices, valid_indices = train_test_split(
            indices, test_size=0.2, random_state=config.seed
        )
        train_sub_dataset = Subset(train_dataset, train_indices)
        valid_sub_dataset = Subset(train_dataset, valid_indices)
        log_info("Using train_test_split for validation dataset")

    log_info(f"Training dataset size: {len(train_sub_dataset)}")
    log_info(f"Validation dataset size: {len(valid_sub_dataset)}")

    train_loader = DataLoader(
        train_sub_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=voxel_collate_fn,
    )

    valid_loader = DataLoader(
        valid_sub_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=voxel_collate_fn,
    )

    # Training loop
    for epoch in range(config.epochs):
        log_info(f"Epoch {epoch + 1}/{config.epochs} started")

        # Training
        train_epoch(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            current_epoch=epoch,
            loss_fn=loss_fn,
        )

        # Validation
        validation_epoch(
            config=config,
            model=model,
            device=device,
            valid_loader=valid_loader,
            current_epoch=epoch,
            loss_fn=loss_fn,
        )

    log_info("FWL-MAE Finetune Model completed!")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train FWL-MAE Finetune Model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    train_fwl_mae_finetune(args.config)


if __name__ == "__main__":
    main()

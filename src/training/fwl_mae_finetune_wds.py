"""FWL-MAE finetuning on the WebDataset shards (config_name: train_wds).

Model / loss / optimizer setup and the epoch loops are shared with
src/training/fwl_mae_finetune.py; only the dataset construction differs.
"""

import argparse
import os
from pprint import pprint

import torch

from src.config import WDSTrainingConfig, load_config_from_yaml
from src.data import FWLWDSDataset, voxel_collate_fn
from src.data.wds_utils import create_wds_loader
from src.training.fwl_mae_finetune import (
    load_pretrained_ghost_fwl_pretrain,
    train_epoch,
    validation_epoch,
)
from src.utils import (
    create_optimizer,
    create_scheduler,
    get_loss_fn,
    get_model,
    set_seed,
    set_wandb,
)
from src.utils.log import log_info, log_warning


def build_finetune_datasets(config: WDSTrainingConfig) -> tuple[FWLWDSDataset, FWLWDSDataset]:
    """Train / valid FWLWDSDataset pair according to the wds config."""
    if not config.wds_root:
        raise ValueError("wds_root must be specified for train_wds configs")
    if not config.train_wds_groups:
        log_warning("train_wds_groups is empty: using ALL samples (e.g. ['scene001', 'scene003'])")

    common = dict(
        root=config.wds_root,
        annotation_key=config.wds_annotation_key,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        divide=config.divide,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
        seed=config.seed,
        shuffle_buffer=config.wds_shuffle_buffer,
        cache_dir=config.wds_cache_dir or None,
        max_shards=config.wds_max_shards,
    )

    if config.valid_wds_groups:
        train_dataset = FWLWDSDataset(groups=config.train_wds_groups, shuffle=True, **common)
        valid_dataset = FWLWDSDataset(groups=config.valid_wds_groups, shuffle=False, **common)
        log_info("Using separate validation groups from config")
    else:
        train_dataset = FWLWDSDataset(
            groups=config.train_wds_groups,
            split="train",
            valid_ratio=config.wds_valid_ratio,
            shuffle=True,
            **common,
        )
        valid_dataset = FWLWDSDataset(
            groups=config.train_wds_groups,
            split="valid",
            valid_ratio=config.wds_valid_ratio,
            shuffle=False,
            **common,
        )
        log_info(f"Using key-hash split for validation (valid_ratio={config.wds_valid_ratio})")
    return train_dataset, valid_dataset


def train_fwl_mae_finetune_wds(config_path: str) -> None:
    """Main training function for FWL-MAE finetune on WebDataset shards."""
    config = load_config_from_yaml(config_path)
    if not isinstance(config, WDSTrainingConfig):
        raise ValueError(f"config is not WDSTrainingConfig (config_name: train_wds): {config}")

    set_seed(config.seed)
    if config.is_log:
        set_wandb(config)
    pprint(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    model = get_model(config).to(device)
    log_info(f"Model: {model.__class__.__name__}")

    if config.pretrained_model_path:
        if os.path.exists(config.pretrained_model_path):
            model = load_pretrained_ghost_fwl_pretrain(
                model, config.pretrained_model_path, device, config.freeze_encoder
            )
        else:
            log_info(
                f"Warning: Pretrained path {config.pretrained_model_path} does not exist. "
                "Training from scratch."
            )

    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        log_info(f"Loading checkpoint from: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Total parameters: {total_params:,}")
    log_info(f"Trainable parameters: {trainable_params:,}")

    loss_fn = get_loss_fn(config)
    optimizer = create_optimizer(config, model.parameters())
    scheduler = create_scheduler(optimizer, config) if config.scheduler else None

    train_dataset, valid_dataset = build_finetune_datasets(config)
    log_info(f"Training dataset size (nominal): {train_dataset.nominal_length}")
    log_info(f"Validation dataset size (nominal): {valid_dataset.nominal_length}")

    train_loader = create_wds_loader(
        train_dataset, config.batch_size, config.num_workers, voxel_collate_fn
    )
    valid_loader = create_wds_loader(
        valid_dataset, config.batch_size, config.num_workers, voxel_collate_fn
    )

    for epoch in range(config.epochs):
        log_info(f"Epoch {epoch + 1}/{config.epochs} started")
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
        validation_epoch(
            config=config,
            model=model,
            device=device,
            valid_loader=valid_loader,
            current_epoch=epoch,
            loss_fn=loss_fn,
        )

    log_info("FWL-MAE Finetune (wds) completed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FWL-MAE Finetune Model on wds shards")
    parser.add_argument("--config", type=str, required=True, help="Path to train_wds YAML")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    train_fwl_mae_finetune_wds(args.config)


if __name__ == "__main__":
    main()

"""FWL-MAE pretraining on the WebDataset shards (config_name: train_wds, mae config).

Epoch loops and checkpointing are shared with src/training/fwl_mae_pretrain.py.
"""

import argparse
import os
from pprint import pprint

import torch

from src.config import WDSTrainingConfig, load_config_from_yaml
from src.data import FWLMAEPWDSDataset, fwl_mae_collate_fn
from src.data.wds_utils import create_wds_loader
from src.training.fwl_mae_pretrain import save_model, train_epoch, validation_epoch
from src.utils import (
    create_optimizer,
    create_scheduler,
    get_loss_fn,
    get_model,
    log_info,
    log_warning,
    set_seed,
    set_wandb,
)


def build_pretrain_datasets(
    config: WDSTrainingConfig,
) -> tuple[FWLMAEPWDSDataset, FWLMAEPWDSDataset]:
    if not config.wds_root:
        raise ValueError("wds_root must be specified for train_wds configs")
    if not config.train_wds_groups:
        log_warning("train_wds_groups is empty: using ALL samples (e.g. ['ghost', 'normal'])")

    common = dict(
        root=config.wds_root,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        max_peaks=config.max_peaks,
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        divide=config.divide,
        patch_size=tuple(config.patch_size[::-1]),  # type: ignore[arg-type]
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
        train_dataset = FWLMAEPWDSDataset(groups=config.train_wds_groups, shuffle=True, **common)
        valid_dataset = FWLMAEPWDSDataset(groups=config.valid_wds_groups, shuffle=False, **common)
        log_info("Using separate validation groups from config")
    else:
        train_dataset = FWLMAEPWDSDataset(
            groups=config.train_wds_groups,
            split="train",
            valid_ratio=config.wds_valid_ratio,
            shuffle=True,
            **common,
        )
        valid_dataset = FWLMAEPWDSDataset(
            groups=config.train_wds_groups,
            split="valid",
            valid_ratio=config.wds_valid_ratio,
            shuffle=False,
            **common,
        )
        log_info(f"Using key-hash split for validation (valid_ratio={config.wds_valid_ratio})")
    return train_dataset, valid_dataset


def train_fwl_mae_pretrain_wds(config_path: str) -> None:
    """Main training function for FWL-MAE pretraining on WebDataset shards."""
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
    log_info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = get_loss_fn(config)
    optimizer = create_optimizer(config, model.parameters())
    scheduler = create_scheduler(optimizer, config) if config.scheduler else None

    train_dataset, valid_dataset = build_pretrain_datasets(config)
    train_loader = create_wds_loader(
        train_dataset,
        config.batch_size,
        config.num_workers,
        fwl_mae_collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )
    valid_loader = create_wds_loader(
        valid_dataset,
        config.batch_size,
        config.num_workers,
        fwl_mae_collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )
    log_info(f"Training dataset size (nominal): {train_dataset.nominal_length}")
    log_info(f"Validation dataset size (nominal): {valid_dataset.nominal_length}")

    for epoch in range(config.epochs):
        log_info(f"Starting epoch {epoch + 1}/{config.epochs}")
        train_epoch(config, model, optimizer, scheduler, train_loader, device, epoch, loss_fn)
        validation_epoch(config, model, device, valid_loader, epoch, loss_fn)
        if (epoch + 1) % config.save_model_interval == 0:
            save_model(config, model, optimizer, epoch + 1, 0.0)

    save_model(config, model, optimizer, config.epochs, 0.0)
    log_info("Pretraining (wds) completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FWL-MAE Pretrain Model on wds shards")
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    train_fwl_mae_pretrain_wds(args.config)

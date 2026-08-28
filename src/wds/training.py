"""Train / test entry points on the WebDataset shards (config_name: train_wds / test_wds).

Model, loss, optimizer setup and the epoch loops are shared with src/training/*;
only the dataset construction (`build_*` below) differs from the directory version.
scripts/run_train.py and scripts/run_test.py dispatch here when the YAML uses a wds config.
"""

import os
from pprint import pprint
from typing import Any, Dict

import torch

from src.config import load_config_from_yaml
from src.config.constants import LABEL_MAP
from src.data import fwl_mae_collate_fn, voxel_collate_fn
from src.training.fwl_mae_finetune import (
    load_pretrained_ghost_fwl_pretrain,
    train_epoch,
    validation_epoch,
)
from src.training.fwl_mae_finetune_test import test_model_voxel_mae_finetune
from src.training.fwl_mae_pretrain import save_model
from src.training.fwl_mae_pretrain import train_epoch as pretrain_epoch
from src.training.fwl_mae_pretrain import validation_epoch as pretrain_validation_epoch
from src.utils import (
    create_optimizer,
    create_scheduler,
    get_loss_fn,
    get_model,
    set_seed,
    set_wandb,
)
from src.utils.log import log_info, log_warning
from src.wds.config import WDSTestConfig, WDSTrainingConfig
from src.wds.dataset import FWLWDSDataset
from src.wds.dataset_mae import FWLMAEPWDSDataset
from src.wds.shards import create_wds_loader

# ------------------------------------------------------------------ finetune (ghost)


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


# ------------------------------------------------------------------ pretrain (mae)


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
        pretrain_epoch(config, model, optimizer, scheduler, train_loader, device, epoch, loss_fn)
        pretrain_validation_epoch(config, model, device, valid_loader, epoch, loss_fn)
        if (epoch + 1) % config.save_model_interval == 0:
            save_model(config, model, optimizer, epoch + 1, 0.0)

    save_model(config, model, optimizer, config.epochs, 0.0)
    log_info("Pretraining (wds) completed!")


# ------------------------------------------------------------------ test (ghost)


def build_test_dataset(config: WDSTestConfig) -> FWLWDSDataset:
    if not config.wds_root:
        raise ValueError("wds_root must be specified for test_wds configs")
    if not config.test_wds_groups:
        log_warning("test_wds_groups is empty: using ALL samples (e.g. ['scene002', 'scene007'])")
    return FWLWDSDataset(
        root=config.wds_root,
        groups=config.test_wds_groups,
        annotation_key=config.wds_annotation_key,
        target_size=config.target_size,
        downsample_z=config.downsample_z,
        divide=config.divide,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
        seed=config.seed,
        shuffle=False,
        cache_dir=config.wds_cache_dir or None,
        max_shards=config.wds_max_shards,
    )


def test_fwl_mae_finetune_wds(config_path: str) -> Dict[str, Any]:
    config = load_config_from_yaml(config_path)
    if not isinstance(config, WDSTestConfig):
        raise ValueError(f"config is not WDSTestConfig (config_name: test_wds): {config}")

    log_info("Test Configuration (wds):")
    log_info(f"  Model checkpoint: {config.checkpoint_path}")
    log_info(f"  Batch size: {config.batch_size}")
    log_info(f"  Device: {config.device}")
    log_info(
        f"  Ignored labels: {config.ignore_visualize_labels} "
        f"({[LABEL_MAP.get(i, f'Class_{i}') for i in config.ignore_visualize_labels]})"
    )
    log_info(f"  Use threshold prediction: {config.use_threshold_prediction}")
    if config.use_threshold_prediction:
        log_info(f"  Prediction threshold: {config.prediction_threshold}")

    set_seed(config.seed)
    if config.is_log:
        set_wandb(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")

    model = get_model(config).to(device)
    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")
    log_info(f"Loading checkpoint from: {config.checkpoint_path}")
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    log_info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_dataset = build_test_dataset(config)
    test_loader = create_wds_loader(
        test_dataset, config.batch_size, config.num_workers, voxel_collate_fn
    )
    log_info(f"Test dataset size (nominal): {test_dataset.nominal_length}")

    return test_model_voxel_mae_finetune(
        config=config, model=model, device=device, test_loader=test_loader
    )

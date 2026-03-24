import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    OneCycleLR,
    StepLR,
)
from torch.optim.optimizer import ParamsT

from src.config import TestConfig, TrainingConfig
from src.models import (
    FWLMAE,
    FocalLoss,
    FWLMAELoss,
    FWLMAEPretrain,
)


def create_optimizer(config: TrainingConfig, model_parameters: ParamsT) -> Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adamw":
        return optim.AdamW(model_parameters, lr=config.lr)
    elif optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=config.lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")


def create_scheduler(
    optimizer: Optimizer, config: TrainingConfig
) -> StepLR | ExponentialLR | CosineAnnealingLR | CosineAnnealingWarmRestarts | OneCycleLR | None:
    scheduler_name = config.scheduler.lower()
    if scheduler_name == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, gamma=config.gamma)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif scheduler_name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
        )
    elif scheduler_name == "onecycle":
        return OneCycleLR(optimizer, max_lr=config.max_lr, total_steps=config.steps)
    else:
        return None


def get_model(config: TrainingConfig | TestConfig) -> torch.nn.Module:
    model_name = config.model_name.lower()
    if model_name == "fwl_mae":
        return FWLMAE(
            voxel_size=config.voxel_size,
            patch_size=config.patch_size,
            in_chans=config.n_channels,
            num_classes=config.num_classes,
            embed_dim=config.encoder_embed_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            mlp_ratio=config.mlp_ratio,
        )
    elif model_name == "fwl_mae_pretrain":
        return FWLMAEPretrain(
            voxel_size=config.voxel_size,
            patch_size=config.patch_size,
            in_chans=config.n_channels,
            num_classes=config.num_classes,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            K=config.max_peaks,
            histogram_bins=config.downsample_z,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_name}")


def get_loss_fn(config: TrainingConfig) -> torch.nn.Module:
    loss_fn_name = config.loss_fn.lower()
    if loss_fn_name == "focal":
        return FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            reduction="mean",
            ignore_index=config.ignore_train_labels[0]
            if len(config.ignore_train_labels) > 0
            else -1,
        )
    elif loss_fn_name == "pretrain_loss":
        return FWLMAELoss(
            patch_size=config.patch_size,  # type: ignore[arg-type]
            position_weight=config.position_weight,
            height_weight=config.height_weight,
            width_weight=config.width_weight,
            mae_reconstruction_weight=config.mae_reconstruction_weight,
            position_loss=config.position_loss,
            height_loss=config.height_loss,
            width_loss=config.width_loss,
        )

    else:
        raise ValueError(f"Invalid loss function: {loss_fn_name}")

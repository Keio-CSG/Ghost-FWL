import os
import random
import socket
from dataclasses import asdict
from datetime import datetime
from getpass import getuser

import numpy as np
import torch
import wandb

from src.config import TestConfig, TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_wandb(config: TrainingConfig | TestConfig) -> None:
    # Get memo from standard input
    memo = input("wandb run name: ").strip()

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dict = asdict(config)  # ty: ignore[no-matching-overload]
    train_or_test = "train" if config.is_train else "test"
    user_host = f"{getuser()}@{socket.gethostname()}"
    if config.is_train:
        wandb.init(
            project="hist_lidar",
            config=config_dict,
            tags=["train", user_host],
            name=f"({memo}) {train_or_test}_{config.model_name}_{time_str}",
        )
    else:
        wandb.init(
            project="hist_lidar",
            config=config_dict,
            tags=["test", user_host],
            name=f"({memo}) {train_or_test}_{config.model_name}_{time_str}",
        )


def load_checkpoint(
    checkpoint_path: str, model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

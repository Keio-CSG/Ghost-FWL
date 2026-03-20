import argparse
import os

from src.config import TestConfig, TrainingConfig, load_config_from_yaml
from src.training import train_fwl_mae_finetune, train_fwl_mae_pretrain


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = arg_parse()
    # Load the configuration
    config: TrainingConfig | TestConfig = load_config_from_yaml(args.config)
    if config.model_name.lower() in ["fwl_mae"]:
        train_fwl_mae_finetune(args.config)
    elif config.model_name.lower() == "fwl_mae_pretrain":
        train_fwl_mae_pretrain(args.config)
    else:
        raise ValueError(f"Invalid model name: {config.model_name}")

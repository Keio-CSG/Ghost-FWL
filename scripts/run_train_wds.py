"""Train on the WebDataset shards (ryhara/Ghost-FWL). Use configs/wds/{pretrain,train}.yaml."""

import argparse
import os

from src.config import load_config_from_yaml
from src.wds.config import WDSTrainingConfig
from src.wds.finetune import train_fwl_mae_finetune_wds
from src.wds.pretrain import train_fwl_mae_pretrain_wds


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = arg_parse()
    config = load_config_from_yaml(args.config)
    if not isinstance(config, WDSTrainingConfig):
        raise ValueError("run_train_wds.py expects a config with `config_name: train_wds`")
    if config.model_name.lower() == "fwl_mae":
        train_fwl_mae_finetune_wds(args.config)
    elif config.model_name.lower() == "fwl_mae_pretrain":
        train_fwl_mae_pretrain_wds(args.config)
    else:
        raise ValueError(f"Invalid model name: {config.model_name}")

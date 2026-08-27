"""Evaluate on the WebDataset shards (ryhara/Ghost-FWL). Use configs/config_test_wds.yaml."""

import argparse

from src.config import WDSTestConfig, load_config_from_yaml
from src.training import test_fwl_mae_finetune_wds


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model testing on wds shards")
    parser.add_argument("--config", type=str, required=True, help="Path to test_wds config")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    config = load_config_from_yaml(args.config)
    if not isinstance(config, WDSTestConfig):
        raise ValueError("run_test_wds.py expects a config with `config_name: test_wds`")
    if config.model_name.lower() == "fwl_mae":
        test_fwl_mae_finetune_wds(args.config)
    else:
        raise ValueError(f"Invalid model name: {config.model_name}")

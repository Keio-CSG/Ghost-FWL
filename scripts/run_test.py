import argparse

from src.config import TestConfig, TrainingConfig, load_config_from_yaml
from src.training import test_fwl_mae_finetune


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model testing")
    parser.add_argument("--config", type=str, required=True, help="Path to test config file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    # Load the configuration
    config: TestConfig | TrainingConfig = load_config_from_yaml(args.config)
    if config.model_name.lower() in ["fwl_mae"]:
        test_fwl_mae_finetune(args.config)
    else:
        raise ValueError(f"Invalid model name: {config.model_name}")

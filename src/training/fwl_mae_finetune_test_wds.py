"""FWL-MAE finetune evaluation on the WebDataset shards (config_name: test_wds).

The metric computation is shared with src/training/fwl_mae_finetune_test.py.
"""

import argparse
import os
from typing import Any, Dict

import torch

from src.config import WDSTestConfig, load_config_from_yaml
from src.config.constants import LABEL_MAP
from src.data import FWLWDSDataset, voxel_collate_fn
from src.data.wds_utils import create_wds_loader
from src.training.fwl_mae_finetune_test import test_model_voxel_mae_finetune
from src.utils import get_model, set_seed, set_wandb
from src.utils.log import log_info, log_warning


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test FWLMAE finetune model on wds shards")
    parser.add_argument("--config", type=str, required=True, help="Path to test_wds YAML")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    results = test_fwl_mae_finetune_wds(args.config)
    peak_eval = results["peak_evaluation"]
    log_info("=" * 50)
    log_info("FINAL TEST RESULTS SUMMARY (Peak Evaluation, wds)")
    log_info("=" * 50)
    log_info(f"Peak Accuracy: {peak_eval['peak_accuracy']:.4f}")
    log_info(f"Total Peaks Detected: {peak_eval['peak_total_count']}")
    log_info(f"Peak Macro Precision: {peak_eval['peak_macro_precision']:.4f}")
    log_info(f"Peak Macro Recall: {peak_eval['peak_macro_recall']:.4f}")
    log_info(f"Peak Macro F1-Score: {peak_eval['peak_macro_f1']:.4f}")
    avg = results["scene_id_peak_average_metrics"]
    log_info(f"Average Peak Accuracy across scenes: {avg['peak_accuracy']:.4f}")
    log_info(f"Average Peak Macro F1 across scenes: {avg['peak_macro_f1']:.4f}")
    log_info("=" * 50)


if __name__ == "__main__":
    main()

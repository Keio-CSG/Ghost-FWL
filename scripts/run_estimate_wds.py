"""Sliding-window inference on the WebDataset shards (ryhara/Ghost-FWL, ghost config).

Same as scripts/run_estimate.py but reads frames selected by `test_wds_groups` from
`wds_root` (local shards or hf://ryhara/Ghost-FWL). Predictions are written to
`output_dir` as `{scene}_{hist}_{frame_id}_prediction_voxel.b2`, the layout expected by
src/visualize/vis_pcd_batch.py and evaluate_pcd_batch.py.

    uv run python scripts/run_estimate_wds.py --config configs/config_estimate_wds.yaml
"""

import argparse
import os
import pathlib
from pprint import pprint

import numpy as np
import torch
from run_estimate import SlidingWindowInference, upsampling_prediction
from tqdm import tqdm

from src.config import WDSTestConfig, load_config_from_yaml
from src.data.wds_raw import FWLWDSRawDataset
from src.utils import get_model, save_blosc2, set_seed
from src.utils.log import log_info, log_warning


def run_estimation_wds(config_path: str) -> None:
    config = load_config_from_yaml(config_path)
    if not isinstance(config, WDSTestConfig):
        raise ValueError("run_estimate_wds.py expects a config with `config_name: test_wds`")
    if not config.wds_root:
        raise ValueError("wds_root must be specified")
    if not config.output_dir:
        raise ValueError("output_dir must be specified in config")
    if not config.test_wds_groups:
        log_warning("test_wds_groups is empty: estimating on ALL frames of the ghost config")

    set_seed(config.seed)
    pprint(config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = get_model(config).to(device)
    if not config.checkpoint_path or not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {config.checkpoint_path}")
    log_info(f"Loading checkpoint from: {config.checkpoint_path}")
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()
    log_info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = FWLWDSRawDataset(
        root=config.wds_root,
        groups=config.test_wds_groups,
        config="ghost",
        members=["voxel.b2"],
        decode_voxel=True,
        downsample_z=config.downsample_z,
        y_crop_top=config.y_crop_top,
        y_crop_bottom=config.y_crop_bottom,
        z_crop_front=config.z_crop_front,
        z_crop_back=config.z_crop_back,
        cache_dir=config.wds_cache_dir or None,
        max_shards=config.wds_max_shards,
    )

    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sliding_window = SlidingWindowInference(
        model=model, device=device, target_size=config.target_size, config=config
    )

    total = dataset.nominal_length
    n_done = 0
    for sample in tqdm(dataset, total=total, desc="Estimating (wds)"):
        meta = sample["meta"]
        frame_id = meta["frame_id"]
        try:
            prediction = sliding_window.predict(sample["voxel_grid"])
            restored = upsampling_prediction(
                prediction=prediction, original_shape=sample["cropped_shape"]
            )
            if not restored.flags["C_CONTIGUOUS"]:
                restored = np.ascontiguousarray(restored)
            out = output_dir / (
                f"{meta['scene_id']}_{meta['hist_id']}_{frame_id}_prediction_voxel.b2"
            )
            save_blosc2(str(out), restored)
            n_done += 1
        except Exception as exn:  # noqa: BLE001 - keep going like run_estimate.py
            log_info(f"Error processing {frame_id}: {exn}")

    log_info(f"Estimation completed: {n_done} predictions saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on wds shards (ghost config)")
    parser.add_argument("--config", type=str, required=True, help="Path to test_wds config")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    run_estimation_wds(args.config)


if __name__ == "__main__":
    main()

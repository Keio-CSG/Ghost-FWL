"""Verify WebDataset shards produced by hf_dataset/convert_to_webdataset.py before upload.

Checks, per config directory (<root>/ghost or <root>/mae):
- every sample has the required members for its config
- keys are globally unique across all shards
- shard list and sample count match manifest.json
- (optionally, --decode) every .b2 member decodes and has the expected shape,
  and every peaks .npy is loadable

Usage:
    uv run python hf_dataset/verify_webdataset.py --root /mnt/nas5/hara/Ghost-FWL-wds --config ghost
    uv run python hf_dataset/verify_webdataset.py --root /mnt/nas5/hara/Ghost-FWL-wds --config mae --decode
"""

import argparse
import io
import json
import pathlib
import sys
import tarfile
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hf_dataset.blosc2_utils import load_blosc2_bytes  # noqa: E402
from src.utils.log import log_error, log_info, log_warning  # noqa: E402

EXPECTED_VOXEL_SHAPE = (400, 512, 700)

REQUIRED_MEMBERS = {
    "ghost": {"voxel.b2", "annotation.b2", "json"},
    "mae": {"voxel.b2", "peaks.npy", "json"},
}
OPTIONAL_MEMBERS = {
    "ghost": {"annotation_expand.b2"},
    "mae": set(),
}


def iter_tar_samples(tar_path: pathlib.Path) -> Dict[str, Dict[str, bytes]]:
    """Group tar members by WebDataset key. Returns {key: {suffix: bytes}}."""
    samples: Dict[str, Dict[str, bytes]] = defaultdict(dict)
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            key, _, suffix = name.partition(".")
            samples[key][suffix] = tar.extractfile(member).read()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--config", choices=["ghost", "mae"], required=True)
    parser.add_argument(
        "--decode", action="store_true", help="Also decode every payload (slow but thorough)"
    )
    parser.add_argument(
        "--no-shape-check",
        action="store_true",
        help=f"With --decode, do not enforce the expected voxel shape {EXPECTED_VOXEL_SHAPE}",
    )
    args = parser.parse_args()

    config_dir = args.root / args.config
    shard_paths = sorted(config_dir.glob(f"{args.config}-*.tar"))
    if not shard_paths:
        log_error(f"No shards found in {config_dir}")
        sys.exit(1)

    manifest = None
    manifest_path = config_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        log_warning(f"No manifest.json in {config_dir}; skipping manifest cross-check")

    required = REQUIRED_MEMBERS[args.config]
    optional = OPTIONAL_MEMBERS[args.config]
    errors: List[str] = []
    seen_keys: set[str] = set()
    total = 0

    for shard_path in shard_paths:
        for key, members in iter_tar_samples(shard_path).items():
            total += 1
            if key in seen_keys:
                errors.append(f"duplicate key across shards: {key}")
            seen_keys.add(key)

            missing = required - members.keys()
            if missing:
                errors.append(f"{key}: missing members {sorted(missing)}")
            unknown = members.keys() - required - optional
            if unknown:
                errors.append(f"{key}: unexpected members {sorted(unknown)}")

            if "json" in members:
                try:
                    metadata = json.loads(members["json"])
                    if "frame_id" not in metadata:
                        errors.append(f"{key}: metadata missing frame_id")
                except json.JSONDecodeError as e:
                    errors.append(f"{key}: invalid json ({e})")

            if args.decode:
                for suffix, data in members.items():
                    if suffix.endswith("b2"):
                        try:
                            array = load_blosc2_bytes(data)
                            if (
                                not args.no_shape_check
                                and tuple(array.shape) != EXPECTED_VOXEL_SHAPE
                            ):
                                errors.append(
                                    f"{key}.{suffix}: shape {tuple(array.shape)} "
                                    f"!= {EXPECTED_VOXEL_SHAPE}"
                                )
                        except Exception as e:  # noqa: BLE001
                            errors.append(f"{key}.{suffix}: decode failed ({e})")
                    elif suffix == "peaks.npy":
                        try:
                            np.load(io.BytesIO(data), allow_pickle=True)
                        except Exception as e:  # noqa: BLE001
                            errors.append(f"{key}.{suffix}: npy load failed ({e})")
        log_info(f"checked {shard_path.name} (cumulative {total} samples)")

    if manifest is not None:
        if sorted(manifest["shards"]) != [p.name for p in shard_paths]:
            errors.append("shard list does not match manifest.json")
        if manifest["num_samples"] != total:
            errors.append(f"sample count {total} != manifest num_samples {manifest['num_samples']}")

    if errors:
        for error in errors[:50]:
            log_error(error)
        if len(errors) > 50:
            log_error(f"... and {len(errors) - 50} more errors")
        log_error(f"FAILED: {len(errors)} problems in {total} samples, {len(shard_paths)} shards")
        sys.exit(1)

    log_info(f"OK: {total} samples in {len(shard_paths)} shards, all checks passed")


if __name__ == "__main__":
    main()

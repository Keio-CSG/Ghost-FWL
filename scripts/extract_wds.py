"""Extract WebDataset shards (ryhara/Ghost-FWL) back into the original directory layout.

Use this for the file-based tools (src/visualize/vis_pcd*.py, evaluate_pcd*.py,
interactive_histogram_viewer.py, scripts/run_estimate.py) that expect the layout of
docs/README_dataset.md:

    ghost:  <out>/<scene>/data/<hist>/<frame>_voxel.b2
            <out>/<scene>/annotation_<ver>/<hist>/<frame>_annotation_voxel.b2
            <out>/<scene>/annotation_<ver>_expand/<hist>/<frame>_annotation_voxel.b2
    mae:    <out>/<category>/<session>/<frame>_voxel.b2
            <out>/<category>/peaks/<session>/<frame>_peak.npy

Examples:
    uv run python scripts/extract_wds.py --config ghost --groups scene009/hist002 \
        --output /path/to/ghost_dataset
    uv run python scripts/extract_wds.py --config mae --groups ghost/20251014142232_voxel_b2 \
        --output /path/to/mae_dataset --root /mnt/nas5/hara/Ghost-FWL-wds
"""

import argparse
import pathlib
from typing import Dict

from tqdm import tqdm

from src.data.wds_raw import FWLWDSRawDataset
from src.utils.log import log_info


def target_paths(config: str, meta: Dict, out: pathlib.Path) -> Dict[str, pathlib.Path]:
    """member name -> destination path for one sample."""
    frame = meta["frame_id"]
    if config == "ghost":
        scene, hist = meta["scene_id"], meta["hist_id"]
        ver = f"annotation_{meta.get('annotation_version', 'v1')}"
        return {
            "voxel.b2": out / scene / "data" / hist / f"{frame}_voxel.b2",
            "annotation.b2": out / scene / ver / hist / f"{frame}_annotation_voxel.b2",
            "annotation_expand.b2": out
            / scene
            / f"{ver}_expand"
            / hist
            / f"{frame}_annotation_voxel.b2",
        }
    category, session = meta["category"], meta["session"]
    return {
        "voxel.b2": out / category / session / f"{frame}_voxel.b2",
        "peaks.npy": out / category / "peaks" / session / f"{frame}_peak.npy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="hf://ryhara/Ghost-FWL", help="hf://repo or local dir")
    parser.add_argument("--config", choices=["ghost", "mae"], required=True)
    parser.add_argument(
        "--groups", nargs="*", default=None, help="group selectors (all if omitted)"
    )
    parser.add_argument("--output", required=True, help="dataset root to write into")
    parser.add_argument("--cache-dir", default=None, help="hf:// only: shard cache directory")
    parser.add_argument("--members", nargs="*", default=None, help="e.g. voxel.b2 (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing files")
    args = parser.parse_args()

    dataset = FWLWDSRawDataset(
        root=args.root,
        groups=args.groups,
        config=args.config,
        members=args.members,
        cache_dir=args.cache_dir,
    )
    out = pathlib.Path(args.output)
    written = skipped = 0
    for sample in tqdm(dataset, total=dataset.nominal_length, desc=f"Extracting {args.config}"):
        for member, data in sample["members"].items():
            path = target_paths(args.config, sample["meta"], out).get(member)
            if path is None:
                continue
            if path.exists() and not args.overwrite:
                skipped += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            written += 1
    log_info(f"Extracted {written} files to {out} ({skipped} already present)")


if __name__ == "__main__":
    main()

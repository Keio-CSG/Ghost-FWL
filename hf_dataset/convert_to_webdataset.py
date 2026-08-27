"""Convert mae_dataset / ghost_dataset into WebDataset shards for the Ghost-FWL HF dataset.

The output layout matches the planned Hugging Face dataset repo (ryhara/Ghost-FWL):

    <output-root>/
    ├── ghost/
    │   ├── ghost-000000.tar
    │   ├── ...
    │   └── manifest.json
    └── mae/
        ├── mae-000000.tar
        ├── ...
        └── manifest.json

Sample layout (WebDataset: files sharing a key prefix form one sample):

    ghost:  <key>.voxel.b2  <key>.annotation.b2  [<key>.annotation_expand.b2]  <key>.json
    mae:    <key>.voxel.b2  <key>.peaks.npy  <key>.json

.b2 payloads are stored verbatim (original Blosc2 representation, no re-encoding,
no crop/downsample). Pairing is done explicitly by frame_id — never by sorted index.

Usage:
    uv run python hf_dataset/convert_to_webdataset.py \
        --input-root /mnt/nas5/hara --output-root /mnt/nas5/hara/Ghost-FWL-wds \
        --config ghost --max-shard-size 1GB

    uv run python hf_dataset/convert_to_webdataset.py \
        --input-root /mnt/nas5/hara --output-root /mnt/nas5/hara/Ghost-FWL-wds \
        --config mae --max-shard-size 1GB
"""

import argparse
import io
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import webdataset as wds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from hf_dataset.blosc2_utils import load_blosc2_bytes  # noqa: E402
from src.utils.log import log_error, log_info, log_warning  # noqa: E402

# Known corrupted frames documented in docs/README_dataset.md.
# Excluded even if the files are still present on disk.
CORRUPTED_GHOST_KEYS = {
    "scene003-hist022-20250929162519_t01759130735367000000_000043",
}

EXPECTED_VOXEL_SHAPE = (400, 512, 700)

# Fixed mtime for tar members so shard bytes are deterministic across runs.
TAR_MTIME = 0.0


def parse_size(text: str) -> int:
    """Parse a human-readable size such as '1GB', '500MB', '2GiB' or plain bytes."""
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]i?B?|B)?\s*", text, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid size: {text!r}")
    value = float(match.group(1))
    unit = (match.group(2) or "B").upper().rstrip("B").rstrip("I")
    factor = {"": 1, "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}[unit]
    return int(value * factor)


def sanitize_key(key: str) -> str:
    """Make a string safe as a WebDataset key ('.' would be parsed as an extension)."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", key)


@dataclass
class GhostSample:
    key: str
    scene_id: str
    hist_id: str
    frame_id: str
    voxel_path: pathlib.Path
    annotation_path: pathlib.Path
    annotation_expand_path: Optional[pathlib.Path]
    annotation_version: str


@dataclass
class MAESample:
    key: str
    category: str
    session: str
    frame_id: str
    voxel_path: pathlib.Path
    peak_path: pathlib.Path


@dataclass
class Issues:
    """Problems found during discovery/verification, written into the manifest."""

    missing_annotation: List[str] = field(default_factory=list)
    missing_annotation_expand: List[str] = field(default_factory=list)
    missing_voxel: List[str] = field(default_factory=list)
    missing_peaks: List[str] = field(default_factory=list)
    duplicate_keys: List[str] = field(default_factory=list)
    excluded_corrupted: List[str] = field(default_factory=list)
    verify_failed: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {k: sorted(v) for k, v in self.__dict__.items()}


def _frame_id_from_voxel(path: pathlib.Path) -> str:
    return path.stem.removesuffix("_voxel")


def _frame_id_from_annotation(path: pathlib.Path) -> str:
    return path.stem.removesuffix("_annotation_voxel")


def _frame_id_from_peak(path: pathlib.Path) -> str:
    return path.stem.removesuffix("_peak")


def _resolve_annotation_version(scene_dir: pathlib.Path, preferred: str) -> Optional[str]:
    """Pick the annotation version to use for a scene.

    Prefer `preferred` when both annotation_<v>/ and annotation_<v>_expand/ exist.
    Otherwise fall back to the newest version that has an _expand/ directory
    (some scenes were re-annotated and only ship a later version with expand).
    """
    if (scene_dir / f"annotation_{preferred}").is_dir() and (
        scene_dir / f"annotation_{preferred}_expand"
    ).is_dir():
        return preferred
    candidates = []
    for expand_dir in scene_dir.glob("annotation_v*_expand"):
        match = re.fullmatch(r"annotation_(v(\d+))_expand", expand_dir.name)
        if match is None or not (scene_dir / f"annotation_{match.group(1)}").is_dir():
            continue
        candidates.append((int(match.group(2)), match.group(1)))
    if candidates:
        return max(candidates)[1]
    if (scene_dir / f"annotation_{preferred}").is_dir():
        return preferred
    return None


def discover_ghost_samples(
    dataset_root: pathlib.Path, annotation_version: str, issues: Issues
) -> List[GhostSample]:
    """Discover ghost samples, joining voxel/annotation/annotation_expand by frame_id."""
    samples: List[GhostSample] = []
    seen_keys: set[str] = set()

    scene_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir() and "scene" in d.name)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories found under {dataset_root}")

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        data_dir = scene_dir / "data"
        if not data_dir.is_dir():
            log_warning(f"{scene_id}: no data/ directory, skipping scene")
            continue
        scene_version = _resolve_annotation_version(scene_dir, annotation_version)
        if scene_version is None:
            log_warning(f"{scene_id}: no annotation_{annotation_version}/ directory, skipping scene")
            continue
        if scene_version != annotation_version:
            log_info(f"{scene_id}: annotation_{annotation_version}_expand not found, using {scene_version}")
        annotation_dir = scene_dir / f"annotation_{scene_version}"
        expand_dir = scene_dir / f"annotation_{scene_version}_expand"

        hist_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        for hist_dir in hist_dirs:
            hist_id = hist_dir.name

            voxel_map = {_frame_id_from_voxel(p): p for p in sorted(hist_dir.glob("*_voxel.b2"))}
            annotation_map = {
                _frame_id_from_annotation(p): p
                for p in sorted((annotation_dir / hist_id).glob("*_annotation_voxel.b2"))
            }
            expand_map = {
                _frame_id_from_annotation(p): p
                for p in sorted((expand_dir / hist_id).glob("*_annotation_voxel.b2"))
            }

            for frame_id in sorted(annotation_map.keys() - voxel_map.keys()):
                issues.missing_voxel.append(f"{scene_id}/{hist_id}/{frame_id}")

            for frame_id, voxel_path in voxel_map.items():
                key = sanitize_key(f"{scene_id}-{hist_id}-{frame_id}")
                if key in CORRUPTED_GHOST_KEYS:
                    issues.excluded_corrupted.append(key)
                    continue
                if key in seen_keys:
                    issues.duplicate_keys.append(key)
                    continue
                annotation_path = annotation_map.get(frame_id)
                if annotation_path is None:
                    issues.missing_annotation.append(f"{scene_id}/{hist_id}/{frame_id}")
                    continue
                expand_path = expand_map.get(frame_id)
                if expand_path is None:
                    issues.missing_annotation_expand.append(f"{scene_id}/{hist_id}/{frame_id}")
                seen_keys.add(key)
                samples.append(
                    GhostSample(
                        key=key,
                        scene_id=scene_id,
                        hist_id=hist_id,
                        frame_id=frame_id,
                        voxel_path=voxel_path,
                        annotation_path=annotation_path,
                        annotation_expand_path=expand_path,
                        annotation_version=scene_version,
                    )
                )

    samples.sort(key=lambda s: s.key)
    return samples


def discover_mae_samples(dataset_root: pathlib.Path, issues: Issues) -> List[MAESample]:
    """Discover MAE samples, joining voxel/peaks by frame_id within each category."""
    samples: List[MAESample] = []
    seen_keys: set[str] = set()

    categories = sorted(
        d.name for d in dataset_root.iterdir() if d.is_dir() and d.name in ("ghost", "normal")
    )
    if not categories:
        raise FileNotFoundError(f"No ghost/normal directories found under {dataset_root}")

    for category in categories:
        category_dir = dataset_root / category
        peaks_root = category_dir / "peaks"

        voxel_map: Dict[str, pathlib.Path] = {}
        session_dirs = sorted(d for d in category_dir.iterdir() if d.is_dir() and d.name != "peaks")
        for session_dir in session_dirs:
            for path in sorted(session_dir.glob("*_voxel.b2")):
                frame_id = _frame_id_from_voxel(path)
                if frame_id in voxel_map:
                    issues.duplicate_keys.append(f"{category}/{frame_id}")
                    continue
                voxel_map[frame_id] = path

        peak_map: Dict[str, pathlib.Path] = {}
        if peaks_root.is_dir():
            for path in sorted(peaks_root.rglob("*_peak.npy")):
                frame_id = _frame_id_from_peak(path)
                if frame_id in peak_map:
                    issues.duplicate_keys.append(f"{category}/peaks/{frame_id}")
                    continue
                peak_map[frame_id] = path
        else:
            log_warning(f"{category}: no peaks/ directory found")

        for frame_id in sorted(peak_map.keys() - voxel_map.keys()):
            issues.missing_voxel.append(f"{category}/{frame_id}")

        for frame_id, voxel_path in voxel_map.items():
            peak_path = peak_map.get(frame_id)
            if peak_path is None:
                issues.missing_peaks.append(f"{category}/{frame_id}")
                continue
            key = sanitize_key(f"{category}-{frame_id}")
            if key in seen_keys:
                issues.duplicate_keys.append(key)
                continue
            seen_keys.add(key)
            samples.append(
                MAESample(
                    key=key,
                    category=category,
                    session=voxel_path.parent.name,
                    frame_id=frame_id,
                    voxel_path=voxel_path,
                    peak_path=peak_path,
                )
            )

    samples.sort(key=lambda s: s.key)
    return samples


def verify_b2(data: bytes, expected_shape: Optional[tuple], label: str) -> Optional[str]:
    """Decode a Blosc2 payload and check its shape. Returns an error message or None."""
    try:
        array = load_blosc2_bytes(data)
    except Exception as e:  # noqa: BLE001
        return f"{label}: failed to decode blosc2 ({e})"
    if expected_shape is not None and tuple(array.shape) != expected_shape:
        return f"{label}: shape {tuple(array.shape)} != expected {expected_shape}"
    return None


def verify_peaks(data: bytes, label: str) -> Optional[str]:
    """Check that a peaks .npy payload is loadable. Returns an error message or None."""
    try:
        np.load(io.BytesIO(data), allow_pickle=True)
    except Exception as e:  # noqa: BLE001
        return f"{label}: failed to load peaks npy ({e})"
    return None


def _write_shards(
    samples: list,
    build_wds_sample: callable,
    output_dir: pathlib.Path,
    shard_prefix: str,
    max_shard_size: int,
    issues: Issues,
) -> List[str]:
    """Write samples into sharded tars, returning the list of shard file names."""
    pattern = str(output_dir / f"{shard_prefix}-%06d.tar")
    written = 0
    total_bytes = 0
    with wds.ShardWriter(
        pattern, maxsize=max_shard_size, maxcount=1_000_000, verbose=0, mtime=TAR_MTIME
    ) as writer:
        for sample in samples:
            wds_sample = build_wds_sample(sample)
            if wds_sample is None:
                continue
            writer.write(wds_sample)
            written += 1
            total_bytes += sum(len(v) for v in wds_sample.values() if isinstance(v, bytes))
            if written % 100 == 0:
                log_info(f"  wrote {written}/{len(samples)} samples ({total_bytes / 1e9:.1f} GB)")
    log_info(f"Wrote {written} samples ({total_bytes / 1e9:.2f} GB of payload)")
    if issues.verify_failed:
        log_warning(f"{len(issues.verify_failed)} samples failed verification and were skipped")
    return sorted(p.name for p in output_dir.glob(f"{shard_prefix}-*.tar"))


def convert_ghost(args: argparse.Namespace, output_dir: pathlib.Path) -> Dict:
    dataset_root = _resolve_dataset_root(args.input_root, "ghost_dataset")
    issues = Issues()
    samples = discover_ghost_samples(dataset_root, args.annotation_version, issues)
    if args.limit:
        samples = samples[: args.limit]
    log_info(f"Discovered {len(samples)} ghost samples in {dataset_root}")
    n_expand = sum(1 for s in samples if s.annotation_expand_path is not None)
    log_info(f"  with annotation_expand: {n_expand}/{len(samples)}")

    expected = None if args.no_shape_check else EXPECTED_VOXEL_SHAPE

    def build(sample: GhostSample) -> Optional[Dict]:
        voxel = sample.voxel_path.read_bytes()
        annotation = sample.annotation_path.read_bytes()
        expand = (
            sample.annotation_expand_path.read_bytes()
            if sample.annotation_expand_path is not None
            else None
        )
        if args.verify:
            for data, name in ((voxel, "voxel"), (annotation, "annotation")):
                error = verify_b2(data, expected, f"{sample.key}.{name}")
                if error is not None:
                    log_error(error)
                    issues.verify_failed.append(error)
                    return None
            if expand is not None:
                error = verify_b2(expand, expected, f"{sample.key}.annotation_expand")
                if error is not None:
                    log_error(error)
                    issues.verify_failed.append(error)
                    return None
        wds_sample = {
            "__key__": sample.key,
            "voxel.b2": voxel,
            "annotation.b2": annotation,
            "json": {
                "frame_id": sample.frame_id,
                "scene_id": sample.scene_id,
                "hist_id": sample.hist_id,
                "annotation_version": sample.annotation_version,
                "has_annotation_expand": expand is not None,
            },
        }
        if expand is not None:
            wds_sample["annotation_expand.b2"] = expand
        return wds_sample

    shards = _write_shards(samples, build, output_dir, "ghost", args.max_shard_size, issues)
    return {
        "config": "ghost",
        "annotation_version": args.annotation_version,
        "num_samples": len(samples) - len(issues.verify_failed),
        "num_samples_with_annotation_expand": n_expand,
        "shards": shards,
        "issues": issues.as_dict(),
    }


def convert_mae(args: argparse.Namespace, output_dir: pathlib.Path) -> Dict:
    dataset_root = _resolve_dataset_root(args.input_root, "mae_dataset")
    issues = Issues()
    samples = discover_mae_samples(dataset_root, issues)
    if args.limit:
        samples = samples[: args.limit]
    log_info(f"Discovered {len(samples)} mae samples in {dataset_root}")
    for category in ("ghost", "normal"):
        count = sum(1 for s in samples if s.category == category)
        log_info(f"  category {category}: {count}")

    expected = None if args.no_shape_check else EXPECTED_VOXEL_SHAPE

    def build(sample: MAESample) -> Optional[Dict]:
        voxel = sample.voxel_path.read_bytes()
        peaks = sample.peak_path.read_bytes()
        if args.verify:
            error = verify_b2(voxel, expected, f"{sample.key}.voxel") or verify_peaks(
                peaks, f"{sample.key}.peaks"
            )
            if error is not None:
                log_error(error)
                issues.verify_failed.append(error)
                return None
        return {
            "__key__": sample.key,
            "voxel.b2": voxel,
            "peaks.npy": peaks,
            "json": {
                "frame_id": sample.frame_id,
                "category": sample.category,
                "session": sample.session,
            },
        }

    shards = _write_shards(samples, build, output_dir, "mae", args.max_shard_size, issues)
    return {
        "config": "mae",
        "num_samples": len(samples) - len(issues.verify_failed),
        "num_samples_per_category": {
            category: sum(1 for s in samples if s.category == category)
            for category in ("ghost", "normal")
        },
        "shards": shards,
        "issues": issues.as_dict(),
    }


def _resolve_dataset_root(input_root: pathlib.Path, dataset_dirname: str) -> pathlib.Path:
    """Accept either the parent directory or the dataset directory itself."""
    if (input_root / dataset_dirname).is_dir():
        return input_root / dataset_dirname
    if input_root.name == dataset_dirname and input_root.is_dir():
        return input_root
    raise FileNotFoundError(f"{dataset_dirname} not found under {input_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=pathlib.Path,
        required=True,
        help="Directory containing mae_dataset/ and ghost_dataset/ (or the dataset dir itself)",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        required=True,
        help="Output directory; shards are written to <output-root>/<config>/",
    )
    parser.add_argument("--config", choices=["ghost", "mae"], required=True)
    parser.add_argument(
        "--max-shard-size",
        type=parse_size,
        default=parse_size("1GB"),
        help="Maximum shard size (e.g. 1GB, 500MB). Default: 1GB",
    )
    parser.add_argument(
        "--annotation-version",
        default="v1",
        help="Annotation version to include for the ghost config (default: v1)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip decoding/validating every payload (faster, less safe)",
    )
    parser.add_argument(
        "--no-shape-check",
        action="store_true",
        help=f"Do not enforce the expected voxel shape {EXPECTED_VOXEL_SHAPE}",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Convert only the first N samples (smoke test)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow writing into a non-empty output directory"
    )
    args = parser.parse_args()

    output_dir = args.output_root / args.config
    existing_shards = sorted(output_dir.glob("*.tar")) if output_dir.exists() else []
    if existing_shards and not args.overwrite:
        parser.error(f"{output_dir} already contains shards; pass --overwrite to replace them")
    for path in existing_shards:
        path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config == "ghost":
        manifest = convert_ghost(args, output_dir)
    else:
        manifest = convert_mae(args, output_dir)

    manifest["max_shard_size"] = args.max_shard_size
    manifest["verified"] = args.verify
    manifest["num_shards"] = len(manifest["shards"])

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log_info(f"Manifest written to {manifest_path}")

    for name, entries in manifest["issues"].items():
        if entries:
            log_warning(f"issues.{name}: {len(entries)} (see manifest.json)")
    log_info(f"Done: {manifest['num_samples']} samples in {manifest['num_shards']} shards")


if __name__ == "__main__":
    main()

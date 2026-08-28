"""Shared helpers for the WebDataset (wds) loaders in src/data.

Covers:
  * resolving where the shards live (local directory or Hugging Face Hub repo)
  * the per-shard group index (`shard_index.json`) used to skip shards that do
    not contain any requested scene/hist (ghost) or category/session (mae)
  * group selectors (fnmatch patterns on "scene001/hist003", "ghost/2025..." ...)
  * deterministic key-hash based train/valid splitting and `divide` subsampling

Shard layout (ryhara/Ghost-FWL on the Hub, see docs/README_huggingface.md):

    <root>/<config>/<config>-NNNNNN.tar   + manifest.json  [+ shard_index.json]
"""

import fnmatch
import hashlib
import json
import os
import pathlib
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.utils.log import log_info, log_warning

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, IterableDataset

WDS_CONFIGS = ("ghost", "mae")
SHARD_INDEX_FILENAME = "shard_index.json"
MANIFEST_FILENAME = "manifest.json"
HF_PREFIX = "hf://"


def group_of(meta: Dict, config: str) -> str:
    """Group id of a sample from its json metadata.

    ghost: "<scene_id>/<hist_id>"   mae: "<category>/<session>"
    """
    if config == "ghost":
        return f"{meta['scene_id']}/{meta['hist_id']}"
    if config == "mae":
        return f"{meta['category']}/{meta['session']}"
    raise ValueError(f"Unknown wds config: {config}")


def match_group(group: str, patterns: Optional[Sequence[str]]) -> bool:
    """True if `group` matches any pattern. `None`/empty patterns match everything.

    A pattern without "/" matches the whole first level ("scene003" == "scene003/*").
    fnmatch wildcards are allowed ("scene00[1-5]/hist*").
    """
    if not patterns:
        return True
    for pattern in patterns:
        if "/" not in pattern:
            pattern = pattern + "/*"
        if fnmatch.fnmatchcase(group, pattern):
            return True
    return False


def key_hash(key: str, seed: int = 0) -> float:
    """Stable hash of a sample key in [0, 1)."""
    digest = hashlib.md5(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def in_split(key: str, split: Optional[str], valid_ratio: float, seed: int) -> bool:
    """Deterministic train/valid membership (replaces sklearn train_test_split)."""
    if split is None or split == "all":
        return True
    is_valid = key_hash(key, seed) < valid_ratio
    if split == "valid":
        return is_valid
    if split == "train":
        return not is_valid
    raise ValueError(f"Unknown split: {split}")


def keep_by_divide(key: str, divide: int, seed: int) -> bool:
    """Deterministic 1/divide subsampling (replaces random.sample in the legacy datasets)."""
    if divide <= 1:
        return True
    return key_hash(key, seed + 1_000_003) < 1.0 / divide


# --------------------------------------------------------------------------------------
# Shard location
# --------------------------------------------------------------------------------------


@dataclass
class ShardSource:
    config: str
    root: str  # local directory or "hf://<repo_id>"
    shard_names: List[str]  # e.g. ["ghost-000000.tar", ...]
    urls: List[str]  # something wds.WebDataset can open, aligned with shard_names
    manifest: Dict = field(default_factory=dict)
    shard_index: Optional[Dict[str, Dict[str, int]]] = None  # shard -> {group: count}
    local_dir: Optional[pathlib.Path] = None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None

    def url_of(self, shard_name: str) -> str:
        return self.urls[self.shard_names.index(shard_name)]


def _read_json(path: pathlib.Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _hf_resolve(repo_id: str, config: str, cache_dir: Optional[str]) -> ShardSource:
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id, f"{config}/{MANIFEST_FILENAME}", repo_type="dataset", cache_dir=cache_dir
    )
    manifest = _read_json(pathlib.Path(manifest_path)) or {}
    shard_index = None
    try:
        index_path = hf_hub_download(
            repo_id, f"{config}/{SHARD_INDEX_FILENAME}", repo_type="dataset", cache_dir=cache_dir
        )
        shard_index = (_read_json(pathlib.Path(index_path)) or {}).get("shards")
    except Exception:  # noqa: BLE001 - index is optional on the Hub
        log_warning(f"{repo_id}: {config}/{SHARD_INDEX_FILENAME} not found, reading all shards")

    shard_names = list(manifest.get("shards", []))
    # Shards are read through a "pipe:" helper (src/data/wds_fetch.py):
    #   fetch  (cache_dir set)   hf_hub_download -> cat. Resumable, sha256-verified, reused
    #                            across epochs/runs; costs disk (~1 GB per shard).
    #   stream (cache_dir unset) HTTP streaming with Range-resume. No disk; a network drop
    #                            mid-shard reconnects at the last byte received instead of
    #                            truncating the tar (curl) or restarting it (curl --retry).
    mode, cache_arg = ("fetch", f" '{cache_dir}'") if cache_dir else ("stream", "")
    helper = pathlib.Path(__file__).with_name("wds_fetch.py")  # standalone: no package import
    urls = [
        f"pipe:{sys.executable} {helper} {mode} {repo_id} {config}/{name}{cache_arg}"
        for name in shard_names
    ]
    log_info(
        f"{repo_id}: {len(shard_names)} {config} shards, "
        + (f"cached in {cache_dir}" if cache_dir else "streamed with Range-resume (no cache)")
    )
    return ShardSource(
        config=config,
        root=f"{HF_PREFIX}{repo_id}",
        shard_names=shard_names,
        urls=urls,
        manifest=manifest,
        shard_index=shard_index,
    )


def resolve_shards(root: str, config: str, cache_dir: Optional[str] = None) -> ShardSource:
    """Locate the shards of `config` under `root`.

    root may be a local directory produced by convert_to_webdataset.py
    ("/mnt/.../Ghost-FWL-wds") or a Hugging Face dataset repo ("hf://ryhara/Ghost-FWL").
    """
    if config not in WDS_CONFIGS:
        raise ValueError(f"wds config must be one of {WDS_CONFIGS}, got {config!r}")
    if root.startswith(HF_PREFIX):
        return _hf_resolve(root[len(HF_PREFIX) :], config, cache_dir)

    config_dir = pathlib.Path(root) / config
    if not config_dir.is_dir():
        raise FileNotFoundError(f"WebDataset config directory not found: {config_dir}")
    manifest = _read_json(config_dir / MANIFEST_FILENAME) or {}
    shard_names = list(manifest.get("shards", [])) or sorted(
        p.name for p in config_dir.glob(f"{config}-*.tar")
    )
    if not shard_names:
        raise FileNotFoundError(f"No {config}-*.tar shards found in {config_dir}")
    index_data = _read_json(config_dir / SHARD_INDEX_FILENAME)
    return ShardSource(
        config=config,
        root=root,
        shard_names=shard_names,
        urls=[str(config_dir / name) for name in shard_names],
        manifest=manifest,
        shard_index=index_data.get("shards") if index_data else None,
        local_dir=config_dir,
    )


# --------------------------------------------------------------------------------------
# Shard index
# --------------------------------------------------------------------------------------


def index_one_shard(path: pathlib.Path, config: str) -> Dict[str, int]:
    """Count samples per group in one shard by reading only its .json members."""
    counts: Dict[str, int] = {}
    with tarfile.open(path, "r:") as tf:  # seekable: skips over the big .b2 payloads
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            meta = json.load(tf.extractfile(member))
            group = group_of(meta, config)
            counts[group] = counts.get(group, 0) + 1
    return counts


def build_shard_index(
    config_dir: pathlib.Path, config: str, workers: int = 8, force: bool = False
) -> Dict[str, Dict[str, int]]:
    """Build (and cache as shard_index.json) the shard -> {group: count} index."""
    index_path = config_dir / SHARD_INDEX_FILENAME
    if index_path.is_file() and not force:
        return _read_json(index_path)["shards"]

    manifest = _read_json(config_dir / MANIFEST_FILENAME) or {}
    shard_names = list(manifest.get("shards", [])) or sorted(
        p.name for p in config_dir.glob(f"{config}-*.tar")
    )
    log_info(
        f"Building shard index for {config_dir} ({len(shard_names)} shards, {workers} workers)"
    )

    def work(name: str) -> tuple[str, Dict[str, int]]:
        return name, index_one_shard(config_dir / name, config)

    shards: Dict[str, Dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (name, counts) in enumerate(pool.map(work, shard_names), 1):
            shards[name] = counts
            if i % 20 == 0 or i == len(shard_names):
                log_info(f"  indexed {i}/{len(shard_names)} shards")

    groups: Dict[str, int] = {}
    for counts in shards.values():
        for group, n in counts.items():
            groups[group] = groups.get(group, 0) + n
    payload = {
        "config": config,
        "num_samples": sum(groups.values()),
        "num_shards": len(shards),
        "groups": dict(sorted(groups.items())),
        "shards": {name: dict(sorted(shards[name].items())) for name in shard_names},
    }
    index_path.write_text(json.dumps(payload, indent=2) + "\n")
    log_info(f"Shard index written to {index_path}")
    return payload["shards"]


def select_shards(
    source: ShardSource, groups: Optional[Sequence[str]]
) -> tuple[List[str], Optional[int]]:
    """Shard urls that may contain the requested groups, and the exact sample count if known.

    Without an index every shard is returned (samples are still filtered per-sample
    by the loader) and the count is only known when no group filter is given.
    """
    if source.shard_index is None:
        if groups:
            log_warning(
                f"{source.root}/{source.config}: no {SHARD_INDEX_FILENAME}; all "
                f"{len(source.urls)} shards will be scanned and the dataset length is unknown. "
                "Create it with build_shard_index() in src/data/wds_utils.py."
            )
            return list(source.urls), None
        return list(source.urls), source.manifest.get("num_samples")

    urls: List[str] = []
    total = 0
    for name in source.shard_names:
        counts = source.shard_index.get(name, {})
        n = sum(c for group, c in counts.items() if match_group(group, groups))
        if n > 0:
            urls.append(source.url_of(name))
            total += n
    return urls, total


def estimate_length(
    exact: Optional[int], split: Optional[str], valid_ratio: float, divide: int
) -> Optional[int]:
    """Expected number of samples after split / divide filtering (for DataLoader length)."""
    if exact is None:
        return None
    n = float(exact)
    if split == "valid":
        n *= valid_ratio
    elif split == "train":
        n *= 1.0 - valid_ratio
    if divide > 1:
        n /= divide
    return max(1, int(round(n)))


def iter_group_names(shard_index: Dict[str, Dict[str, int]]) -> Iterable[str]:
    seen = set()
    for counts in shard_index.values():
        for group in counts:
            if group not in seen:
                seen.add(group)
                yield group


# --------------------------------------------------------------------------------------
# DataLoader helpers
# --------------------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own numpy RNG stream.

    torch seeds `random` and `torch` per worker but not numpy, so without this every
    worker would draw the same random-crop origins.
    """
    import numpy as np
    import torch

    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def create_wds_loader(
    dataset: "IterableDataset",
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    **kwargs: Any,
) -> "DataLoader":
    """DataLoader over an iterable wds dataset (no sampler shuffle, per-worker seeding)."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # shuffling is done inside the wds pipeline
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        **kwargs,
    )

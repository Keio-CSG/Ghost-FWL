"""Helpers for tools that need the WebDataset shards in a non-training shape.

* `FWLWDSRawDataset`  — iterable over *full-frame* samples (crop / downsample but no
  random crop), used by scripts/run_estimate_wds.py and scripts/extract_wds.py.
* `WDSSequentialAccess` — wraps any wds IterableDataset so index-based tools
  (src/visualize/vis_pred.py) can call `len()`, `dataset[i]`, `get_sample_info(i)`.
  Samples are pulled sequentially and cached, so random access is cheap only for
  indices already visited.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

import webdataset as wds
from torch.utils.data import IterableDataset

from src.data.transforms_wds import DownsampleZ, YCrop, ZCrop, decode_b2
from src.data.wds_utils import group_of, match_group, resolve_shards, select_shards
from src.utils.log import log_info


class FWLWDSRawDataset(IterableDataset):
    """Yield every sample of the selected groups as raw tar members + metadata.

    Each item is {"__key__", "meta": dict, "members": {member_name: bytes}} where
    member_name is e.g. "voxel.b2", "annotation_expand.b2", "peaks.npy".
    If `decode_voxel` is set, "voxel_grid" holds the decoded voxel after y/z crop and
    z downsample (no random crop) and "cropped_shape" its shape before the downsample,
    matching SimpleFWLDataset in scripts/run_estimate.py.
    """

    def __init__(
        self,
        root: str,
        groups: Optional[Sequence[str]] = None,
        config: str = "ghost",
        members: Optional[Sequence[str]] = None,
        decode_voxel: bool = False,
        downsample_z: Optional[int] = None,
        y_crop_top: int = 0,
        y_crop_bottom: int = 0,
        z_crop_front: int = 0,
        z_crop_back: int = 0,
        cache_dir: Optional[str] = None,
        max_shards: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.groups = list(groups) if groups else None
        self.members = set(members) if members else None  # None -> keep every member
        self.decode_voxel = decode_voxel
        self.y_crop = YCrop(y_crop_top, y_crop_bottom)
        self.z_crop = ZCrop(z_crop_front, z_crop_back)
        self.downsample = DownsampleZ(downsample_z)
        self.source = resolve_shards(root, config, cache_dir)
        self.urls, self.nominal_length = select_shards(self.source, self.groups)
        if max_shards > 0 and len(self.urls) > max_shards:
            self.nominal_length = None
            self.urls = self.urls[:max_shards]
        log_info(
            f"[FWLWDSRawDataset] {self.source.root}/{config}: "
            f"{len(self.urls)}/{len(self.source.urls)} shards, groups={self.groups}, "
            f"~{self.nominal_length} samples"
        )

    def __len__(self) -> int:
        if self.nominal_length is None:
            raise TypeError("Dataset length is unknown (max_shards set or no shard_index.json)")
        return self.nominal_length

    def _wanted(self, name: str) -> bool:
        if name.endswith(".json") or self.members is None:
            return True
        return any(name.endswith(f".{m}") for m in self.members)

    def _process(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        meta = json.loads(raw["json"])
        if not match_group(group_of(meta, self.config), self.groups):
            return None
        members = {k: v for k, v in raw.items() if not k.startswith("__") and k != "json"}
        sample: Dict[str, Any] = {"__key__": raw["__key__"], "meta": meta, "members": members}
        if self.decode_voxel:
            voxel = self.z_crop(self.y_crop(decode_b2(members["voxel.b2"])))
            sample["cropped_shape"] = voxel.shape
            sample["voxel_grid"] = self.downsample(voxel)
        return sample

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pipeline = wds.WebDataset(
            self.urls, shardshuffle=False, select_files=self._wanted, empty_check=False
        ).map(self._process)
        return iter(pipeline)


class WDSSequentialAccess:
    """Index-style facade over an IterableDataset (sequential pull + cache)."""

    def __init__(self, dataset: IterableDataset) -> None:
        self.dataset = dataset
        self._iter: Optional[Iterator[Dict[str, Any]]] = None
        self._cache: List[Dict[str, Any]] = []
        self._exhausted = False

    def __len__(self) -> int:
        if self._exhausted:
            return len(self._cache)
        nominal = getattr(self.dataset, "nominal_length", None)
        return nominal if nominal is not None else len(self.dataset)

    def _fill(self, index: int) -> None:
        if self._iter is None:
            self._iter = iter(self.dataset)
        while len(self._cache) <= index and not self._exhausted:
            try:
                self._cache.append(next(self._iter))
            except StopIteration:
                self._exhausted = True

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._fill(index)
        if index >= len(self._cache):
            raise IndexError(f"index {index} out of range (dataset has {len(self._cache)})")
        return self._cache[index]

    def get_sample_info(self, index: int) -> Dict[str, Any]:
        sample = self[index]
        return {k: sample.get(k, "unknown") for k in ("frame_id", "scene_id", "hist_id")}

"""WebDataset-backed counterpart of FWLDataset (ghost config of ryhara/Ghost-FWL).

Reads the WebDataset shards of ryhara/Ghost-FWL and yields sample
dicts with exactly the same keys / dtypes / shapes as FWLDataset.__getitem__:

    {"frame_id", "scene_id", "hist_id", "voxel_grid" (X, Y, Z), "annotation" (X, Y, Z)}

so the existing `voxel_collate_fn` and training loops can be reused unchanged.

Differences from the directory-based dataset:
  * it is an IterableDataset (WebDataset streams tar shards); use shuffle=False in
    the DataLoader and let the dataset shuffle shards / a sample buffer itself.
  * data are selected by *group* selectors ("scene003", "scene003/hist012",
    "scene00[1-5]/hist*") instead of directory lists.
  * train/valid splitting and `divide` subsampling are deterministic functions of
    the sample key (see wds_utils.in_split / keep_by_divide).
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

import webdataset as wds
from torch.utils.data import IterableDataset

from src.config import LABEL_MAP
from src.data.transforms_wds import VoxelPreprocess, decode_b2
from src.data.wds_utils import (
    ShardSource,
    estimate_length,
    group_of,
    in_split,
    keep_by_divide,
    match_group,
    resolve_shards,
    select_shards,
)
from src.utils.log import log_info

ANNOTATION_KEYS = ("annotation_expand", "annotation")


class FWLWDSDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        groups: Optional[Sequence[str]] = None,
        annotation_key: str = "annotation_expand",
        target_size: Optional[List[int]] = None,
        downsample_z: Optional[int] = None,
        divide: int = 1,
        ignore_label: int = 0,
        y_crop_top: int = 0,
        y_crop_bottom: int = 0,
        z_crop_front: int = 0,
        z_crop_back: int = 0,
        split: Optional[str] = None,
        valid_ratio: float = 0.2,
        seed: int = 42,
        shuffle: bool = False,
        shuffle_buffer: int = 16,
        cache_dir: Optional[str] = None,
        max_shards: int = 0,
        config: str = "ghost",
    ) -> None:
        """
        Args:
            root: Directory holding `<config>/<config>-*.tar` (output of
                convert_to_webdataset.py) or a Hub repo as "hf://ryhara/Ghost-FWL".
            groups: Group selectors; None/empty selects every sample.
            annotation_key: "annotation_expand" (annotation_v*_expand) or "annotation".
            split: None (all), "train" or "valid" — deterministic key-hash split with
                `valid_ratio`, replacing train_test_split on the legacy dataset.
            shuffle: Shuffle shard order and a `shuffle_buffer`-sample buffer.
            max_shards: Debug knob — only read the first N selected shards (0 = all).
            The remaining arguments mirror FWLDataset.
        """
        super().__init__()
        if annotation_key not in ANNOTATION_KEYS:
            raise ValueError(f"annotation_key must be one of {ANNOTATION_KEYS}: {annotation_key}")
        self.config = config
        self.groups = list(groups) if groups else None
        self.annotation_key = annotation_key
        self.annotation_member = f"{annotation_key}.b2"
        self.divide = divide
        self.ignore_label = ignore_label
        self.split = split
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer

        self.preprocess = VoxelPreprocess(
            target_size=target_size,
            downsample_z=downsample_z,
            y_crop_top=y_crop_top,
            y_crop_bottom=y_crop_bottom,
            z_crop_front=z_crop_front,
            z_crop_back=z_crop_back,
        )
        self.target_size = self.preprocess.target_size
        self.downsample_z = downsample_z

        self.label_names = LABEL_MAP
        self.num_classes = len(self.label_names)

        self.source: ShardSource = resolve_shards(root, config, cache_dir)
        self.urls, exact = select_shards(self.source, self.groups)
        if max_shards > 0 and len(self.urls) > max_shards:
            # Debug mode: keep the first shards; the length becomes a pro-rata estimate.
            exact = None if exact is None else round(exact * max_shards / len(self.urls))
            self.urls = self.urls[:max_shards]
            log_info(f"  max_shards={max_shards}: using {max_shards} shards (length estimated)")
        self._length = estimate_length(exact, split, valid_ratio, divide)
        self._log_summary(exact)

    # ------------------------------------------------------------------ setup / info
    def _log_summary(self, exact: Optional[int]) -> None:
        log_info(
            f"[{self.__class__.__name__}] {self.source.root}/{self.config}: "
            f"{len(self.urls)}/{len(self.source.urls)} shards selected"
            + (
                f", groups={self.groups}"
                if self.groups and len(self.groups) <= 8
                else f", {len(self.groups)} groups ({self.groups[0]} ... {self.groups[-1]})"
                if self.groups
                else ""
            )
        )
        if exact is not None:
            log_info(
                f"  {exact} samples in selected groups; split={self.split} -> ~{self._length}"
                + (f" (divide={self.divide})" if self.divide > 1 else "")
            )
        if self.preprocess.target_shape:
            log_info(f"  Target size for cropping: {self.preprocess.target_shape}")
        if self.downsample_z:
            log_info(f"  Target z size for histogram downsampling: {self.downsample_z}")

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError(
                "Dataset length is unknown (no shard_index.json and a group filter is set). "
                "Create it with build_shard_index() in src/data/wds_utils.py."
            )
        return self._length

    @property
    def nominal_length(self) -> Optional[int]:
        return self._length

    # ------------------------------------------------------------------ pipeline
    def _wanted_members(self, name: str) -> bool:
        """Only read the tar members we actually use (skips the unused annotation variant)."""
        return not name.endswith(".b2") or name.endswith(
            (".voxel.b2", f".{self.annotation_member}")
        )

    def _accept(self, key: str, meta: Dict[str, Any]) -> bool:
        return (
            match_group(group_of(meta, self.config), self.groups)
            and in_split(key, self.split, self.valid_ratio, self.seed)
            and keep_by_divide(key, self.divide, self.seed)
        )

    def _process(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Raw wds sample (bytes) -> FWLDataset-style sample, or None to drop it."""
        key = raw["__key__"]
        meta = json.loads(raw["json"])
        if not self._accept(key, meta):
            return None
        if self.annotation_member not in raw:
            raise KeyError(f"{key}: member {self.annotation_member} missing from shard")

        voxel_grid = decode_b2(raw["voxel.b2"])
        annotation_voxel = decode_b2(raw[self.annotation_member])

        voxel_grid = self.preprocess.prepare(voxel_grid)
        annotation_voxel = self.preprocess.prepare(annotation_voxel)

        start_coords = self.preprocess.sample_coords(voxel_grid.shape)
        voxel_grid = self.preprocess.crop(voxel_grid, start_coords)
        annotation_voxel = self.preprocess.crop(annotation_voxel, start_coords)

        return {
            "frame_id": meta["frame_id"],
            "scene_id": meta["scene_id"],
            "hist_id": meta["hist_id"],
            "voxel_grid": voxel_grid,
            "annotation": annotation_voxel,
        }

    def _build_pipeline(self) -> wds.WebDataset:
        dataset = wds.WebDataset(
            self.urls,
            shardshuffle=len(self.urls) if self.shuffle else False,
            select_files=self._wanted_members,
            empty_check=False,  # workers may receive no shard when num_workers > shards
        )
        dataset = dataset.map(self._process)  # wds.map drops samples mapped to None
        if self.shuffle and self.shuffle_buffer > 1:
            dataset = dataset.shuffle(self.shuffle_buffer)
        return dataset

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._build_pipeline())

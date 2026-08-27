"""WebDataset-backed counterpart of FWLMAEPDataset (mae config of ryhara/Ghost-FWL).

Yields sample dicts identical to FWLMAEPDataset.__getitem__:

    {"frame_id", "patch_size", "mask_ratio",
     "original_voxel" (X, Y, Z), "peak_positions" / "peak_heights" / "peak_widths" (K, X, Y)}

so `fwl_mae_collate_fn` and the pretraining loop are reused unchanged.
Groups are "<category>/<session>" ("ghost/20251014142232_voxel_b2", "normal", ...).
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from src.data.dataset_fwl_wds import FWLWDSDataset
from src.data.transforms_wds import (
    PeaksToTensors,
    crop_peak_tensors,
    decode_b2,
    decode_peaks,
)
from src.utils.log import log_info


class FWLMAEPWDSDataset(FWLWDSDataset):
    def __init__(
        self,
        root: str,
        groups: Optional[Sequence[str]] = None,
        target_size: Optional[List[int]] = None,
        downsample_z: Optional[int] = None,
        max_peaks: int = 4,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0,
        divide: int = 1,
        patch_size: tuple[int, int, int] = (4, 4, 128),
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
    ) -> None:
        self.max_peaks = max_peaks
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.patch_size = patch_size
        self.peaks_to_tensors = PeaksToTensors(max_peaks)
        super().__init__(
            root=root,
            groups=groups,
            annotation_key="annotation",  # unused for mae; keeps the base class happy
            target_size=target_size,
            downsample_z=downsample_z,
            divide=divide,
            y_crop_top=y_crop_top,
            y_crop_bottom=y_crop_bottom,
            z_crop_front=z_crop_front,
            z_crop_back=z_crop_back,
            split=split,
            valid_ratio=valid_ratio,
            seed=seed,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            cache_dir=cache_dir,
            max_shards=max_shards,
            config="mae",
        )
        log_info(f"  Max peaks per voxel: {self.max_peaks}")

    def _wanted_members(self, name: str) -> bool:
        return name.endswith((".voxel.b2", ".peaks.npy", ".json"))

    def _process(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = raw["__key__"]
        meta = json.loads(raw["json"])
        if not self._accept(key, meta):
            return None

        voxel_grid = decode_b2(raw["voxel.b2"])
        voxel_grid = self.preprocess.prepare(voxel_grid)

        # Peaks are rasterised on the (already y/z-cropped, downsampled) grid before the
        # random crop, then cropped with the same origin — same order as FWLMAEPDataset.
        peak_data = decode_peaks(raw["peaks.npy"])
        peak_tensors = self.peaks_to_tensors(peak_data, voxel_grid.shape)

        start_coords = self.preprocess.sample_coords(voxel_grid.shape)
        voxel_grid = self.preprocess.crop(voxel_grid, start_coords)
        if self.preprocess.target_shape is not None:
            peak_tensors = crop_peak_tensors(
                peak_tensors, start_coords, self.preprocess.target_shape
            )

        return {
            "frame_id": meta["frame_id"],
            "category": meta["category"],
            "session": meta["session"],
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "original_voxel": voxel_grid,
            "peak_positions": peak_tensors["peak_positions"],
            "peak_heights": peak_tensors["peak_heights"],
            "peak_widths": peak_tensors["peak_widths"],
        }

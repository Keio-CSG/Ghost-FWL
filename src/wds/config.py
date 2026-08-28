"""Config dataclasses for training / testing on the WebDataset (wds) shards.

They extend the directory-based configs with the wds-specific fields, so every
model / optimizer / preprocessing option keeps the same name. Select them with
`config_name: train_wds` / `config_name: test_wds` in the YAML.

Group selectors (`*_wds_groups`) replace the directory lists:
    ghost: "scene003", "scene003/hist012", "scene00[1-5]/hist*"
    mae:   "ghost", "normal/20251014142232", "ghost/*"
"""

from dataclasses import dataclass, field

from src.config.config import TestConfig, TrainingConfig


@dataclass
class WDSTrainingConfig(TrainingConfig):
    config_name: str = "train_wds"
    # Local directory with <config>/<config>-NNNNNN.tar shards, or "hf://ryhara/Ghost-FWL"
    wds_root: str = ""
    # "annotation_expand" (annotation_v*_expand, default) or "annotation" (annotation_v*)
    wds_annotation_key: str = "annotation_expand"
    train_wds_groups: list[str] = field(default_factory=lambda: [])
    valid_wds_groups: list[str] = field(default_factory=lambda: [])
    # Used when valid_wds_groups is empty: key-hash split of the training groups
    wds_valid_ratio: float = 0.2
    # Sample shuffle buffer (samples are already cropped, ~8 MB each for 128x128x256 uint16)
    wds_shuffle_buffer: int = 16
    # Cache directory for Hub downloads (only used with hf:// roots)
    wds_cache_dir: str = ""
    # Debug: read only the first N selected shards (0 = all). Length becomes unknown.
    wds_max_shards: int = 0


@dataclass
class WDSTestConfig(TestConfig):
    config_name: str = "test_wds"
    wds_root: str = ""
    wds_annotation_key: str = "annotation_expand"
    test_wds_groups: list[str] = field(default_factory=lambda: [])
    wds_cache_dir: str = ""
    wds_max_shards: int = 0

"""WebDataset support for ryhara/Ghost-FWL (see docs/README_huggingface.md).

Everything needed to train / test / estimate on the WebDataset shards, kept apart from
the directory-based implementation in src/data, src/config and src/training:

    shards.py       shard resolution (local / hf://), shard index, key-hash split, loader
    fetch.py        `pipe:` helper that streams or caches hf:// shards (standalone script)
    transforms.py   crop / downsample / random-crop / peaks transforms on decoded arrays
    dataset.py      FWLWDSDataset      (ghost config, finetune / test)
    dataset_mae.py  FWLMAEPWDSDataset  (mae config, pretrain)
    raw.py          FWLWDSRawDataset (full frames) and WDSSequentialAccess (index facade)
    config.py       WDSTrainingConfig / WDSTestConfig (config_name: train_wds / test_wds)
    pretrain.py, finetune.py, test.py   entry points used by scripts/run_*_wds.py
"""

from .config import WDSTestConfig, WDSTrainingConfig
from .dataset import FWLWDSDataset
from .dataset_mae import FWLMAEPWDSDataset
from .raw import FWLWDSRawDataset, WDSSequentialAccess
from .shards import build_shard_index, create_wds_loader, resolve_shards, select_shards

__all__ = [
    "FWLMAEPWDSDataset",
    "FWLWDSDataset",
    "FWLWDSRawDataset",
    "WDSSequentialAccess",
    "WDSTestConfig",
    "WDSTrainingConfig",
    "build_shard_index",
    "create_wds_loader",
    "resolve_shards",
    "select_shards",
]

---
license: cc-by-nc-4.0
configs:
  - config_name: ghost
    data_files: "ghost/*.tar"
    default: true
  - config_name: mae
    data_files: "mae/*.tar"
tags:
  - full-waveform-lidar
  - lidar
  - point-cloud
  - webdataset
---

# Ghost-FWL

Full-waveform LiDAR voxel dataset for ghost point detection, distributed as
[WebDataset](https://github.com/webdataset/webdataset) shards.

<a href='https://keio-csg.github.io/Ghost-FWL/'><img
src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2603.28224'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://github.com/Keio-CSG/Ghost-FWL'><img src='https://img.shields.io/badge/Code-GitHub-black'></a>
</div>



## Citation
```bibtex
@inproceedings{ikeda2026ghostfwl,
  title = {Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal},
  author = {Ikeda, Kazuma and Hara, Ryosei and Nagata, Rokuto and Sako, Ozora and Ding, Zihao and Kado, Takahiro and Fujioka, Ibuki and Beppu, Taro and Isogawa, Mariko and Yoshioka, Kentaro},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2026},
}
```


## Configs

### `ghost` — annotated dataset for ghost detection

One sample per frame:

| member | content |
|---|---|
| `<key>.voxel.b2` | voxel grid, Blosc2-packed `(400, 512, 700)` array |
| `<key>.annotation.b2` | annotation voxel (`annotation_v1`), same shape |
| `<key>.annotation_expand.b2` | expanded annotation (`annotation_v1_expand`), when available |
| `<key>.json` | `frame_id`, `scene_id`, `hist_id`, `annotation_version`, `has_annotation_expand` |

### `mae` — dataset for MAE pretraining

One sample per frame:

| member | content |
|---|---|
| `<key>.voxel.b2` | voxel grid, Blosc2-packed `(400, 512, 700)` array |
| `<key>.peaks.npy` | per-pixel peak list, `np.load(..., allow_pickle=True)`, `(204800, 3)` |
| `<key>.json` | `frame_id`, `category` (`"ghost"` or `"normal"`), `session` |

## Loading

With the `webdataset` package (recommended for training):

```python
import io
import json

import blosc2
import numpy as np
import webdataset as wds
from huggingface_hub import HfApi, get_token

repo = "ryhara/Ghost-FWL"
files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".tar")]
urls = [f"https://huggingface.co/datasets/{repo}/resolve/main/{f}" for f in files if f.startswith("ghost/")]

dataset = wds.WebDataset(urls, shardshuffle=True).shuffle(100)
for sample in dataset:
    metadata = json.loads(sample["json"])
    voxel = blosc2.unpack_array2(sample["voxel.b2"])         # (400, 512, 700)
    annotation = blosc2.unpack_array2(sample["annotation.b2"])
    break
```

With `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("ryhara/Ghost-FWL", "mae", streaming=True)
```

Each `<config>/manifest.json` records the sample count, shard list, and any
files excluded during conversion. `<config>/shard_index.json` maps every shard to
its per-group sample counts (`scene001/hist003` for ghost, `ghost/<session>` for
mae) so loaders can open only the shards they need.

The [Ghost-FWL training code](https://github.com/Keio-CSG/Ghost-FWL) ships loaders
that stream these shards with the same preprocessing as the original directory
layout (`src/data/dataset_fwl_wds.py`, `src/data/dataset_fwl_mae_wds.py`); set
`wds_root: hf://ryhara/Ghost-FWL` in `configs/config_*_wds.yaml`.


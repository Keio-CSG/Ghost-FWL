# Hugging Face Dataset (Updated on 2026-08)

https://huggingface.co/datasets/ryhara/Ghost-FWL

The dataset is published as [WebDataset](https://github.com/webdataset/webdataset) shards.
The `*_wds.py` code paths read these shards directly (locally or from the Hub) with the
same preprocessing as the directory-based dataset described in
[README_dataset.md](README_dataset.md).

- requirements
    - `hf auth login` (the dataset is gated; request access on the Hub first)
    - `uv sync` (installs `webdataset`, `huggingface_hub`, `blosc2`)

## Dataset Structure

```
ryhara/Ghost-FWL
├── README.md                  # dataset card
├── ghost/                     # finetune / test (config: ghost, 262 shards, ~260 GB)
│   ├── ghost-000000.tar ...
│   ├── manifest.json          # sample count, shard list, excluded files
│   └── shard_index.json       # shard -> {group: sample count}
└── mae/                       # pretrain (config: mae, 467 shards)
    ├── mae-000000.tar ...
    ├── manifest.json
    └── shard_index.json
```

Each shard is a ~1 GB tar. Samples are stored as-is (Blosc2 `.b2`, no re-encoding):

```
ghost:  <key>.voxel.b2  <key>.annotation.b2  [<key>.annotation_expand.b2]  <key>.json
        key  = <scene_id>-<hist_id>-<frame_id>
        json = {frame_id, scene_id, hist_id, annotation_version, has_annotation_expand}

mae:    <key>.voxel.b2  <key>.peaks.npy  <key>.json
        key  = <category>-<frame_id>          (category: ghost | normal)
        json = {frame_id, category, session}
```

Only the `ghost` config is registered for `datasets` / the Dataset Viewer; `mae` contains
object-dtype `.npy` files that `datasets` cannot decode, so read it with `webdataset` or
the loaders below.

## Config

The wds variants live next to the original files and share the epoch loop, loss and collate:

| | directory version | WebDataset version |
| --- | --- | --- |
| YAML | `configs/config_{pretrain,train,test}.yaml` | `configs/config_{pretrain,train,test}_wds.yaml` |
| scripts | `scripts/run_train.py` / `run_test.py` | `scripts/run_train_wds.py` / `run_test_wds.py` |
| dataset | `src/data/dataset_fwl.py`, `dataset_fwl_mae.py` | `src/data/dataset_fwl_wds.py`, `dataset_fwl_mae_wds.py` |
| shard resolution / split | – | `src/data/wds_utils.py`, `wds_fetch.py` |

Keys specific to the wds YAMLs:

```yaml
wds_root: hf://ryhara/Ghost-FWL   # or a local directory containing ghost/ and mae/
wds_cache_dir: ''                 # hf:// only, see "Reading from the Hub"
wds_annotation_key: annotation_expand   # ghost: annotation_expand (default) | annotation
wds_shuffle_buffer: 16
wds_max_shards: 0                 # debug: read only the first N selected shards

train_wds_groups:                 # instead of directory lists, select data by group
- scene005/hist001                #   ghost: scene003, scene003/hist012, scene00[1-5]/hist*
- ghost/20251014142232_voxel_b2   #   mae:   ghost, normal/2025*, ghost/<session>
valid_wds_groups: []              # empty -> split train groups 8:2 by sample-key hash
```

`shard_index.json` lets the loader open only the shards that contain the selected groups
and know the dataset length in advance. Train/valid split and `divide` are deterministic
(hash of the sample key), so the same YAML always yields the same subset.

### Reading from the Hub

With `wds_root: hf://ryhara/Ghost-FWL`, `wds_cache_dir` controls how shards are fetched:

| `wds_cache_dir` | behaviour |
| --- | --- |
| `''` (empty) | stream over HTTP, resuming with `Range` after a network drop; no disk |
| a path | download each shard into that directory with `hf_hub_download` (resumable, sha256-verified); later epochs / runs read locally |

For anything beyond a smoke test, set `wds_cache_dir` to a large disk (the full `ghost`
config is ~260 GB) or copy the shards locally and point `wds_root` at the directory.

## Pretrain
```bash
uv run python scripts/run_train_wds.py --config configs/config_pretrain_wds.yaml
```

## Train
```bash
uv run python scripts/run_train_wds.py --config configs/config_train_wds.yaml
```

## Test
```bash
uv run python scripts/run_test_wds.py --config configs/config_test_wds.yaml
```

Set `checkpoint_path` in the YAML. Add `wds_max_shards: 1` for a quick end-to-end check.

## Reading shards without the training code

```python
import webdataset as wds
from src.utils import load_blosc2_bytes

urls = "https://huggingface.co/datasets/ryhara/Ghost-FWL/resolve/main/ghost/ghost-{000000..000261}.tar"
for sample in wds.WebDataset(urls, shardshuffle=False):
    voxel = load_blosc2_bytes(sample["voxel.b2"])              # (400, 512, 700)
    annotation = load_blosc2_bytes(sample["annotation_expand.b2"])
    break
```

For a gated repo pass the token, e.g. `pipe:curl -sfL -H "Authorization: Bearer $HF_TOKEN" <url>`,
or use `src.data.wds_utils.resolve_shards("hf://ryhara/Ghost-FWL", "ghost").urls`.

"""Upload WebDataset shards to the Hugging Face Hub as a public dataset.

Expects the output of hf_dataset/convert_to_webdataset.py:

    <root>/
    ├── ghost/ghost-*.tar + manifest.json
    └── mae/mae-*.tar + manifest.json

The dataset card (hf_dataset/hf_dataset_card.md) is copied to README.md in the repo,
defining the `mae` and `ghost` configs via its YAML header.

Requires `hf auth login` (or the HF_TOKEN environment variable) beforehand.

Usage:
    uv run python hf_dataset/upload_to_hf.py \
        --root /mnt/nas5/hara/Ghost-FWL-wds --repo-id ryhara/Ghost-FWL

Use --private for a dry-run-style private upload first, then make it public
from the repo settings (or re-run without --private).
Use --config mae (repeatable) to upload only some configs; re-running later with
another config adds it to the same repo.
"""

import argparse
import pathlib
import shutil
import sys

from huggingface_hub import HfApi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.log import log_info, log_warning  # noqa: E402

DATASET_CARD = pathlib.Path(__file__).resolve().parent / "hf_dataset_card.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", default="ryhara/Ghost-FWL")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument(
        "--config",
        choices=("ghost", "mae"),
        action="append",
        help="Upload only these configs (repeatable). Default: both.",
    )
    args = parser.parse_args()
    configs = tuple(args.config) if args.config else ("ghost", "mae")

    for config in configs:
        config_dir = args.root / config
        if not any(config_dir.glob(f"{config}-*.tar")):
            log_warning(f"{config_dir} has no shards — uploading anyway? Ctrl-C to abort")

    readme_dst = args.root / "README.md"
    if DATASET_CARD.exists():
        shutil.copyfile(DATASET_CARD, readme_dst)
        log_info(f"Copied dataset card to {readme_dst}")
    elif not readme_dst.exists():
        log_warning("No dataset card found; the repo will have no README/config metadata")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    allow_patterns = ["README.md", *(f"{c}/*" for c in configs)]
    log_info(f"Uploading {args.root} ({', '.join(configs)}) -> {args.repo_id}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.root),
        allow_patterns=allow_patterns,
    )
    log_info(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

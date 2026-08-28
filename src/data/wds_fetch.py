"""`pipe:` helpers that feed hf:// WebDataset shards to the wds pipeline.

Run as a standalone script (not `-m`) by src.data.wds_utils._hf_resolve so that the
`src.data` package (torch, datasets, ...) is not imported in every shard subprocess:

    python src/data/wds_fetch.py fetch  <repo_id> <path_in_repo> <cache_dir>
    python src/data/wds_fetch.py stream <repo_id> <path_in_repo>
"""

import shutil
import sys
from typing import Optional


def fetch_and_cat(repo_id: str, filename: str, cache_dir: str) -> None:
    """`pipe:` helper: download the shard into the HF cache (resumable) then cat it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=cache_dir)
    with open(path, "rb") as f:
        shutil.copyfileobj(f, sys.stdout.buffer, length=1 << 20)


def stream_and_cat(
    repo_id: str, filename: str, out=None, max_retries: int = 20, chunk: int = 1 << 20
) -> int:
    """`pipe:` helper: stream the shard over HTTP, resuming with Range after any drop.

    Returns the number of bytes written. Raises after `max_retries` consecutive failures
    without progress.
    """
    import time

    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    out = out or sys.stdout.buffer
    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    base_headers = build_hf_headers()
    pos, total, failures = 0, None, 0
    while total is None or pos < total:
        headers = dict(base_headers)
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                if pos and r.status_code != 206:
                    raise RuntimeError(f"{filename}: server ignored Range request")
                if total is None:
                    total = int(r.headers["Content-Length"])
                for data in r.iter_content(chunk):
                    out.write(data)
                    pos += len(data)
                    failures = 0
        except (requests.RequestException, RuntimeError) as exn:
            failures += 1
            if failures > max_retries:
                raise RuntimeError(f"{filename}: giving up at byte {pos}/{total}") from exn
            print(
                f"[wds stream] {filename}: {type(exn).__name__} at byte {pos}/{total}, "
                f"resuming (attempt {failures}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(min(2 * failures, 30))
    out.flush()
    return pos


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 4 and argv[0] == "fetch":
        fetch_and_cat(argv[1], argv[2], argv[3])
    elif len(argv) == 3 and argv[0] == "stream":
        stream_and_cat(argv[1], argv[2])
    else:
        sys.exit(
            f"usage: {sys.argv[0]} fetch <repo_id> <path_in_repo> <cache_dir>\n"
            f"       {sys.argv[0]} stream <repo_id> <path_in_repo>"
        )

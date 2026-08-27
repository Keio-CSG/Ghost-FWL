"""Blosc2 helper for the WebDataset conversion.

The decoder now lives in src/utils (shared with the src/data WebDataset loaders);
this module re-exports it so the conversion scripts keep working unchanged.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.utils.custom_blosc2 import load_blosc2_bytes  # noqa: E402

__all__ = ["load_blosc2_bytes"]

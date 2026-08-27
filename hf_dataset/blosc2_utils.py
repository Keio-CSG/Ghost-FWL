"""Blosc2 helpers for the WebDataset conversion, self-contained in hf_dataset/.

Kept separate from src/ so the dataset packaging work does not touch the
training code. When the WebDataset loading pipeline is integrated into
training later, this can move into src/utils.
"""

import blosc2
import numpy as np


def load_blosc2_bytes(data: bytes) -> np.ndarray:
    """Decode a Blosc2 payload (the raw content of a .b2 file) into an ndarray.

    In-memory counterpart of src.utils.load_blosc2(), e.g. for WebDataset tar members.
    """
    return blosc2.unpack_array2(data)

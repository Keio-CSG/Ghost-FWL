import pathlib
from typing import Union

import blosc2
import numpy as np


def load_npy_file(file_path: str) -> np.ndarray:
    """Load numpy file with pickle support"""
    data = np.load(file_path, allow_pickle=True)
    return data


def save_blosc2(path: Union[str, pathlib.Path], x: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(blosc2.pack_array2(x))


def load_blosc2(path: Union[str, pathlib.Path]) -> np.ndarray:
    return blosc2.load_array(path)

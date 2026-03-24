import logging
import sys
from typing import Union


def _create_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)  # ty: ignore[no-matching-overload]
    fmt = "[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _create_logger(__name__)


def set_level(level: Union[int, str]) -> None:
    if isinstance(level, str):
        level = level.upper()
        if level in logging._nameToLevel:
            level = logging._nameToLevel[level]
        else:
            raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(level)


def log_info(msg: str) -> None:
    logger.info(msg)


def log_warning(msg: str) -> None:
    logger.warning(msg)


def log_error(msg: str) -> None:
    logger.error(msg)


def log_critical(msg: str) -> None:
    logger.critical(msg)

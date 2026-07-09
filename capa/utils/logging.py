# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""Logging helpers for CAPA."""

import logging
import sys

try:
    import colorlog

    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

_NOISY_LOGGERS = [
    "PIL",
    "matplotlib",
    "urllib3",
    "dinov2",
]


def get_local_logger(
    name: str, level: int = logging.INFO, use_color: bool = True
) -> logging.Logger:
    """Return a logger for *name*, configuring the 'capa' namespace on first call.

    Usage (in any capa module)::

        from capa.utils.logging import get_local_logger
        logger = get_local_logger(__name__)

    Configures only the 'capa' logger namespace; the root logger is untouched.
    """
    capa_logger = logging.getLogger("capa")
    if not capa_logger.handlers:
        fmt = "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s >> %(message)s"
        datefmt = None

        if use_color and _HAS_COLORLOG:
            formatter = colorlog.ColoredFormatter(
                f"%(log_color)s{fmt}%(reset)s",
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        else:
            formatter = logging.Formatter(fmt, datefmt=datefmt)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        capa_logger.addHandler(handler)
        capa_logger.setLevel(level)

        for noisy in _NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(name)

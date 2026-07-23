"""Shared helpers for logging, seeding, and distributed coordination."""

import logging
import os
import random
from pathlib import Path

import numpy


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_state_dict(state):
    """Strip torch.compile and DistributedDataParallel prefixes from keys.

    A checkpoint saved from a wrapped model carries '_orig_mod.' (compile)
    or 'module.' (DDP) key prefixes, in either order, which a plain model
    cannot load strictly. Stripping at load time keeps every checkpoint
    readable regardless of how the training run wrapped its model.
    """
    prefixes = ('_orig_mod.', 'module.')
    stripped = True
    while stripped and state:
        stripped = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state):
                state = {
                    key[len(prefix):]: value for key, value in state.items()
                }
                stripped = True
    return state


def is_distributed():
    return int(os.environ.get('WORLD_SIZE', '1')) > 1


def get_rank():
    return int(os.environ.get('RANK', '0'))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', '1'))


def is_main_process():
    return get_rank() == 0

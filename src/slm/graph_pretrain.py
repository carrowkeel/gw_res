"""Graph stage four: pretrain the graph-context model from scratch.

Runs the same pretraining loop as the base stage on the graph-packed binaries,
writing checkpoints to a separate directory so the flat and graph models
coexist under one run. Multi-GPU works the same way:

    python -m slm.graph_pretrain --config configs/poc.yaml
    torchrun --nproc_per_node=4 -m slm.graph_pretrain --config configs/poc.yaml
"""

import argparse

from .config import load_config
from .pretrain import run as run_pretrain


def run(config):
    return run_pretrain(
        config,
        packed_directory=config.graph_packed_dir,
        checkpoint_root=config.graph_pretrain_dir,
    )


def main():
    parser = argparse.ArgumentParser(description='Pretrain graph-context model')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == '__main__':
    main()

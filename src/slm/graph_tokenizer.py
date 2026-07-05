"""Graph stage two: train the tokenizer used by the graph-context model.

Trains the same byte-level BPE on the same corpus as the base tokenizer stage,
with the structural marker tokens reserved so linearized graphs can be encoded
without the markers fragmenting into byte pieces. The artifact is written next
to the base tokenizer so the flat and graph models remain separately loadable.

    python -m slm.graph_tokenizer --config configs/poc.yaml
"""

import argparse

from .config import load_config
from .graph import STRUCTURE_TOKENS
from .tokenizer import train as train_base


def marker_ids(tokenizer):
    """Resolve the structural marker token ids from a trained tokenizer."""
    names = ['graph_open', 'graph_close', 'node_open', 'node_close', 'next']
    markers = {}
    for name, token in zip(names, STRUCTURE_TOKENS):
        token_id = tokenizer.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError('structure token %r missing from tokenizer' % token)
        markers[name] = token_id
    return markers


def train(config):
    """Train the graph tokenizer with the structural tokens reserved."""
    return train_base(
        config,
        extra_special_tokens=STRUCTURE_TOKENS,
        output_path=config.graph_tokenizer_path,
    )


def main():
    parser = argparse.ArgumentParser(description='Train graph BPE tokenizer')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    train(load_config(arguments.config))


if __name__ == '__main__':
    main()

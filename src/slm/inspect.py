"""Inspect a generated corpus before training on it.

Reports per-type document counts, generation yield against the configured
target, text-length statistics, and any kept text that still trips the
referent-free filter. Use this after the generate stage to decide whether the
corpus is clean enough to train on.

    python -m slm.inspect --config configs/pilot.yaml
"""

import argparse
import json
from collections import Counter

from . import filters
from .config import load_config
from .utils import get_logger

logger = get_logger('inspect')


def _load_pretrain(config):
    documents = []
    pretrain_directory = config.data_dir / 'pretrain'
    for shard in sorted(pretrain_directory.glob('shard_*.jsonl')):
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    documents.append(json.loads(stripped))
    return documents


def _mean_length(values):
    return sum(values) // len(values) if values else 0


def inspect(config):
    """Print a summary of the generated corpus and return the counts."""
    severity = config.generate.severity
    documents = _load_pretrain(config)
    type_counts = Counter(document.get('type', 'unknown') for document in documents)
    lengths = [len(document['text']) for document in documents]

    flagged = []
    for document in documents:
        reasons = filters.check_text(document['text'], severity)
        if reasons:
            flagged.append((document.get('type', 'unknown'), reasons, document['text']))

    target = config.generate.number_of_texts
    yield_percent = 100.0 * len(documents) / target if target else 0.0

    print('pretrain documents: %d (target %d, yield %.1f%%)'
          % (len(documents), target, yield_percent))
    print('per type:')
    for text_type in sorted(type_counts):
        print('  %-14s %d' % (text_type, type_counts[text_type]))
    if lengths:
        print('char length: min %d mean %d max %d'
              % (min(lengths), _mean_length(lengths), max(lengths)))
    print('filter-tripping kept documents: %d' % len(flagged))
    for text_type, reasons, text in flagged[:5]:
        print('  [%s] %s' % (text_type, ', '.join(reasons)))
        print('    %r' % text[:160])

    pairs_path = config.data_dir / 'sft' / 'sft.jsonl'
    if pairs_path.exists():
        pairs = []
        with open(pairs_path) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    pairs.append(json.loads(stripped))
        prompt_lengths = [len(pair['prompt']) for pair in pairs]
        response_lengths = [len(pair['response']) for pair in pairs]
        pair_target = config.generate.number_of_pairs
        pair_yield = 100.0 * len(pairs) / pair_target if pair_target else 0.0
        pair_flagged = sum(
            1 for pair in pairs
            if not (filters.passes(pair['prompt'], severity)
                    and filters.passes(pair['response'], severity))
        )
        print('sft pairs: %d (target %d, yield %.1f%%)'
              % (len(pairs), pair_target, pair_yield))
        print('  prompt chars mean %d, response chars mean %d'
              % (_mean_length(prompt_lengths), _mean_length(response_lengths)))
        print('  filter-tripping pairs: %d' % pair_flagged)

    return {'documents': len(documents), 'by_type': dict(type_counts),
            'flagged': len(flagged)}


def main():
    parser = argparse.ArgumentParser(description='Inspect a generated corpus')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    inspect(load_config(arguments.config))


if __name__ == '__main__':
    main()

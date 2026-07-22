"""Print raw completions from a trained checkpoint for quick inspection.

The evaluation report judges the finetuned model on chat prompts, which is the
harshest view of a small model: the finetuned model is weakly trained and the
probe questions are out of distribution. This command instead completes short
in-distribution seeds with the base pretrained model and no repetition penalty,
which is the fair measure of what pretraining actually learned.

Seeds are drawn from the run's own corpus (random document prefixes), so they
stay in distribution no matter how the generation recipe changes; an empty
seed is always included. The fixed fallback list is only used when no corpus
is present.

    python -m slm.sample --config runs/t1/config.resolved.yaml
    python -m slm.sample --config runs/world/pico/config.yaml --stage sft --penalty 1.3
"""

import argparse
import json
import random

from .config import load_config
from .infer import StudentModel
from .utils import get_logger

logger = get_logger('sample')

FALLBACK_SEEDS = [
    'The wood stands on the higher ground, and',
    'In one of the valleys there is',
    'The near bank had given way since',
    'Beyond the ridge the ground',
    'There is a pool at the foot of the slope, and',
    'The mist lay over the marsh until',
]


def _prefix(text, word_count):
    words = text.split()
    return ' '.join(words[:word_count])


def corpus_seeds(config, seed_count, word_count=8):
    """Sample document prefixes from the run's corpus as completion seeds."""
    directory = config.corpus_pretrain_dir
    shards = sorted(directory.glob('shard_*.jsonl')) if directory.exists() else []
    if not shards:
        return list(FALLBACK_SEEDS)
    random_generator = random.Random(config.project.seed)
    chosen = []
    seen = 0
    for shard in shards:
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                seen += 1
                if len(chosen) < seed_count:
                    chosen.append(record['text'])
                else:
                    slot = random_generator.randrange(seen)
                    if slot < seed_count:
                        chosen[slot] = record['text']
    return [_prefix(text, word_count) for text in chosen]


def run(config, stage, penalty, max_new_tokens, temperature, top_p, count,
        seed_count):
    checkpoint_base = config.sft_dir if stage == 'sft' else config.pretrain_dir
    checkpoint_path = checkpoint_base / 'ckpt_last.pt'
    if stage == 'pretrain':
        best = config.pretrain_dir / 'ckpt_best.pt'
        if best.exists():
            checkpoint_path = best
    logger.info('loading %s checkpoint %s', stage, checkpoint_path)
    student = StudentModel(config, checkpoint_path)
    for seed in corpus_seeds(config, seed_count) + ['']:
        print('=' * 70)
        print('seed: %r' % seed)
        for _ in range(count):
            completion = student.complete(
                seed, max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=penalty,
            )
            print('  ->', repr((seed + completion).strip()))


def main():
    parser = argparse.ArgumentParser(description='Sample completions')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', default='pretrain', choices=['pretrain', 'sft'])
    parser.add_argument('--penalty', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=120)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--seed-count', type=int, default=6)
    arguments = parser.parse_args()
    run(load_config(arguments.config), arguments.stage, arguments.penalty,
        arguments.max_new_tokens, arguments.temperature, arguments.top_p,
        arguments.count, arguments.seed_count)


if __name__ == '__main__':
    main()

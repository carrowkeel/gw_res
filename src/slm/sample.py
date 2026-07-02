"""Print raw completions from a trained checkpoint for quick inspection.

The evaluation report judges the finetuned model on chat prompts, which is the
harshest view of a small model: the finetuned model is weakly trained and the
probe questions are out of distribution. This command instead completes short
in-distribution seeds with the base pretrained model and no repetition penalty,
which is the fair measure of what pretraining actually learned.

    python -m slm.sample --config configs/scale/s1_nano.yaml
    python -m slm.sample --config configs/scale/s1_nano.yaml --stage sft --penalty 1.3
"""

import argparse

from .config import load_config
from .infer import StudentModel
from .utils import get_logger

logger = get_logger('sample')

DEFAULT_SEEDS = [
    'The wood stands on the higher ground, and',
    'In one of the valleys there is',
    'The near bank had given way since',
    'Beyond the ridge the ground',
    'There is a pool at the foot of the slope, and',
    'The mist lay over the marsh until',
    '',
]


def run(config, stage, penalty, max_new_tokens, temperature, top_p, count):
    checkpoint_base = config.sft_dir if stage == 'sft' else config.pretrain_dir
    checkpoint_path = checkpoint_base / 'ckpt_last.pt'
    if stage == 'pretrain':
        best = config.pretrain_dir / 'ckpt_best.pt'
        if best.exists():
            checkpoint_path = best
    logger.info('loading %s checkpoint %s', stage, checkpoint_path)
    student = StudentModel(config, checkpoint_path)
    for seed in DEFAULT_SEEDS:
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
    arguments = parser.parse_args()
    run(load_config(arguments.config), arguments.stage, arguments.penalty,
        arguments.max_new_tokens, arguments.temperature, arguments.top_p,
        arguments.count)


if __name__ == '__main__':
    main()

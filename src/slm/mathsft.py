"""Arithmetic SFT: teach basic arithmetic in the corpus's dialogue format.

The second bridging stage between language pretraining and the simulator.
The sim demands integer addition, subtraction, and comparison over prices,
cash, and share counts; this stage teaches those operations in the stage-1
name-and-colon dialogue format, with speakers and subjects deliberately
distinct from the sim's labels. Unlike every other generation stage the
data is generated programmatically: the program authors the numbers and
derives the answers, so ground truth is exact and supply is unlimited, with
template pools supplying surface variety. Training and evaluation run
through the same sftstage and sfteval machinery as the communication SFT,
starting from the communication checkpoint, and the eval report breaks the
answered rate down by operation kind so each skill is tracked on its own.

    python -m slm.mathsft --config configs/t1_full.yaml
"""

import argparse
import json
import random

import numpy

from . import seeds, sfteval, sftstage
from .config import load_config
from .generate import _normalized_hash
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('mathsft')

ITEMS = [
    'crates', 'sacks', 'tickets', 'chairs', 'lamps', 'bricks', 'jars',
    'barrels', 'planks', 'baskets', 'bolts of cloth', 'reams of paper',
]

ADJECTIVE_PAIRS = [
    ('oak', 'pine'), ('iron', 'tin'), ('glazed', 'plain'),
    ('northern', 'southern'), ('large', 'small'),
]

FILLERS = [
    'That went faster than last week.',
    'The ledger should say the same.',
    'Good, the buyer arrives tomorrow.',
    'I thought it would be more.',
    'The tally took most of the morning.',
]

ADDITION_SCENARIOS = [
    ('We counted %d %s in the front room.',
     'There are another %d in the back.',
     'How many %s are there in all?',
     'There are %d %s in all.'),
    ('The first cart carried %d %s.',
     'The second cart brought %d more.',
     'How many %s did the two carts bring together?',
     'Together the carts brought %d %s.'),
    ('I sold %d %s before noon.',
     'And %d more after noon.',
     'How many %s did you sell today?',
     'I sold %d %s today.'),
]

SUBTRACTION_SCENARIOS = [
    ('We started the week with %d %s.',
     'We have shipped %d of them since.',
     'How many %s are left?',
     'There are %d %s left.'),
    ('The store had %d %s on Monday.',
     'By Friday %d of them had been sold.',
     'How many %s remain?',
     'There are %d %s remaining.'),
]

CHANGE_SCENARIOS = [
    ('The price of a %s stood at %d last month.',
     'It stands at %d now.',
     'By how much did the price of a %s change?',
     'It changed by %d.'),
]

DIFFERENCE_SCENARIOS = [
    ('The north yard holds %d %s.',
     'The south yard holds %d.',
     'How many more %s does the north yard hold?',
     'The north yard holds %d more %s.'),
    ('Our stall took %d orders for %s this week.',
     'The stall across the way took %d.',
     'How many more orders for %s did our stall take?',
     'Our stall took %d more orders for %s.'),
]

COMPARISON_SCENARIOS = [
    ('The %s %s cost %d each.',
     'The %s %s cost %d each.',
     'Which %s cost less?',
     'The %s %s, at %d against %d.'),
]


def _bands(maximum_value):
    bands = [(2, 20)]
    if maximum_value > 20:
        bands.append((21, min(200, maximum_value)))
    if maximum_value > 200:
        bands.append((201, maximum_value))
    return bands


def _pair(random_generator, maximum_value, distinct=False):
    low, high = random_generator.choice(_bands(maximum_value))
    first = random_generator.randint(low, high)
    second = random_generator.randint(low, high)
    while distinct and second == first:
        second = random_generator.randint(low, high)
    return first, second


def _addition(random_generator, maximum_value):
    first, second = _pair(random_generator, maximum_value)
    item = random_generator.choice(ITEMS)
    statement_one, statement_two, question, answer = (
        random_generator.choice(ADDITION_SCENARIOS)
    )
    return (
        statement_one % (first, item),
        statement_two % second,
        question % item,
        answer % (first + second, item),
    )


def _subtraction(random_generator, maximum_value):
    first, second = _pair(random_generator, maximum_value, distinct=True)
    if second > first:
        first, second = second, first
    item = random_generator.choice(ITEMS)
    statement_one, statement_two, question, answer = (
        random_generator.choice(SUBTRACTION_SCENARIOS)
    )
    return (
        statement_one % (first, item),
        statement_two % second,
        question % item,
        answer % (first - second, item),
    )


def _change(random_generator, maximum_value):
    first, second = _pair(random_generator, maximum_value, distinct=True)
    item = random_generator.choice(ITEMS).rstrip('s')
    statement_one, statement_two, question, answer = CHANGE_SCENARIOS[0]
    return (
        statement_one % (item, first),
        statement_two % second,
        question % item,
        answer % abs(first - second),
    )


def _difference(random_generator, maximum_value):
    first, second = _pair(random_generator, maximum_value, distinct=True)
    if second > first:
        first, second = second, first
    item = random_generator.choice(ITEMS)
    statement_one, statement_two, question, answer = (
        random_generator.choice(DIFFERENCE_SCENARIOS)
    )
    return (
        statement_one % (first, item),
        statement_two % second,
        question % item,
        answer % (first - second, item),
    )


def _comparison(random_generator, maximum_value):
    first, second = _pair(random_generator, maximum_value, distinct=True)
    item = random_generator.choice(ITEMS)
    adjective_a, adjective_b = random_generator.choice(ADJECTIVE_PAIRS)
    statement_one, statement_two, question, answer = COMPARISON_SCENARIOS[0]
    cheaper_adjective = adjective_a if first < second else adjective_b
    cheaper, dearer = min(first, second), max(first, second)
    return (
        statement_one % (adjective_a, item, first),
        statement_two % (adjective_b, item, second),
        question % item,
        answer % (cheaper_adjective, item, cheaper, dearer),
    )


_KIND_BUILDERS = {
    'addition': _addition,
    'subtraction': _subtraction,
    'change': _change,
    'difference': _difference,
    'comparison': _comparison,
}


def build_dialogue(random_generator, maximum_value):
    """Author one arithmetic dialogue with a program-derived answer."""
    kind = random_generator.choice(sorted(_KIND_BUILDERS))
    statement_one, statement_two, question, answer = (
        _KIND_BUILDERS[kind](random_generator, maximum_value)
    )
    speaker_a = seeds.invented_name(random_generator)
    speaker_b = seeds.invented_name(random_generator)
    while speaker_b == speaker_a:
        speaker_b = seeds.invented_name(random_generator)
    second_speaker = (
        speaker_a if random_generator.random() < 0.5 else speaker_b
    )
    lines = [
        '%s: %s' % (speaker_a, statement_one),
        '%s: %s' % (second_speaker, statement_two),
    ]
    if random_generator.random() < 0.5:
        lines.append(
            '%s: %s' % (speaker_b, random_generator.choice(FILLERS))
        )
    lines.append('%s: %s' % (speaker_b, question))
    return {
        'prompt': '\n'.join(lines) + '\n%s:' % speaker_a,
        'response': answer,
        'question': question,
        'kind': kind,
    }


def generate_records(config):
    """Author the arithmetic dialogue pool, deduplicated, written durably."""
    mathsft_config = config.mathsft
    target = mathsft_config.number_of_dialogues
    output_path = config.mathsft_data_dir / 'mathsft.jsonl'
    ensure_directory(output_path.parent)
    random_generator = random.Random(config.project.seed + 47)
    records = []
    seen = set()
    attempts = 0
    while len(records) < target and attempts < target * 20:
        attempts += 1
        record = build_dialogue(
            random_generator, mathsft_config.maximum_value
        )
        fingerprint = _normalized_hash(record['prompt'] + record['response'])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append(record)
    with open(output_path, 'w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info('authored %d arithmetic dialogues -> %s',
                len(records), output_path)
    return records


def split_records(config, records):
    random_generator = numpy.random.default_rng(config.project.seed + 17)
    order = random_generator.permutation(len(records)).tolist()
    holdout_size = max(1, int(len(records) * config.mathsft.holdout_fraction))
    holdout = [records[index] for index in order[:holdout_size]]
    train = [records[index] for index in order[holdout_size:]]
    for name, subset in (('train', train), ('holdout', holdout)):
        path = config.mathsft_data_dir / ('%s.jsonl' % name)
        with open(path, 'w') as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info('split: %d train, %d holdout', len(train), len(holdout))
    return train, holdout


def run(config):
    set_seed(config.project.seed)
    records = generate_records(config)
    train, holdout = split_records(config, records)

    source_checkpoint = sftstage.resolve_checkpoint(config.commsft_dir)
    if source_checkpoint is None:
        source_checkpoint = sftstage.resolve_checkpoint(config.pretrain_dir)
        if source_checkpoint is None:
            raise SystemExit(
                'no commsft or pretrain checkpoint to start from; run the '
                'earlier stages first'
            )
        logger.warning(
            'no commsft checkpoint, starting from pretrain: %s',
            source_checkpoint,
        )
    checkpoint_directory = ensure_directory(config.mathsft_dir)

    baseline = sfteval.evaluate_checkpoint(
        config, source_checkpoint, holdout,
        checkpoint_directory / 'eval_baseline.json', 'mathsft-baseline',
    )

    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    best_checkpoint = sftstage.train_stage(
        config, config.mathsft, tokenizer, train, holdout,
        source_checkpoint, checkpoint_directory, 'mathsft',
    )

    report = sfteval.evaluate_checkpoint(
        config, best_checkpoint, holdout,
        checkpoint_directory / 'eval_report.json', 'mathsft',
    )
    logger.info(
        'arithmetic SFT: answered rate %.3f -> %.3f (baseline -> trained)',
        baseline['answered_rate'], report['answered_rate'],
    )
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Arithmetic SFT: programmatic generation, training, '
                    'evaluation'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    arguments = parser.parse_args()
    run(load_config(arguments.config, run_id=arguments.run_id))


if __name__ == '__main__':
    main()

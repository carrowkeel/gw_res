"""Communication SFT: teach the pretrained model to answer questions.

The first of the two bridging stages between language pretraining and the
simulator. The base model continues dialogue as chitchat; the simulator
needs a model that responds to what was asked. This stage generates its own
data inside the stage job (never into the stage-1 corpus): dialogues in the
corpus's name-and-colon turn format whose last turn answers a direct
question from the conversation, trained with loss on the answer turn only
and stage-1 replay against forgetting. The held-out dialogues are scored by
answer similarity through sfteval, against both the source checkpoint (the
baseline) and the trained one, so the stage's improvement is tracked
explicitly.

    python -m slm.commsft --config configs/t1_full.yaml
"""

import argparse
import gc
import json
import random

import numpy

from . import filters, prompts, sfteval, sftstage
from .config import load_config
from .generate import _chat, _load_engine, _normalized_hash
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('commsft')


def _records_path(config):
    return config.commsft_data_dir / 'commsft.jsonl'


def _scan_records(path):
    records = []
    seen = set()
    if not path.exists():
        return records, seen
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            records.append(record)
            seen.add(_normalized_hash(record['prompt'] + record['response']))
    return records, seen


def generate_dialogues(config):
    """Generate question-answering dialogues up to the configured count.

    Resumable: counts the records already written and generates only the
    shortfall, deduplicating against them. Returns the record list.
    """
    commsft_config = config.commsft
    generate_config = config.generate
    target = commsft_config.number_of_dialogues
    output_path = _records_path(config)
    records, seen = _scan_records(output_path)
    if len(records) >= target:
        logger.info('dialogue pool already complete (%d)', len(records))
        return records

    ensure_directory(output_path.parent)
    random_generator = random.Random(config.project.seed + 31)
    engine, sampling = _load_engine(
        generate_config.default_model, generate_config
    )
    system_prompt = prompts.build_system_prompt()
    example_turns = prompts.qa_dialogue_example_turns()
    kept = len(records)
    attempts = 0
    maximum_attempts = (target - kept) * 4 + generate_config.batch_size
    with open(output_path, 'a') as handle:
        while kept < target and attempts < maximum_attempts:
            size = min(generate_config.batch_size, (target - kept) * 2 + 1)
            user_prompts = [
                prompts.build_qa_dialogue_prompt(random_generator)
                for _ in range(size)
            ]
            texts = _chat(
                engine, sampling, system_prompt, user_prompts, example_turns
            )
            attempts += size
            for text in texts:
                if kept >= target:
                    break
                record = prompts.parse_qa_dialogue(
                    text, commsft_config.minimum_turns
                )
                if record is None:
                    continue
                if generate_config.apply_filter and not filters.passes(text):
                    continue
                fingerprint = _normalized_hash(
                    record['prompt'] + record['response']
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                handle.write(
                    json.dumps(record, ensure_ascii=False) + '\n'
                )
                records.append(record)
                kept += 1
            handle.flush()
            logger.info('dialogues: kept %d / %d', kept, target)
    del engine
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    if kept < target:
        raise SystemExit(
            'dialogue generation fell short (%d/%d); rerun the stage to '
            'continue from the existing pool' % (kept, target)
        )
    return records


def split_records(config, records):
    """Deterministically split the pool into train and holdout files."""
    random_generator = numpy.random.default_rng(config.project.seed + 13)
    order = random_generator.permutation(len(records)).tolist()
    holdout_size = max(1, int(len(records) * config.commsft.holdout_fraction))
    holdout = [records[index] for index in order[:holdout_size]]
    train = [records[index] for index in order[holdout_size:]]
    for name, subset in (('train', train), ('holdout', holdout)):
        path = config.commsft_data_dir / ('%s.jsonl' % name)
        with open(path, 'w') as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info('split: %d train, %d holdout', len(train), len(holdout))
    return train, holdout


def run(config):
    set_seed(config.project.seed)
    records = generate_dialogues(config)
    train, holdout = split_records(config, records)

    source_checkpoint = sftstage.resolve_checkpoint(config.pretrain_dir)
    if source_checkpoint is None:
        raise SystemExit(
            'no pretrain checkpoint in %s; run pretrain first'
            % config.pretrain_dir
        )
    checkpoint_directory = ensure_directory(config.commsft_dir)

    baseline = sfteval.evaluate_checkpoint(
        config, source_checkpoint, holdout,
        checkpoint_directory / 'eval_baseline.json', 'commsft-baseline',
    )

    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    best_checkpoint = sftstage.train_stage(
        config, config.commsft, tokenizer, train, holdout,
        source_checkpoint, checkpoint_directory, 'commsft',
    )

    report = sfteval.evaluate_checkpoint(
        config, best_checkpoint, holdout,
        checkpoint_directory / 'eval_report.json', 'commsft',
    )
    logger.info(
        'communication SFT: answered rate %.3f -> %.3f (baseline -> trained)',
        baseline['answered_rate'], report['answered_rate'],
    )
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Communication SFT: generation, training, evaluation'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    arguments = parser.parse_args()
    run(load_config(arguments.config, run_id=arguments.run_id))


if __name__ == '__main__':
    main()

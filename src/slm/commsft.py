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

NUMBER_WORDS = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
    'dozen', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
    'seventh', 'eighth', 'ninth', 'tenth',
}


_UNICODE_PUNCTUATION = str.maketrans({
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2014': '-', '\u2013': '-', '\u2026': '...', '\u00a0': ' ',
})


def _numeric(token):
    return any(character.isdigit() for character in token) \
        or token in NUMBER_WORDS


def grounded(record, minimum_overlap):
    """Reject answers that introduce content absent from the dialogue.

    The generator, asked to answer from the conversation, still fabricates
    plausible particulars (a novel place name, a date like April 18th) in
    a substantial share of dialogues, and training on those teaches
    invention instead of retrieval. Fabrication is objectively detectable:
    every numeric token in the answer (anything containing a digit, or a
    number word) must appear in the dialogue, and the answer's content
    words must mostly come from it.
    """
    prompt_tokens = set(sfteval.normalize_tokens(record['prompt']))
    content = [
        token for token in sfteval.normalize_tokens(record['response'])
        if len(token) > 3 or _numeric(token)
    ]
    if not content:
        return False
    for token in content:
        if _numeric(token) and token not in prompt_tokens:
            return False
    matched = sum(token in prompt_tokens for token in content)
    return matched / len(content) >= minimum_overlap


def copies_earlier_turn(record):
    """True when the answer is a verbatim copy of a whole earlier turn.

    An answer that repeats an entire prior turn is maximally grounded and
    usually wrong: it is the generator dodging the question with a line
    that was already said, the degenerate case the overlap measure alone
    rewards.
    """
    response_normalized = ' '.join(
        sfteval.normalize_tokens(record['response'])
    )
    for line in record['prompt'].split('\n')[:-1]:
        _, _, turn_text = line.partition(':')
        if turn_text.strip() and ' '.join(
                sfteval.normalize_tokens(turn_text)) == response_normalized:
            return True
    return False


def acceptable_text(text):
    """True when the text is ASCII after mapping typographic punctuation.

    Curly quotes and dashes are legitimate generator output; anything
    beyond them (stray CJK, mojibake) is contamination this stage's data
    should not carry.
    """
    return text.translate(_UNICODE_PUNCTUATION).isascii()


def usable(record, minimum_overlap):
    return (
        acceptable_text(record['prompt'] + record['response'])
        and grounded(record, minimum_overlap)
        and not copies_earlier_turn(record)
    )


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
    scanned = len(records)
    records = [
        record for record in records
        if usable(record, commsft_config.minimum_grounding)
    ]
    if scanned > len(records):
        logger.info(
            'dropped %d unusable dialogues of %d on disk; topping up',
            scanned - len(records), scanned,
        )
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
    maximum_attempts = (target - kept) * 8 + generate_config.batch_size
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
                if not usable(record, commsft_config.minimum_grounding):
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

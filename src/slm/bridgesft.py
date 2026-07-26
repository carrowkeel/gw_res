"""Bridging SFT: one diversified stage for communication, arithmetic, and judgment.

Replaces the separate communication and arithmetic SFT stages, whose
per-stage generators each taught a narrow schema: the model learned the
trigger (cart dialogues, friendly chats) as faithfully as the skill, and
outside the trigger the skill never fired. Here one corpus mixes three
record kinds under program-owned diversity axes (register, naming style,
tone, arrangement), sampled independently per record: the program owns the
truth (operands, results, facts, structure) and the seeds, and the LLM
contributes language only.

Record kinds:
- qa: a seeded conversation plus an extracted question and answer.
- arithmetic (addition, subtraction, change, difference, comparison): the
  program draws operands and derives the exact answer; the LLM re-voices
  the facts, question, and answer in the seeded register and may not alter
  a number.
- decision: an interaction whose situation demands a decision, cue either
  asked (the last turn asks what to do) or unasked (the last turn only
  states the urgency), so the response is a commitment either way.

A probe pool generated only from held-out registers measures out-of-schema
transfer; every record carries its axis values so the eval stratifies
accuracy by axis and trigger-narrowness shows up as a table, not an
anecdote.

    python -m slm.bridgesft --config configs/t1_full.yaml
"""

import argparse
import gc
import json
import random

import numpy

from . import filters, mathsft, prompts, seeds, sfteval, sftstage
from .commsft import acceptable_text, numbers_grounded
from .config import load_config
from .generate import _chat, _load_engine, _normalized_hash
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('bridgesft')

RECIPE = 2

AXIS_FIELDS = ['register', 'naming', 'tone', 'arrangement', 'cue', 'form']

META_ANSWER_WORDS = {
    'conversation', 'mentioned', 'specified', 'stated', 'dialogue',
    'transcript', 'context',
}


def sample_axes(random_generator, heldout=False):
    registers = (
        seeds.HELDOUT_REGISTERS if heldout else seeds.INTERACTION_REGISTERS
    )
    return {
        'register': random_generator.choice(registers),
        'naming': random_generator.choice(seeds.NAMING_STYLES),
        'tone': random_generator.choice(seeds.TONES),
        'heldout': heldout,
    }


def draw_kind(random_generator, bridgesft_config):
    roll = random_generator.random()
    if roll < bridgesft_config.qa_fraction:
        return 'qa'
    if roll < bridgesft_config.qa_fraction + bridgesft_config.decision_fraction:
        return 'decision'
    return random_generator.choice(sorted(mathsft._KIND_BUILDERS))


def build_task(random_generator, bridgesft_config, heldout=False):
    """Sample one record's seeds and program-owned content.

    Arithmetic tasks stay short by design (a single rendered remark in the
    merged arrangement, a 2 to 4 turn exchange in the split one): the
    numeric relation should sit next to the question, not be buried in
    conversation, and a small unambiguous request keeps the bulk of the
    generation distribution valid instead of filtering toward outliers.
    """
    kind = draw_kind(random_generator, bridgesft_config)
    axes = sample_axes(random_generator, heldout)
    participants = 2 if random_generator.random() < 0.5 else 3
    names = seeds.sample_speakers(
        axes['naming'], participants, random_generator
    )
    task = {'kind': kind, 'axes': axes, 'names': names}
    if kind == 'qa':
        turn_range = random_generator.choice([(3, 5), (4, 6), (5, 7)])
        task['minimum_turns'] = 3
        task['request'] = prompts.build_seeded_conversation_prompt(
            axes, names, random_generator.choice(seeds.SUBJECT_DOMAINS),
            turn_range, random_generator,
        )
    elif kind == 'decision':
        axes['cue'] = (
            'asked' if random_generator.random() < 0.5 else 'unasked'
        )
        axes['form'] = (
            'exchange' if random_generator.random() < 0.5
            else 'conversation'
        )
        task['minimum_turns'] = 2
        task['request'] = prompts.build_decision_interaction_prompt(
            axes, names, random_generator.choice(seeds.DECISION_SITUATIONS),
            random_generator,
        )
    else:
        axes['arrangement'] = (
            'merged' if random_generator.random() < 0.5 else 'split'
        )
        statement_one, statement_two, question, answer = (
            mathsft._KIND_BUILDERS[kind](
                random_generator, bridgesft_config.maximum_value
            )
        )
        question_speaker = names[-1]
        task['question'] = question
        task['reference_answer'] = answer
        task['operand_values'] = sfteval.numeric_values(
            statement_one + ' ' + statement_two
        )
        task['question_speaker'] = question_speaker
        task['minimum_turns'] = 2
        if axes['arrangement'] == 'merged':
            task['request'] = prompts.build_arithmetic_utterance_prompt(
                axes, [statement_one, statement_two], question,
            )
        else:
            task['request'] = prompts.build_arithmetic_exchange_prompt(
                axes, names, [statement_one, statement_two], question,
                question_speaker,
            )
    return task


def _joined(turns):
    return '\n'.join('%s: %s' % turn for turn in turns)


def _contains_operands(text, operand_values):
    text_values = sfteval.numeric_values(text)
    return all(value in text_values for value in operand_values)


def parse_conversation(task, text, minimum_turns, maximum_turns):
    """Validate a first-request completion against its task; None to reject.

    In the merged arithmetic arrangement the completion is one rendered
    remark, not turns: the program attaches the speaker label itself, so
    the merged shape is true by construction. Otherwise the turns must
    parse, carry at least two of the requested labels and no others, and
    satisfy the task's structural demand - a split arithmetic exchange
    must contain every operand, spread beyond the question turn (all
    operands inside the final turn is the merged shape wearing a split
    label), and end with the question speaker asking; a decision
    interaction must end with a question only under the asked cue.
    """
    kind = task['kind']
    if kind not in ('qa', 'decision') \
            and task['axes']['arrangement'] == 'merged':
        remark = text.strip().split('\n')[0].strip()
        if not remark or len(remark) > 400:
            return None
        if not acceptable_text(remark):
            return None
        if '?' not in remark:
            return None
        if not _contains_operands(remark, task['operand_values']):
            return None
        if prompts.parse_turns(remark) is not None:
            return None
        return [(task['question_speaker'], remark)]
    minimum_turns = task.get('minimum_turns', minimum_turns)
    turns = prompts.parse_turns(text)
    if turns is None:
        lines = text.strip().split('\n')[:-1]
        turns = prompts.parse_turns('\n'.join(lines))
    if turns is None or len(turns) < minimum_turns:
        return None
    turns = turns[:maximum_turns]
    allowed = set(task['names'])
    speakers = {speaker for speaker, _ in turns}
    if not speakers <= allowed or len(speakers) < 2:
        return None
    joined = _joined(turns)
    if not acceptable_text(joined):
        return None
    if kind == 'qa':
        return turns
    last_speaker, last_text = turns[-1]
    if kind == 'decision':
        has_question = '?' in last_text
        if task['axes']['cue'] == 'asked' and not has_question:
            return None
        if task['axes']['cue'] == 'unasked' and has_question:
            return None
        return turns
    if not _contains_operands(joined, task['operand_values']):
        return None
    if '?' not in last_text:
        return None
    if last_speaker != task['question_speaker']:
        return None
    if _contains_operands(last_text, task['operand_values']):
        return None
    return turns


def build_followup(task, turns, random_generator):
    """Return (responder, second-request prompt) for a parsed conversation."""
    last_speaker = turns[-1][0]
    others = [name for name in task['names'] if name != last_speaker]
    responder = random_generator.choice(others)
    conversation = _joined(turns)
    kind = task['kind']
    if kind == 'qa':
        return responder, prompts.build_qa_extraction_prompt(conversation)
    if kind == 'decision':
        return responder, prompts.build_decision_response_prompt(
            conversation, responder,
        )
    return responder, prompts.build_answer_render_prompt(
        conversation, responder, task['question'], task['reference_answer'],
    )


def assemble(task, turns, responder, followup_text, random_generator):
    """Build the final record from the second request; None to reject."""
    kind = task['kind']
    if kind == 'qa':
        pair = prompts.split_question_answer(followup_text)
        if pair is None:
            return None
        answer_tokens = set(sfteval.normalize_tokens(pair[1]))
        if answer_tokens & META_ANSWER_WORDS:
            return None
        question_normalized = ' '.join(sfteval.normalize_tokens(pair[0]))
        spoken = ' '.join(
            sfteval.normalize_tokens(_joined(turns))
        )
        if question_normalized and question_normalized in spoken:
            return None
        from .commsft import assemble_record

        record = assemble_record(turns, pair[0], pair[1], random_generator)
        if record is None or not numbers_grounded(record):
            return None
    else:
        response = followup_text.strip().split('\n')[0].strip()
        if response.startswith('%s:' % responder):
            response = response[len(responder) + 1:].strip()
        if not response or len(response) > 300:
            return None
        record = {
            'prompt': '%s\n%s:' % (_joined(turns), responder),
            'response': response,
            'question': task.get('question'),
        }
        if kind == 'decision':
            if response.rstrip().endswith('?'):
                return None
            if not numbers_grounded(record):
                return None
        else:
            if not sfteval.numbers_correct(
                    response, task['reference_answer']):
                return None
    if not acceptable_text(record['prompt'] + record['response']):
        return None
    record['kind'] = kind
    record['recipe'] = RECIPE
    record['heldout'] = task['axes']['heldout']
    for field in AXIS_FIELDS:
        if field in task['axes']:
            record[field] = task['axes'][field]
    return record


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
            if record.get('recipe') != RECIPE:
                continue
            records.append(record)
            seen.add(_normalized_hash(record['prompt'] + record['response']))
    return records, seen


def _generate_pool(config, engine, conversation_sampling, followup_sampling,
                   random_generator, target, output_path, heldout):
    """Fill one record pool up to target; resumable from what is on disk."""
    bridgesft_config = config.bridgesft
    generate_config = config.generate
    records, seen = _scan_records(output_path)
    if len(records) >= target:
        logger.info('%s pool already complete (%d)', output_path.stem,
                    len(records))
        return records
    system_prompt = prompts.build_system_prompt()
    kept = len(records)
    attempts = 0
    maximum_attempts = (target - kept) * 4 + generate_config.batch_size
    with open(output_path, 'a') as handle:
        while kept < target and attempts < maximum_attempts:
            size = min(generate_config.batch_size, (target - kept) * 2 + 1)
            tasks = [
                build_task(random_generator, bridgesft_config, heldout)
                for _ in range(size)
            ]
            conversations = _chat(
                engine, conversation_sampling, system_prompt,
                [task['request'] for task in tasks],
            )
            attempts += size
            parsed = []
            for task, text in zip(tasks, conversations):
                turns = parse_conversation(
                    task, text, bridgesft_config.minimum_turns,
                    bridgesft_config.maximum_turns,
                )
                if turns is None:
                    continue
                if generate_config.apply_filter \
                        and task['kind'] == 'qa' \
                        and not filters.passes(_joined(turns)):
                    continue
                parsed.append((task, turns))
            if not parsed:
                logger.info('%s: kept %d / %d', output_path.stem, kept,
                            target)
                continue
            followups = []
            for task, turns in parsed:
                responder, followup_prompt = build_followup(
                    task, turns, random_generator
                )
                followups.append((task, turns, responder, followup_prompt))
            followup_texts = _chat(
                engine, followup_sampling, system_prompt,
                [item[3] for item in followups],
            )
            for (task, turns, responder, _), followup_text in zip(
                    followups, followup_texts):
                if kept >= target:
                    break
                record = assemble(
                    task, turns, responder, followup_text, random_generator
                )
                if record is None:
                    continue
                fingerprint = _normalized_hash(
                    record['prompt'] + record['response']
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
                kept += 1
            handle.flush()
            logger.info('%s: kept %d / %d', output_path.stem, kept, target)
    if kept < target:
        raise SystemExit(
            '%s generation fell short (%d/%d); rerun the stage to continue '
            'from the existing pool' % (output_path.stem, kept, target)
        )
    return records


def _shard_target(total, worker_count, worker_index):
    """Return this worker's share of a pool, distributing the remainder."""
    base = total // worker_count
    return base + (1 if worker_index < total % worker_count else 0)


def _pool_path(config, worker_index):
    if worker_index is None:
        return config.bridgesft_data_dir / 'bridgesft.jsonl'
    return config.bridgesft_data_dir / (
        'bridgesft_worker_%03d.jsonl' % worker_index
    )


def _release_engine(engine):
    del engine
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _engine_and_sampling(config):
    generate_config = config.generate
    engine, _ = _load_engine(generate_config.default_model, generate_config)
    from vllm import SamplingParams

    conversation_sampling = SamplingParams(
        temperature=generate_config.temperature,
        top_p=generate_config.top_p,
        frequency_penalty=generate_config.frequency_penalty,
        presence_penalty=generate_config.presence_penalty,
        max_tokens=config.bridgesft.conversation_max_tokens,
    )
    followup_sampling = SamplingParams(
        temperature=0.3, top_p=0.9, max_tokens=120,
    )
    return engine, conversation_sampling, followup_sampling


def generate_records(config, worker_count=1, worker_index=None):
    """Generate record pools; shardable across independent workers.

    With worker_index set, this process generates only its share of the
    main pool (worker zero also owns the probe pool) into its own file and
    returns None; the dedicated training job merges the shards afterward.
    Each worker draws from its own seed stream, and the merge deduplicates
    across shards, so workers never coordinate and any of them can be
    requeued or rerun independently.
    """
    bridgesft_config = config.bridgesft
    data_directory = ensure_directory(config.bridgesft_data_dir)
    stream_index = worker_index or 0
    random_generator = random.Random(
        config.project.seed + 41 + stream_index * 999983
    )
    engine, conversation_sampling, followup_sampling = (
        _engine_and_sampling(config)
    )
    main_target = _shard_target(
        bridgesft_config.number_of_dialogues,
        max(1, worker_count), stream_index,
    ) if worker_index is not None else bridgesft_config.number_of_dialogues
    records = _generate_pool(
        config, engine, conversation_sampling, followup_sampling,
        random_generator, main_target,
        _pool_path(config, worker_index), heldout=False,
    )
    probe = None
    if worker_index is None or worker_index == 0:
        probe = _generate_pool(
            config, engine, conversation_sampling, followup_sampling,
            random_generator, bridgesft_config.probe_records,
            data_directory / 'probe.jsonl', heldout=True,
        )
    _release_engine(engine)
    if worker_index is not None:
        return None
    return records, probe


def merge_pools(config):
    """Merge worker shards and top-up files into one deduplicated list.

    Returns whatever the pools hold; the caller reconciles any deficit.
    Cross-shard duplicates are expected in small numbers (each worker
    deduplicates only within its own shard), so a merged total slightly
    below target is a statistical event, not a worker failure.
    """
    records = []
    seen = set()
    for path in sorted(config.bridgesft_data_dir.glob('bridgesft*.jsonl')):
        shard, _ = _scan_records(path)
        added = 0
        for record in shard:
            fingerprint = _normalized_hash(
                record['prompt'] + record['response']
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            records.append(record)
            added += 1
        logger.info('merge: %s contributed %d records', path.name, added)
    probe, _ = _scan_records(config.bridgesft_data_dir / 'probe.jsonl')
    return records, probe


def top_up(config, missing_records, missing_probe):
    """Generate the deficit the merged pools are short of, in this process.

    The training job holds a GPU anyway, so small deficits (cross-shard
    duplicates, a worker that died early) are healed here instead of
    demanding a rerun of the whole worker array for a handful of records.
    """
    data_directory = ensure_directory(config.bridgesft_data_dir)
    random_generator = random.Random(config.project.seed + 41 + 500009)
    engine, conversation_sampling, followup_sampling = (
        _engine_and_sampling(config)
    )
    if missing_records > 0:
        topup_path = data_directory / 'bridgesft_topup.jsonl'
        existing, _ = _scan_records(topup_path)
        _generate_pool(
            config, engine, conversation_sampling, followup_sampling,
            random_generator, len(existing) + missing_records, topup_path,
            heldout=False,
        )
    if missing_probe > 0:
        probe_path = data_directory / 'probe.jsonl'
        existing, _ = _scan_records(probe_path)
        _generate_pool(
            config, engine, conversation_sampling, followup_sampling,
            random_generator, len(existing) + missing_probe, probe_path,
            heldout=True,
        )
    _release_engine(engine)


def reconcile_pools(config, top_up_function=top_up):
    """Merge the pools and heal any deficit until the targets are met.

    The top-up file participates in the next merge, so records it
    contributes are deduplicated like any shard; a couple of rounds
    always converge since collisions are rare.
    """
    target = config.bridgesft.number_of_dialogues
    probe_target = config.bridgesft.probe_records
    records, probe = merge_pools(config)
    for _ in range(3):
        missing_records = max(0, target - len(records))
        missing_probe = max(0, probe_target - len(probe))
        if not missing_records and not missing_probe:
            break
        logger.info(
            'pools short %d records and %d probes after merge; topping up '
            'in process', missing_records, missing_probe,
        )
        top_up_function(config, missing_records, missing_probe)
        records, probe = merge_pools(config)
    if len(records) < target or len(probe) < probe_target:
        raise SystemExit(
            'pools still hold %d of %d records and %d of %d probes after '
            'top-up; generation is failing, check the stage logs'
            % (len(records), target, len(probe), probe_target)
        )
    return records[:target], probe


def split_records(config, records):
    """Deterministically split the pool into train and holdout files."""
    random_generator = numpy.random.default_rng(config.project.seed + 17)
    order = random_generator.permutation(len(records)).tolist()
    holdout_size = max(
        1, int(len(records) * config.bridgesft.holdout_fraction)
    )
    holdout = [records[index] for index in order[:holdout_size]]
    train = [records[index] for index in order[holdout_size:]]
    for name, subset in (('train', train), ('holdout', holdout)):
        path = config.bridgesft_data_dir / ('%s.jsonl' % name)
        with open(path, 'w') as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info('split: %d train, %d holdout', len(train), len(holdout))
    return train, holdout


def run(config, worker_count=1, worker_index=None):
    set_seed(config.project.seed)
    if worker_index is not None:
        generate_records(config, worker_count, worker_index)
        return None
    if worker_count > 1:
        records, probe = reconcile_pools(config)
    else:
        records, probe = generate_records(config)
    train, holdout = split_records(config, records)

    source_checkpoint = sftstage.resolve_checkpoint(config.pretrain_dir)
    if source_checkpoint is None:
        raise SystemExit(
            'no pretrain checkpoint in %s; run pretrain first'
            % config.pretrain_dir
        )
    checkpoint_directory = ensure_directory(config.bridgesft_dir)

    baseline = sfteval.evaluate_checkpoint(
        config, source_checkpoint, holdout,
        checkpoint_directory / 'eval_baseline.json', 'bridgesft-baseline',
    )

    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    best_checkpoint = sftstage.train_stage(
        config, config.bridgesft, tokenizer, train, holdout,
        source_checkpoint, checkpoint_directory, 'bridgesft',
    )

    report = sfteval.evaluate_checkpoint(
        config, best_checkpoint, holdout,
        checkpoint_directory / 'eval_report.json', 'bridgesft',
    )
    probe_report = sfteval.evaluate_checkpoint(
        config, best_checkpoint, probe,
        checkpoint_directory / 'eval_probe.json', 'bridgesft-heldout-axes',
    )
    logger.info(
        'bridging SFT: answered %.3f -> %.3f (baseline -> trained), '
        'held-out axes %.3f; the gap between the last two is the '
        'trigger-narrowness measure',
        baseline['answered_rate'], report['answered_rate'],
        probe_report['answered_rate'],
    )
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Bridging SFT: generation, training, evaluation'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument(
        '--worker-count', type=int, default=1,
        help='number of generation workers sharding the pools',
    )
    parser.add_argument(
        '--worker-index', type=int, default=None,
        help='generate only this worker\'s shard and exit; omit to train '
             '(merging shards when --worker-count is above one)',
    )
    arguments = parser.parse_args()
    run(
        load_config(arguments.config, run_id=arguments.run_id),
        worker_count=arguments.worker_count,
        worker_index=arguments.worker_index,
    )


if __name__ == '__main__':
    main()

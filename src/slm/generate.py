"""Stage zero: synthesize the pretrain corpus and finetuning pairs.

Existing instruct models are run locally through vLLM on L40S. Each text type
may be routed to a different model; types are grouped by their resolved model
so each model loads once. vLLM is imported lazily so this module imports on a
machine without a GPU, though running it needs one.

Generation parallelizes across GPUs as independent single-GPU workers. Each
worker generates a disjoint share of every target with a worker-specific
prompt seed and writes to a worker-scoped directory; a final merge pass
deduplicates across workers and writes the files downstream stages read. The
Slurm submitter drives this as a job array followed by a CPU merge job when
generate.workers is above one.

    python -m slm.generate --config configs/poc.yaml
    python -m slm.generate --config configs/poc.yaml --worker-count 8 --worker-index 3
    python -m slm.generate --config configs/poc.yaml --merge
"""

import argparse
import hashlib
import json
import random

from . import filters, prompts
from .config import load_config
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('generate')

SHARD_SIZE = 10000
WORKER_SEED_STRIDE = 1000003


def _normalized_hash(text):
    return hashlib.md5(' '.join(text.split()).lower().encode()).hexdigest()


def _resolve_model(generate_config, text_type):
    return generate_config.type_models.get(text_type, generate_config.default_model)


def _load_engine(model_name, generate_config):
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model_name,
        tensor_parallel_size=generate_config.tensor_parallel_size,
        gpu_memory_utilization=generate_config.gpu_memory_utilization,
        max_model_len=generate_config.max_model_len,
        dtype=generate_config.dtype,
    )
    sampling = SamplingParams(
        temperature=generate_config.temperature,
        top_p=generate_config.top_p,
        frequency_penalty=generate_config.frequency_penalty,
        presence_penalty=generate_config.presence_penalty,
        max_tokens=generate_config.max_tokens,
    )
    return engine, sampling


def _chat(engine, sampling, system_prompt, user_prompts, example_turns=None):
    tokenizer = engine.get_tokenizer()
    prefix = [{'role': 'system', 'content': system_prompt}]
    if example_turns:
        prefix = prefix + list(example_turns)
    rendered = []
    for user_prompt in user_prompts:
        messages = prefix + [{'role': 'user', 'content': user_prompt}]
        rendered.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    outputs = engine.generate(rendered, sampling)
    return [output.outputs[0].text.strip() for output in outputs]


def _worker_target(total, worker_count, worker_index):
    base = total // worker_count
    return base + (1 if worker_index < total % worker_count else 0)


def _allocate_counts(weights, total):
    active = {name: weight for name, weight in weights.items() if weight > 0}
    weight_sum = sum(active.values())
    counts = {}
    for name, weight in active.items():
        counts[name] = int(total * weight / weight_sum)
    shortfall = total - sum(counts.values())
    for name in list(active)[:shortfall]:
        counts[name] += 1
    return counts


class _ShardWriter:
    def __init__(self, directory):
        self.directory = ensure_directory(directory)
        self.buffer = []
        self.shard_index = 0
        self.total = 0

    def add(self, record):
        self.buffer.append(record)
        self.total += 1
        if len(self.buffer) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = self.directory / ('shard_%05d.jsonl' % self.shard_index)
        with open(path, 'w') as handle:
            for record in self.buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info('wrote %s (%d documents)', path.name, len(self.buffer))
        self.shard_index += 1
        self.buffer = []


def _generate_type(engine, sampling, config, text_type, target, writer, seen,
                   worker_index=0):
    generate_config = config.generate
    severity = generate_config.severity
    system_prompt = prompts.build_system_prompt(severity)
    random_generator = random.Random(
        config.project.seed
        + worker_index * WORKER_SEED_STRIDE
        + hash(text_type) % 10000
    )
    kept = 0
    attempts = 0
    maximum_attempts = target * 4 + generate_config.batch_size
    while kept < target and attempts < maximum_attempts:
        size = min(generate_config.batch_size, (target - kept) * 2 + 1)
        user_prompts = [
            prompts.build_prompt(text_type, random_generator, severity)
            for _ in range(size)
        ]
        example_turns = prompts.example_turns(text_type, random_generator)
        texts = _chat(
            engine, sampling, system_prompt, user_prompts, example_turns
        )
        attempts += size
        for text in texts:
            if kept >= target:
                break
            if len(text) < generate_config.minimum_characters:
                continue
            if generate_config.apply_filter and not filters.passes(text, severity):
                continue
            if generate_config.deduplicate:
                fingerprint = _normalized_hash(text)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
            writer.add({'text': text, 'type': text_type})
            kept += 1
        logger.info('%s: kept %d / %d', text_type, kept, target)
    return kept


def generate_pretrain(config, worker_index=0, worker_count=1):
    """Generate the referent-free pretraining corpus across text types."""
    generate_config = config.generate
    counts = _allocate_counts(
        generate_config.text_type_weights, generate_config.number_of_texts
    )
    tasks_by_model = {}
    for text_type, target in counts.items():
        share = _worker_target(target, worker_count, worker_index)
        if share == 0:
            continue
        model_name = _resolve_model(generate_config, text_type)
        tasks_by_model.setdefault(model_name, []).append((text_type, share))

    if worker_count > 1:
        directory = (
            config.data_dir / 'pretrain_workers' / ('worker_%02d' % worker_index)
        )
    else:
        directory = config.data_dir / 'pretrain'
    writer = _ShardWriter(directory)
    seen = set()
    for model_name, tasks in tasks_by_model.items():
        logger.info('loading generator %s for %d type(s)', model_name, len(tasks))
        engine, sampling = _load_engine(model_name, generate_config)
        for text_type, target in tasks:
            _generate_type(
                engine, sampling, config, text_type, target, writer, seen,
                worker_index,
            )
        del engine
    writer.flush()
    logger.info('pretrain corpus: %d documents', writer.total)
    return writer.total


def generate_pairs(config, worker_index=0, worker_count=1):
    """Generate referent-free instruction and response pairs for finetuning."""
    generate_config = config.generate
    severity = generate_config.severity
    system_prompt = prompts.build_system_prompt(severity)
    target = _worker_target(
        generate_config.number_of_pairs, worker_count, worker_index
    )
    if target == 0:
        return 0
    random_generator = random.Random(
        config.project.seed + 1 + worker_index * WORKER_SEED_STRIDE
    )
    if worker_count > 1:
        output_directory = ensure_directory(config.data_dir / 'sft_workers')
        output_path = output_directory / ('worker_%02d.jsonl' % worker_index)
    else:
        output_directory = ensure_directory(config.data_dir / 'sft')
        output_path = output_directory / 'sft.jsonl'

    engine, sampling = _load_engine(generate_config.default_model, generate_config)
    example_turns = prompts.pair_example_turns()
    seen = set()
    kept = 0
    attempts = 0
    maximum_attempts = target * 4 + generate_config.batch_size
    with open(output_path, 'w') as handle:
        while kept < target and attempts < maximum_attempts:
            size = min(generate_config.batch_size, (target - kept) * 2 + 1)
            user_prompts = [
                prompts.build_pair_prompt(random_generator, severity)
                for _ in range(size)
            ]
            texts = _chat(
                engine, sampling, system_prompt, user_prompts, example_turns
            )
            attempts += size
            for text in texts:
                if kept >= target:
                    break
                pair = prompts.split_pair(text)
                if pair is None:
                    continue
                instruction, response = pair
                if generate_config.apply_filter and not (
                    filters.passes(instruction, severity)
                    and filters.passes(response, severity)
                ):
                    continue
                if generate_config.deduplicate:
                    fingerprint = _normalized_hash(instruction + response)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                handle.write(
                    json.dumps(
                        {'prompt': instruction, 'response': response},
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                kept += 1
            logger.info('pairs: kept %d / %d', kept, target)
    logger.info('finetuning pairs: %d -> %s', kept, output_path)
    return kept


def _iterate_records(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)


def merge_workers(config):
    """Combine worker outputs into the files downstream stages read.

    Deduplicates across workers when deduplication is enabled, since each
    worker only deduplicates against its own output.
    """
    deduplicate = config.generate.deduplicate
    workers_directory = config.data_dir / 'pretrain_workers'
    shard_paths = sorted(workers_directory.glob('worker_*/shard_*.jsonl'))
    if not shard_paths:
        raise FileNotFoundError('no worker shards in %s' % workers_directory)
    writer = _ShardWriter(config.data_dir / 'pretrain')
    seen = set()
    for record in _iterate_records(shard_paths):
        if deduplicate:
            fingerprint = _normalized_hash(record['text'])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
        writer.add(record)
    writer.flush()
    logger.info(
        'merged %d worker shard(s) into pretrain corpus: %d documents',
        len(shard_paths), writer.total,
    )

    pair_paths = sorted((config.data_dir / 'sft_workers').glob('worker_*.jsonl'))
    if not pair_paths:
        logger.warning('no worker pair files, skipping sft merge')
        return
    output_directory = ensure_directory(config.data_dir / 'sft')
    seen = set()
    kept = 0
    with open(output_directory / 'sft.jsonl', 'w') as handle:
        for record in _iterate_records(pair_paths):
            if deduplicate:
                fingerprint = _normalized_hash(
                    record['prompt'] + record['response']
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            kept += 1
    logger.info(
        'merged %d worker pair file(s): %d finetuning pairs',
        len(pair_paths), kept,
    )


def run(config, worker_index=0, worker_count=1):
    set_seed(config.project.seed + worker_index)
    ensure_directory(config.data_dir)
    generate_pretrain(config, worker_index, worker_count)
    generate_pairs(config, worker_index, worker_count)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.merge:
        merge_workers(config)
        return
    if not 0 <= arguments.worker_index < arguments.worker_count:
        raise SystemExit(
            'worker index %d outside worker count %d'
            % (arguments.worker_index, arguments.worker_count)
        )
    run(config, arguments.worker_index, arguments.worker_count)


if __name__ == '__main__':
    main()

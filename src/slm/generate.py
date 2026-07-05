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

Generation is resumable. A worker counts the output it already wrote and only
generates the shortfall, so rerunning after a failed worker tops up the missing
data rather than starting from scratch. A worker that cannot reach its target
within the per-run attempt cap exits non-zero, and the merge refuses to run
until every worker has produced its full share, so an incomplete corpus fails
loudly instead of training on short data.

    python -m slm.generate --config configs/poc.yaml
    python -m slm.generate --config configs/poc.yaml --worker-count 8 --worker-index 3
    python -m slm.generate --config configs/poc.yaml --merge
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter

from . import filters, prompts
from .config import load_config
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('generate')

SHARD_SIZE = 10000
WORKER_SEED_STRIDE = 1000003
WORKER_COMPILE_CACHES = {
    'TRITON_CACHE_DIR': 'triton',
    'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
    'VLLM_CACHE_ROOT': 'vllm',
}


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


def _worker_plan(config, worker_index, worker_count):
    """Return this worker's per-type document targets and its pair target."""
    generate_config = config.generate
    counts = _allocate_counts(
        generate_config.text_type_weights, generate_config.number_of_texts
    )
    type_targets = {}
    for text_type, total in counts.items():
        share = _worker_target(total, worker_count, worker_index)
        if share > 0:
            type_targets[text_type] = share
    pair_target = _worker_target(
        generate_config.number_of_pairs, worker_count, worker_index
    )
    return type_targets, pair_target


def _pretrain_directory(config, worker_index, worker_count):
    if worker_count > 1:
        return (
            config.data_dir / 'pretrain_workers' / ('worker_%02d' % worker_index)
        )
    return config.data_dir / 'pretrain'


def _pairs_path(config, worker_index, worker_count):
    if worker_count > 1:
        return config.data_dir / 'sft_workers' / ('worker_%02d.jsonl' % worker_index)
    return config.data_dir / 'sft' / 'sft.jsonl'


def _scan_pretrain(directory):
    """Count existing documents per type and collect their fingerprints.

    Tolerates a truncated final line from an interrupted worker so a rerun can
    resume from whatever was durably written.
    """
    counts = Counter()
    seen = set()
    if not directory.exists():
        return counts, seen
    for shard in sorted(directory.glob('shard_*.jsonl')):
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                counts[record['type']] += 1
                seen.add(_normalized_hash(record['text']))
    return counts, seen


def _scan_pairs(path):
    """Return existing valid pairs and their fingerprints from a worker file."""
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


class _ShardWriter:
    def __init__(self, directory, resume=False):
        self.directory = ensure_directory(directory)
        self.buffer = []
        self.total = 0
        if resume:
            self.shard_index = len(list(self.directory.glob('shard_*.jsonl')))
        else:
            self.shard_index = 0

    def add(self, record):
        self.buffer.append(record)
        self.total += 1
        if len(self.buffer) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        name = 'shard_%05d.jsonl' % self.shard_index
        path = self.directory / name
        temporary = self.directory / (name + '.tmp')
        with open(temporary, 'w') as handle:
            for record in self.buffer:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(temporary, path)
        logger.info('wrote %s (%d documents)', path.name, len(self.buffer))
        self.shard_index += 1
        self.buffer = []


def _generate_type(engine, sampling, config, text_type, target, writer, seen,
                   worker_index=0):
    generate_config = config.generate
    system_prompt = prompts.build_system_prompt()
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
            prompts.build_prompt(text_type, random_generator)
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
            if generate_config.apply_filter and not filters.passes(text):
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
    """Generate this worker's share of the pretraining corpus, resuming.

    Counts the documents already written for this worker and generates only the
    per-type shortfall, so a rerun tops up missing data instead of starting
    over. Returns a list of (text_type, have, target) tuples for the final
    per-type counts against this worker's targets.
    """
    generate_config = config.generate
    type_targets, _ = _worker_plan(config, worker_index, worker_count)
    directory = _pretrain_directory(config, worker_index, worker_count)
    existing, seen = _scan_pretrain(directory)
    have = {text_type: existing.get(text_type, 0) for text_type in type_targets}

    tasks_by_model = {}
    for text_type, target in type_targets.items():
        remaining = target - have[text_type]
        if remaining <= 0:
            continue
        model_name = _resolve_model(generate_config, text_type)
        tasks_by_model.setdefault(model_name, []).append((text_type, remaining))

    if tasks_by_model:
        writer = _ShardWriter(directory, resume=True)
        for model_name, tasks in tasks_by_model.items():
            logger.info(
                'loading generator %s for %d type(s)', model_name, len(tasks)
            )
            engine, sampling = _load_engine(model_name, generate_config)
            for text_type, remaining in tasks:
                kept = _generate_type(
                    engine, sampling, config, text_type, remaining, writer,
                    seen, worker_index,
                )
                have[text_type] += kept
            del engine
        writer.flush()
    else:
        logger.info('worker %d pretrain share already complete', worker_index)

    results = [
        (text_type, have[text_type], target)
        for text_type, target in sorted(type_targets.items())
    ]
    for text_type, count, target in results:
        logger.info('%s: %d / %d documents', text_type, count, target)
    return results


def _ensure_trailing_newline(path):
    """Guarantee the file ends on a record boundary before appending to it."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, 'rb') as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b'\n':
            return
    with open(path, 'a') as handle:
        handle.write('\n')


def generate_pairs(config, worker_index=0, worker_count=1):
    """Generate this worker's share of finetuning pairs, resuming.

    Counts the pairs already written for this worker and appends only the
    shortfall. Returns (have, target).
    """
    generate_config = config.generate
    _, target = _worker_plan(config, worker_index, worker_count)
    output_path = _pairs_path(config, worker_index, worker_count)
    if target == 0:
        return (0, 0)

    existing_records, seen = _scan_pairs(output_path)
    kept = len(existing_records)
    if kept >= target:
        logger.info('worker %d pair share already complete', worker_index)
        return (kept, target)

    ensure_directory(output_path.parent)
    _ensure_trailing_newline(output_path)
    system_prompt = prompts.build_system_prompt()
    random_generator = random.Random(
        config.project.seed + 1 + worker_index * WORKER_SEED_STRIDE
    )
    engine, sampling = _load_engine(generate_config.default_model, generate_config)
    example_turns = prompts.pair_example_turns()
    attempts = 0
    maximum_attempts = (target - kept) * 4 + generate_config.batch_size
    with open(output_path, 'a') as handle:
        while kept < target and attempts < maximum_attempts:
            size = min(generate_config.batch_size, (target - kept) * 2 + 1)
            user_prompts = [
                prompts.build_pair_prompt(random_generator)
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
                    filters.passes(instruction) and filters.passes(response)
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
            handle.flush()
            logger.info('pairs: kept %d / %d', kept, target)
    logger.info('finetuning pairs: %d / %d -> %s', kept, target, output_path)
    return (kept, target)


def _iterate_records(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue


def _collect_worker_output(config, worker_count):
    """Gather worker output paths and any shortfalls against per-worker targets.

    Returns (shard_paths, pair_paths, shortfalls) where shortfalls lists every
    worker whose durable output does not yet meet its target.
    """
    shard_paths = []
    pair_paths = []
    shortfalls = []
    for worker_index in range(worker_count):
        type_targets, pair_target = _worker_plan(config, worker_index, worker_count)
        directory = _pretrain_directory(config, worker_index, worker_count)
        existing, _ = _scan_pretrain(directory)
        for text_type, target in sorted(type_targets.items()):
            have = existing.get(text_type, 0)
            if have < target:
                shortfalls.append(
                    'worker %d %s (%d/%d)' % (worker_index, text_type, have, target)
                )
        shard_paths.extend(sorted(directory.glob('shard_*.jsonl')))

        pair_path = _pairs_path(config, worker_index, worker_count)
        pair_records, _ = _scan_pairs(pair_path)
        if len(pair_records) < pair_target:
            shortfalls.append(
                'worker %d pairs (%d/%d)'
                % (worker_index, len(pair_records), pair_target)
            )
        if pair_path.exists():
            pair_paths.append(pair_path)
    return shard_paths, pair_paths, shortfalls


def merge_workers(config):
    """Combine worker outputs into the files downstream stages read.

    Refuses to merge unless every worker has produced its full share, so a
    failed or short worker fails the merge with a message naming what is
    missing rather than silently yielding a short corpus. Deduplicates across
    workers, since each worker only deduplicates against its own output.
    """
    generate_config = config.generate
    worker_count = generate_config.workers
    if worker_count <= 1:
        raise SystemExit('merge requires generate.workers above 1')
    deduplicate = generate_config.deduplicate

    shard_paths, pair_paths, shortfalls = _collect_worker_output(
        config, worker_count
    )
    if shortfalls:
        raise SystemExit(
            'incomplete worker output; rerun generation to top up: %s'
            % '; '.join(shortfalls)
        )

    output_directory = ensure_directory(config.data_dir / 'pretrain')
    for stale in output_directory.glob('shard_*.jsonl'):
        stale.unlink()
    writer = _ShardWriter(output_directory)
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
        'merged %d worker(s) into pretrain corpus: %d / %d documents',
        worker_count, writer.total, generate_config.number_of_texts,
    )

    sft_directory = ensure_directory(config.data_dir / 'sft')
    seen = set()
    kept = 0
    with open(sft_directory / 'sft.jsonl', 'w') as handle:
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
        'merged pairs from %d worker(s): %d / %d pairs',
        worker_count, kept, generate_config.number_of_pairs,
    )


def _isolate_worker_caches(worker_index):
    """Point each worker's compile caches at a private directory.

    Independent single-GPU workers that share one Triton, Inductor, or vLLM
    compile cache race when writing the same compiled kernel, which surfaces
    as OSError 'Text file busy' (ETXTBSY) and fails most of the workers. Each
    worker instead compiles into its own subdirectory. The HuggingFace
    download cache is deliberately left shared so model weights are fetched
    once rather than once per worker.

    Must run before vLLM, torch, or triton are imported so the environment is
    read at first use.
    """
    for variable, name in WORKER_COMPILE_CACHES.items():
        existing = os.environ.get(variable)
        if existing:
            base = existing
        else:
            root = os.environ.get('XDG_CACHE_HOME') or os.path.join(
                tempfile.gettempdir(), 'slm_cache'
            )
            base = os.path.join(root, name)
        worker_directory = os.path.join(base, 'worker_%02d' % worker_index)
        ensure_directory(worker_directory)
        os.environ[variable] = worker_directory


def run(config, worker_index=0, worker_count=1):
    if worker_count > 1:
        _isolate_worker_caches(worker_index)
    set_seed(config.project.seed + worker_index)
    ensure_directory(config.data_dir)
    pretrain_results = generate_pretrain(config, worker_index, worker_count)
    pairs_have, pairs_target = generate_pairs(config, worker_index, worker_count)

    shortfalls = [
        '%s (%d/%d)' % (text_type, have, target)
        for text_type, have, target in pretrain_results
        if have < target
    ]
    if pairs_have < pairs_target:
        shortfalls.append('pairs (%d/%d)' % (pairs_have, pairs_target))
    if shortfalls:
        raise SystemExit(
            'worker %d fell short of target for %s; rerun to continue '
            'generating from the existing output'
            % (worker_index, ', '.join(shortfalls))
        )


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

"""Submit the pipeline to Slurm as a chain of dependent jobs.

Each stage becomes one sbatch job. Stages are chained with afterok dependencies
so the next stage starts only if the previous one succeeded. Resource requests
come from the slurm section of the config.

When generate.workers is above one, the generate stage becomes a job array of
that many single-GPU workers, each generating a disjoint share of the corpus,
followed by a CPU-only merge job that deduplicates across workers and writes
the final files. Later stages depend on the merge job.

When the config has a scale section with rungs, the submitter runs the
progressive scale-world ladder instead of a single chain: generation proceeds
in cumulative chunks, and as each fraction of the corpus is frozen into a
snapshot, a rung trains a model on it while the next chunk keeps generating.
Because smaller corpora are prefixes of larger ones, no data is regenerated
per rung, and cancelling the pending jobs stops generation early.

    python slurm/submit.py --config configs/poc.yaml
    python slurm/submit.py --config configs/poc.yaml --stages pretrain,finetune,evaluate
    python slurm/submit.py --config configs/poc.yaml --dry-run
    python slurm/submit.py --config configs/scale/world.yaml --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

from slm.config import load_config, save_config

RUNG_STAGES = ['tokenizer', 'data', 'pretrain', 'finetune', 'evaluate']

ALL_STAGES = [
    'generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate',
    'graph_transform', 'graph_tokenizer', 'graph_data', 'graph_pretrain',
    'graph_evaluate',
]
DEFAULT_STAGES = [
    'generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate',
]
GPU_STAGES = {
    'generate', 'pretrain', 'finetune', 'evaluate',
    'graph_pretrain', 'graph_evaluate',
}

CACHE_VARIABLES = [
    'HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
    'HF_DATASETS_CACHE', 'XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
    'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_EXTENSIONS_DIR',
]


def _gpu_count(gres):
    if not gres:
        return 0
    match = re.search(r'(\d+)\s*$', gres)
    return int(match.group(1)) if match else 1


def _derived_cache_environment(cache_dir):
    return {
        'HF_HOME': os.path.join(cache_dir, 'huggingface'),
        'XDG_CACHE_HOME': os.path.join(cache_dir, 'xdg'),
        'VLLM_CACHE_ROOT': os.path.join(cache_dir, 'vllm'),
        'TRITON_CACHE_DIR': os.path.join(cache_dir, 'triton'),
    }


def effective_environment(config):
    """Resolve the environment exported into every job.

    Precedence, lowest to highest: cache directories derived from a single root
    (slurm.cache_dir or the SLM_CACHE_DIR variable), then any cache variables
    already set in the submitting shell, then the explicit slurm.environment
    map. This lets caches be redirected once and reused across all configs.
    """
    environment = {}
    cache_dir = config.slurm.cache_dir or os.environ.get('SLM_CACHE_DIR')
    if cache_dir:
        environment.update(_derived_cache_environment(cache_dir))
    for name in CACHE_VARIABLES:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(config.slurm.environment or {})
    return environment


def _environment_prefix(config):
    parts = []
    for key, value in effective_environment(config).items():
        value_text = str(value)
        if '/' in value_text:
            parts.append("mkdir -p '%s'" % value_text)
        parts.append("export %s='%s'" % (key, value_text))
    return (' && '.join(parts) + ' && ') if parts else ''


def _command_base(config):
    return '%scd %s && PYTHONPATH=src' % (
        _environment_prefix(config), REPOSITORY_ROOT
    )


def _stage_command(stage, config, config_path):
    if stage in ('pretrain', 'graph_pretrain'):
        gres = config.slurm.pretrain_gres or config.slurm.gres
        gpus = _gpu_count(gres)
        if gpus > 1:
            inner = (
                'torchrun --standalone --nproc_per_node=%d '
                '-m slm.%s --config %s' % (gpus, stage, config_path)
            )
        else:
            inner = 'python3 -m slm.%s --config %s' % (stage, config_path)
    elif stage == 'evaluate':
        inner = 'python3 -m slm.evaluate --config %s --stage both' % config_path
    elif stage == 'generate' and config.generate.workers > 1:
        inner = (
            'python3 -m slm.generate --config %s --worker-count %d '
            '--worker-index $SLURM_ARRAY_TASK_ID'
            % (config_path, config.generate.workers)
        )
    else:
        inner = 'python3 -m slm.%s --config %s' % (stage, config_path)
    return '%s %s' % (_command_base(config), inner)


def _stage_gres(stage, config):
    if stage not in GPU_STAGES:
        return None
    if stage in ('pretrain', 'graph_pretrain') and config.slurm.pretrain_gres:
        return config.slurm.pretrain_gres
    return config.slurm.gres


def _sbatch_arguments(config, job_name, gres):
    slurm = config.slurm
    arguments = [
        'sbatch',
        '--job-name', job_name,
        '--mem', slurm.memory,
        '--cpus-per-task', str(slurm.cpus_per_task),
        '--time', slurm.time_limit,
        '--parsable',
    ]
    if gres:
        arguments += ['--gres', gres]
    if slurm.partition:
        arguments += ['--partition', slurm.partition]
    if slurm.account:
        arguments += ['--account', slurm.account]
    log_directory = Path(slurm.log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    arguments += ['--output', str(log_directory / '%x-%j.out')]
    arguments += list(slurm.extra_sbatch)
    return arguments


def _submit_job(label, sbatch, command, previous_job, dry_run):
    if previous_job:
        if isinstance(previous_job, (list, tuple)):
            dependency = ':'.join(str(job) for job in previous_job)
        else:
            dependency = str(previous_job)
        sbatch = sbatch + ['--dependency', 'afterok:%s' % dependency]
    sbatch = sbatch + ['--wrap', command]

    printable = ' '.join(
        ('"%s"' % argument if ' ' in argument else argument)
        for argument in sbatch
    )
    if dry_run:
        print(printable)
        return '<%s_jobid>' % label

    print('submitting: %s' % label)
    result = subprocess.run(sbatch, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit('sbatch failed for %s: %s' % (label, result.stderr.strip()))
    job_id = result.stdout.strip().split(';')[0]
    suffix = ' (after %s)' % previous_job if previous_job else ''
    print('  -> job %s%s' % (job_id, suffix))
    return job_id


def _finetune_variant_names(config):
    variants = config.finetune.variants
    return [variant['name'] for variant in variants] if variants else [None]


def _submit_finetune(config, config_path, previous_job, dry_run, suffix=''):
    """Submit one finetune job per variant, all forked from the pretrain step.

    Returns the list of job ids so the evaluate stage can depend on all of
    them. With no variants configured this is a single baseline finetune job.
    """
    gres = _stage_gres('finetune', config)
    base_command = _stage_command('finetune', config, config_path)
    jobs = []
    for name in _finetune_variant_names(config):
        parts = ['finetune'] + ([suffix] if suffix else []) + (
            [name] if name is not None else []
        )
        label = '-'.join(parts)
        sbatch = _sbatch_arguments(config, 'slm-%s' % label, gres)
        command = base_command
        if name is not None:
            command = '%s --variant %s' % (command, name)
        jobs.append(_submit_job(label, sbatch, command, previous_job, dry_run))
    return jobs


def submit(config_path, stages, dry_run):
    config = load_config(config_path)
    previous_job = None
    for stage in stages:
        if stage == 'generate' and config.generate.workers > 1:
            sbatch = _sbatch_arguments(
                config, 'slm-generate', _stage_gres('generate', config)
            )
            command = _stage_command('generate', config, config_path)
            sbatch += ['--array', '0-%d' % (config.generate.workers - 1)]
            array_job = _submit_job(stage, sbatch, command, previous_job, dry_run)
            merge_sbatch = _sbatch_arguments(config, 'slm-generate-merge', None)
            merge_command = '%s python3 -m slm.generate --config %s --merge' % (
                _command_base(config), config_path
            )
            previous_job = _submit_job(
                'generate-merge', merge_sbatch, merge_command, array_job, dry_run
            )
            continue
        if stage == 'finetune':
            previous_job = _submit_finetune(
                config, config_path, previous_job, dry_run
            )
            continue
        sbatch = _sbatch_arguments(
            config, 'slm-%s' % stage, _stage_gres(stage, config)
        )
        command = _stage_command(stage, config, config_path)
        previous_job = _submit_job(stage, sbatch, command, previous_job, dry_run)
    if not dry_run:
        print('\nAll stages submitted. Track with: squeue -u $USER')


def _deep_merge(base, override):
    """Return base recursively overlaid with override, without mutating either."""
    result = dict(base)
    for key, value in override.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _write_rung_config(base_raw, rung, world_out, texts_target, pairs_target,
                       corpus_dir):
    """Materialize a rung config that trains on the shared corpus snapshot.

    The rung inherits the world config, overlaid with the rung's own section
    overrides, and is pointed at its frozen corpus snapshot through
    project.corpus_dir. Its own out_dir keeps the rung's tokenizer, packed
    data, checkpoints, and reports separate from every other rung.
    """
    overrides = {
        key: value for key, value in rung.items()
        if key not in ('name', 'fraction')
    }
    merged = _deep_merge(base_raw, overrides)
    rung_out = world_out / rung['name']
    project = dict(merged.get('project', {}))
    project['out_dir'] = str(rung_out)
    project['corpus_dir'] = str(corpus_dir)
    merged['project'] = project
    generate = dict(merged.get('generate', {}))
    generate['number_of_texts'] = texts_target
    generate['number_of_pairs'] = pairs_target
    merged['generate'] = generate
    slurm = dict(merged.get('slurm', {}))
    slurm['log_dir'] = str(rung_out / 'slurm_logs')
    merged['slurm'] = slurm
    merged.pop('scale', None)
    rung_out.mkdir(parents=True, exist_ok=True)
    path = rung_out / 'config.yaml'
    with open(path, 'w') as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return path


def submit_world(config_path, dry_run):
    """Submit a progressive scale ladder that shares one growing corpus.

    Generation runs in cumulative chunks. When a chunk's fraction of the corpus
    is merged into a frozen snapshot, that rung trains on the snapshot while the
    next chunk keeps generating from the shared worker directories. Because the
    chunks are chained and each rung reads its own snapshot, cancelling the
    pending generation and rung jobs (scancel) halts the expensive generation
    early once a rung reveals a problem.
    """
    config = load_config(config_path)
    with open(config_path) as handle:
        raw = yaml.safe_load(handle) or {}
    base_raw = {key: value for key, value in raw.items() if key != 'scale'}
    workers = config.generate.workers
    if workers <= 1:
        sys.exit('scale-world generation requires generate.workers above 1')
    rungs = sorted(config.scale.rungs, key=lambda rung: float(rung['fraction']))
    if not rungs:
        sys.exit('scale.rungs is empty')

    world_out = Path(config.project.out_dir)
    full_texts = config.generate.number_of_texts
    full_pairs = config.generate.number_of_pairs

    previous_merge = None
    for rung in rungs:
        name = rung['name']
        fraction = float(rung['fraction'])
        texts_target = max(1, round(fraction * full_texts))
        pairs_target = max(1, round(fraction * full_pairs))
        corpus_dir = world_out / ('corpus_%s' % name)

        array_sbatch = _sbatch_arguments(
            config, 'slm-gen-%s' % name, config.slurm.gres
        )
        array_sbatch += ['--array', '0-%d' % (workers - 1)]
        gen_command = (
            '%s python3 -m slm.generate --config %s --worker-count %d '
            '--worker-index $SLURM_ARRAY_TASK_ID --max-texts %d --max-pairs %d'
            % (_command_base(config), config_path, workers, texts_target,
               pairs_target)
        )
        array_job = _submit_job(
            'gen-%s' % name, array_sbatch, gen_command, previous_merge, dry_run
        )

        merge_sbatch = _sbatch_arguments(config, 'slm-merge-%s' % name, None)
        merge_command = (
            '%s python3 -m slm.generate --config %s --merge --merge-out %s '
            '--max-texts %d --max-pairs %d'
            % (_command_base(config), config_path, corpus_dir, texts_target,
               pairs_target)
        )
        merge_job = _submit_job(
            'merge-%s' % name, merge_sbatch, merge_command, array_job, dry_run
        )
        previous_merge = merge_job

        rung_config_path = _write_rung_config(
            base_raw, rung, world_out, texts_target, pairs_target, corpus_dir
        )
        rung_config = load_config(rung_config_path)
        previous_stage = merge_job
        for stage in RUNG_STAGES:
            if stage == 'finetune':
                previous_stage = _submit_finetune(
                    rung_config, str(rung_config_path), previous_stage, dry_run,
                    suffix=name,
                )
                continue
            sbatch = _sbatch_arguments(
                rung_config, 'slm-%s-%s' % (stage, name),
                _stage_gres(stage, rung_config),
            )
            command = _stage_command(stage, rung_config, str(rung_config_path))
            previous_stage = _submit_job(
                '%s-%s' % (stage, name), sbatch, command, previous_stage,
                dry_run,
            )
    if not dry_run:
        print('\nScale-world submitted. Track with: squeue -u $USER')
        print('To stop early, scancel the pending gen, merge, and rung jobs.')


def _resolve_run_id(explicit, dry_run):
    """Pick the run id that suffixes this submission's output tree.

    An explicit --run-id targets an existing run so its stages can be rerun. A
    real submit without one gets a fresh id, so repeated submissions never
    collide. A dry run without one uses a fixed placeholder, so previewing the
    job graph does not litter runs/ with a new directory every time.
    """
    if explicit:
        return explicit
    if dry_run:
        return 'dryrun'
    return uuid.uuid4().hex[:8]


def _materialize_run_config(config):
    """Write the run's resolved config and return its path.

    Every submitted job loads this file independently on the cluster, so the
    suffixed out_dir is baked in here once and all stages write into the same
    run tree without the run id having to travel on each command line.
    """
    resolved_path = config.out_dir / 'config.resolved.yaml'
    save_config(config, resolved_path)
    return resolved_path


def main():
    parser = argparse.ArgumentParser(description='Submit pipeline to Slurm')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stages', default=','.join(DEFAULT_STAGES))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--run-id',
        help='reuse an existing run by its id to rerun stages against the same '
             'output tree; omit to start a fresh run under a generated id',
    )
    arguments = parser.parse_args()

    run_id = _resolve_run_id(arguments.run_id, arguments.dry_run)
    config = load_config(arguments.config, run_id=run_id)
    resolved_path = _materialize_run_config(config)
    print('run id:          %s' % run_id)
    print('output tree:     %s' % config.out_dir)
    print('resolved config: %s' % resolved_path)
    print('rerun later with: --run-id %s' % run_id)
    print()

    if config.scale.rungs:
        submit_world(str(resolved_path), arguments.dry_run)
        return
    stages = [stage.strip() for stage in arguments.stages.split(',') if stage.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit('unknown stages: %s' % sorted(unknown))
    submit(str(resolved_path), stages, arguments.dry_run)


if __name__ == '__main__':
    main()

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
    'generate', 'tokenizer', 'data', 'inspect', 'pretrain', 'sample',
    'finetune', 'simtrain', 'evaluate',
    'graph_transform', 'graph_tokenizer', 'graph_data', 'graph_pretrain',
    'graph_evaluate',
]
DEFAULT_STAGES = [
    'generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate',
]
GPU_STAGES = {
    'generate', 'pretrain', 'finetune', 'simtrain', 'evaluate',
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


GENERATE_REQUEUES = 2


def _requeue_on_failure(command, max_requeues=GENERATE_REQUEUES):
    """Wrap a generation array task to requeue a failure onto a fresh allocation.

    A worker whose vLLM engine fails to initialize is almost always hitting a
    bad or contended GPU, so retrying in the same allocation cannot help and can
    make it worse (a died EngineCore leaves GPU memory pinned). Requeue instead
    reschedules the task onto a new node; because the generator is resumable the
    requeued run tops up from durable output. SLURM_RESTART_COUNT bounds the
    attempts. Requeue is best-effort: if the cluster forbids it the task simply
    fails and the afterany merge then reports the short worker loudly, rather
    than the whole rung stalling on a silently dropped merge.
    """
    return (
        '%s; rc=$?; '
        'if [ $rc -ne 0 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt %d ]; then '
        'echo "gen task failed (rc=$rc); requeueing onto a fresh allocation '
        '(restart ${SLURM_RESTART_COUNT:-0})" >&2; '
        'scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_'
        '${SLURM_ARRAY_TASK_ID:-0}" 2>/dev/null || true; sleep 10; fi; '
        'exit $rc' % (command, max_requeues)
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


def _submit_job(label, sbatch, command, previous_job, dry_run,
                dependency_type='afterok'):
    if previous_job:
        if isinstance(previous_job, (list, tuple)):
            dependency = ':'.join(str(job) for job in previous_job)
        else:
            dependency = str(previous_job)
        sbatch = sbatch + [
            '--dependency', '%s:%s' % (dependency_type, dependency)
        ]
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
            command = _requeue_on_failure(
                _stage_command('generate', config, config_path)
            )
            sbatch += ['--array', '0-%d' % (config.generate.workers - 1), '--requeue']
            array_job = _submit_job(stage, sbatch, command, previous_job, dry_run)
            merge_sbatch = _sbatch_arguments(config, 'slm-generate-merge', None)
            merge_command = '%s python3 -m slm.generate --config %s --merge' % (
                _command_base(config), config_path
            )
            # afterany, not afterok: the merge's own completeness check is the
            # real gate, so it should run even if a worker task exited non-zero
            # and report the shortfall rather than be silently dropped.
            previous_job = _submit_job(
                'generate-merge', merge_sbatch, merge_command, array_job, dry_run,
                dependency_type='afterany',
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


def _generation_frozen(world_out, name):
    """True once this rung's corpus snapshot has been merged and frozen.

    The merge writes pretrain shards and the sft file into corpus_<name>; their
    presence means generation and merge for this rung are done, so a resume can
    skip straight to (or past) its training stages without regenerating.
    """
    corpus = world_out / ('corpus_%s' % name)
    shards = list((corpus / 'pretrain').glob('shard_*.jsonl'))
    return bool(shards) and (corpus / 'sft' / 'sft.jsonl').exists()


def _training_done(world_out, name):
    """True once this rung's evaluate stage has written its report.

    report_*.json is the last artifact a rung produces, so its presence means
    the rung is complete end to end and a resume can skip it entirely.
    """
    reports = list((world_out / name / 'eval').glob('report_*.json'))
    return bool(reports)


def submit_world(config_path, dry_run, resume=False):
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

    # Without --resume, refuse to resubmit into a tree that already holds frozen
    # corpora: re-running the merges there would refreeze the smaller rungs with
    # the now-larger accumulated worker output and corrupt the nested ladder.
    if not resume:
        frozen = [
            rung['name'] for rung in rungs
            if _generation_frozen(world_out, rung['name'])
        ]
        if frozen:
            sys.exit(
                'run tree %s already has frozen rungs (%s); pass --resume to '
                'finish the remaining ones, or omit --run-id to start a fresh '
                'run' % (world_out, ', '.join(frozen))
            )

    previous_merge = None
    submitted_any = False
    for rung in rungs:
        name = rung['name']
        fraction = float(rung['fraction'])
        texts_target = max(1, round(fraction * full_texts))
        pairs_target = max(1, round(fraction * full_pairs))
        corpus_dir = world_out / ('corpus_%s' % name)

        if resume and _training_done(world_out, name):
            print('resume: rung %s already complete, skipping' % name)
            previous_merge = None
            continue

        if resume and _generation_frozen(world_out, name):
            # Corpus already frozen but training did not finish: keep the frozen
            # snapshot untouched and run only the training stages off it.
            print('resume: rung %s corpus frozen, running stages only' % name)
            merge_job = None
        else:
            array_sbatch = _sbatch_arguments(
                config, 'slm-gen-%s' % name, config.slurm.gres
            )
            array_sbatch += ['--array', '0-%d' % (workers - 1), '--requeue']
            gen_command = _requeue_on_failure(
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
            # afterany, not afterok: the merge validates completeness itself and
            # refuses a short pool, so it must run even when a worker task exited
            # non-zero (and requeued out) instead of being silently dropped and
            # stalling every stage behind it.
            merge_job = _submit_job(
                'merge-%s' % name, merge_sbatch, merge_command, array_job, dry_run,
                dependency_type='afterany',
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
        submitted_any = True

    if resume and not submitted_any:
        print('resume: nothing to do, every rung is already complete')
    elif not dry_run:
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
    parser.add_argument(
        '--resume', action='store_true',
        help='for a scale ladder, skip rungs already complete in the target '
             'run and finish only the incomplete ones; requires --run-id',
    )
    parser.add_argument(
        '--base-run',
        help='stage-1 run tree the simtrain stage builds on (tokenizer, '
             'pretrain checkpoint, packed replay data); overrides '
             'simtrain.base_run_dir and is baked into the resolved config',
    )
    arguments = parser.parse_args()

    if arguments.resume and not arguments.run_id:
        sys.exit('--resume needs --run-id to name the run tree to resume')

    run_id = _resolve_run_id(arguments.run_id, arguments.dry_run)
    config = load_config(arguments.config, run_id=run_id)
    if arguments.base_run:
        config.simtrain.base_run_dir = arguments.base_run
    if arguments.resume and not config.scale.rungs:
        sys.exit(
            '--resume applies to scale ladders; for a single-config run, rerun '
            'stages with --run-id %s --stages <stages>' % run_id
        )
    resolved_path = _materialize_run_config(config)
    print('run id:          %s' % run_id)
    print('output tree:     %s' % config.out_dir)
    print('resolved config: %s' % resolved_path)
    print('rerun later with: --run-id %s' % run_id)
    print()

    if config.scale.rungs:
        submit_world(str(resolved_path), arguments.dry_run, arguments.resume)
        return
    stages = [stage.strip() for stage in arguments.stages.split(',') if stage.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit('unknown stages: %s' % sorted(unknown))
    if 'simtrain' in stages and not config.simtrain.base_run_dir:
        sys.exit(
            'simtrain needs a stage-1 base: pass --base-run '
            'runs/<t1-run-tree> or set simtrain.base_run_dir in the config'
        )
    submit(str(resolved_path), stages, arguments.dry_run)


if __name__ == '__main__':
    main()

"""Submit the pipeline to Slurm as a chain of dependent jobs.

Each stage becomes one sbatch job. Stages are chained with afterok dependencies
so the next stage starts only if the previous one succeeded. Resource requests
come from the slurm section of the config.

When generate.workers is above one, the generate stage becomes a job array of
that many single-GPU workers, each generating a disjoint share of the corpus,
followed by a CPU-only merge job that deduplicates across workers and writes
the final files. Later stages depend on the merge job.

    python slurm/submit.py --config configs/poc.yaml
    python slurm/submit.py --config configs/poc.yaml --stages pretrain,finetune,evaluate
    python slurm/submit.py --config configs/poc.yaml --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

from slm.config import load_config

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
        sbatch = sbatch + ['--dependency', 'afterok:%s' % previous_job]
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


def submit(config_path, stages, dry_run):
    config = load_config(config_path)
    previous_job = None
    for stage in stages:
        sbatch = _sbatch_arguments(
            config, 'slm-%s' % stage, _stage_gres(stage, config)
        )
        command = _stage_command(stage, config, config_path)
        if stage == 'generate' and config.generate.workers > 1:
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
        previous_job = _submit_job(stage, sbatch, command, previous_job, dry_run)
    if not dry_run:
        print('\nAll stages submitted. Track with: squeue -u $USER')


def main():
    parser = argparse.ArgumentParser(description='Submit pipeline to Slurm')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stages', default=','.join(DEFAULT_STAGES))
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()
    stages = [stage.strip() for stage in arguments.stages.split(',') if stage.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit('unknown stages: %s' % sorted(unknown))
    submit(arguments.config, stages, arguments.dry_run)


if __name__ == '__main__':
    main()

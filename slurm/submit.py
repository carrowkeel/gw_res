"""Submit the pipeline to Slurm as a chain of dependent jobs.

Each stage becomes one sbatch job. Stages are chained with afterok dependencies
so the next stage starts only if the previous one succeeded. Resource requests
come from the slurm section of the config.

    python slurm/submit.py --config configs/poc.yaml
    python slurm/submit.py --config configs/poc.yaml --stages pretrain,finetune,evaluate
    python slurm/submit.py --config configs/poc.yaml --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))

from slm.config import load_config

ALL_STAGES = ['generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate']
GPU_STAGES = {'generate', 'pretrain', 'finetune', 'evaluate'}


def _gpu_count(gres):
    if not gres:
        return 0
    match = re.search(r'(\d+)\s*$', gres)
    return int(match.group(1)) if match else 1


def _stage_command(stage, config, config_path):
    base = 'cd %s && PYTHONPATH=src' % REPOSITORY_ROOT
    if stage == 'pretrain':
        gres = config.slurm.pretrain_gres or config.slurm.gres
        gpus = _gpu_count(gres)
        if gpus > 1:
            inner = (
                'torchrun --standalone --nproc_per_node=%d '
                '-m slm.pretrain --config %s' % (gpus, config_path)
            )
        else:
            inner = 'python3 -m slm.pretrain --config %s' % config_path
    elif stage == 'evaluate':
        inner = 'python3 -m slm.evaluate --config %s --stage sft' % config_path
    else:
        inner = 'python3 -m slm.%s --config %s' % (stage, config_path)
    return '%s %s' % (base, inner)


def _sbatch_arguments(stage, config):
    slurm = config.slurm
    arguments = [
        'sbatch',
        '--job-name', 'slm-%s' % stage,
        '--mem', slurm.memory,
        '--cpus-per-task', str(slurm.cpus_per_task),
        '--time', slurm.time_limit,
        '--parsable',
    ]
    if stage in GPU_STAGES:
        gres = (
            slurm.pretrain_gres
            if stage == 'pretrain' and slurm.pretrain_gres
            else slurm.gres
        )
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


def submit(config_path, stages, dry_run):
    config = load_config(config_path)
    previous_job = None
    for stage in stages:
        sbatch = _sbatch_arguments(stage, config)
        if previous_job:
            sbatch += ['--dependency', 'afterok:%s' % previous_job]
        sbatch += ['--wrap', _stage_command(stage, config, config_path)]

        printable = ' '.join(
            ('"%s"' % argument if ' ' in argument else argument)
            for argument in sbatch
        )
        if dry_run:
            print(printable)
            previous_job = '<%s_jobid>' % stage
            continue

        print('submitting: %s' % stage)
        result = subprocess.run(sbatch, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit('sbatch failed for %s: %s' % (stage, result.stderr.strip()))
        job_id = result.stdout.strip().split(';')[0]
        suffix = ' (after %s)' % previous_job if previous_job else ''
        print('  -> job %s%s' % (job_id, suffix))
        previous_job = job_id
    if not dry_run:
        print('\nAll stages submitted. Track with: squeue -u $USER')


def main():
    parser = argparse.ArgumentParser(description='Submit pipeline to Slurm')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stages', default=','.join(ALL_STAGES))
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()
    stages = [stage.strip() for stage in arguments.stages.split(',') if stage.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        sys.exit('unknown stages: %s' % sorted(unknown))
    submit(arguments.config, stages, arguments.dry_run)


if __name__ == '__main__':
    main()

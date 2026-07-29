"""Submit parallel simulation-training experiments to Slurm.

A sweep file names a base config and a list of variants, each a set of
section overrides on that base. Every variant becomes its own run tree
under a shared sweep directory and its own chain of Slurm jobs, all
independent of each other, so eight or sixteen simulation trainings run
side by side instead of one experiment at a time. The merged config of
each variant is validated at submission, so a typo in an override fails
on the login node, not an hour into a queued job.

The sweep file:

    base_config: configs/sim.yaml
    sweep_name: sim-options
    stages: [simtrain]
    common:
      simtrain:
        listener_mode: pattern
    variants:
      - name: baseline
        overrides: {}
      - name: order-clause
        overrides:
          simtrain:
            loss_scope: order_clause

Common overrides apply to every variant, variant overrides win over
common. Each variant writes into <out_root>/<sweep_name>-<run_id>/<name>
and the sweep directory carries a sweep.json manifest that the
comparison report reads:

    python slurm/sweep.py --sweep configs/experiments/sim_options.yaml \
        --base-run runs/<t1-tree> --dry-run
    python -m slm.simreport --sweep runs/sweeps/<sweep_name>-<run_id>
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / 'src'))
sys.path.insert(0, str(REPOSITORY_ROOT / 'slurm'))

from slm.config import load_config

from submit import (
    _command_base, _deep_merge, _sbatch_arguments, _stage_gres, _submit_job,
)

SWEEP_STAGES = ['gate', 'simtrain']
DEFAULT_OUT_ROOT = 'runs/sweeps'


def load_sweep(path):
    with open(path) as handle:
        sweep = yaml.safe_load(handle) or {}
    for key in ('base_config', 'sweep_name', 'variants'):
        if not sweep.get(key):
            sys.exit('sweep file %s is missing %r' % (path, key))
    stages = sweep.get('stages') or ['simtrain']
    unknown = [stage for stage in stages if stage not in SWEEP_STAGES]
    if unknown:
        sys.exit('sweep stages must be a subset of %s, got %s'
                 % (SWEEP_STAGES, unknown))
    sweep['stages'] = stages
    names = [variant.get('name') for variant in sweep['variants']]
    if len(names) != len(set(names)) or not all(names):
        sys.exit('every variant needs a unique non-empty name')
    return sweep


def materialize_variant(base_raw, sweep, variant, sweep_root, base_run):
    """Write a variant's merged config into its run tree and validate it.

    The variant inherits the base config overlaid with the sweep's common
    overrides and then its own. Its out_dir and slurm log_dir are forced
    into the sweep tree, and the command-line base run fills
    simtrain.base_run_dir only where the merged config left it empty, so
    a variant comparing base models keeps its own.
    """
    raw = _deep_merge(base_raw, sweep.get('common') or {})
    raw = _deep_merge(raw, variant.get('overrides') or {})
    variant_out = sweep_root / variant['name']
    project = dict(raw.get('project') or {})
    project['name'] = '%s-%s' % (sweep['sweep_name'], variant['name'])
    project['out_dir'] = str(variant_out)
    raw['project'] = project
    slurm_section = dict(raw.get('slurm') or {})
    slurm_section['log_dir'] = str(sweep_root / 'slurm_logs')
    raw['slurm'] = slurm_section
    simtrain_section = dict(raw.get('simtrain') or {})
    if base_run and not simtrain_section.get('base_run_dir'):
        simtrain_section['base_run_dir'] = base_run
    raw['simtrain'] = simtrain_section
    variant_out.mkdir(parents=True, exist_ok=True)
    path = variant_out / 'config.yaml'
    with open(path, 'w') as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return path, load_config(path)


def submit_sweep(sweep, sweep_root, base_run, only, dry_run):
    with open(sweep['base_config']) as handle:
        base_raw = yaml.safe_load(handle) or {}
    manifest = {
        'sweep_name': sweep['sweep_name'],
        'base_config': sweep['base_config'],
        'stages': sweep['stages'],
        'variants': [],
    }
    for variant in sweep['variants']:
        name = variant['name']
        if only and name not in only:
            continue
        config_path, config = materialize_variant(
            base_raw, sweep, variant, sweep_root, base_run
        )
        if not config.simtrain.base_run_dir:
            sys.exit(
                'variant %s has no simtrain.base_run_dir: pass --base-run '
                'or set it in the sweep file' % name
            )
        previous_job = None
        job_ids = []
        for stage in sweep['stages']:
            sbatch = _sbatch_arguments(
                config, 'slm-%s-%s' % (name, stage),
                _stage_gres(stage, config),
            )
            command = '%s python3 -m slm.%s --config %s' % (
                _command_base(config), stage, config_path
            )
            previous_job = _submit_job(
                '%s-%s' % (name, stage), sbatch, command, previous_job,
                dry_run,
            )
            job_ids.append(str(previous_job))
        manifest['variants'].append({
            'name': name,
            'out_dir': str(sweep_root / name),
            'overrides': variant.get('overrides') or {},
            'jobs': job_ids,
        })
    if not manifest['variants']:
        sys.exit('no variants matched --variants')
    with open(sweep_root / 'sweep.json', 'w') as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Submit a sweep of parallel simulation experiments'
    )
    parser.add_argument('--sweep', required=True)
    parser.add_argument(
        '--base-run',
        help='stage-1 run tree used by variants that do not set their own '
             'simtrain.base_run_dir',
    )
    parser.add_argument(
        '--run-id',
        help='reuse an existing sweep directory; omit for a fresh one',
    )
    parser.add_argument(
        '--variants',
        help='comma-separated variant names to submit, defaulting to all',
    )
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()

    sweep = load_sweep(arguments.sweep)
    run_id = arguments.run_id or (
        'dryrun' if arguments.dry_run else uuid.uuid4().hex[:8]
    )
    out_root = Path(sweep.get('out_root') or DEFAULT_OUT_ROOT)
    sweep_root = out_root / ('%s-%s' % (sweep['sweep_name'], run_id))
    only = None
    if arguments.variants:
        only = {name.strip() for name in arguments.variants.split(',')
                if name.strip()}
    print('sweep:      %s' % sweep['sweep_name'])
    print('sweep tree: %s' % sweep_root)
    print('stages:     %s' % ','.join(sweep['stages']))
    print()
    manifest = submit_sweep(
        sweep, sweep_root, arguments.base_run, only, arguments.dry_run
    )
    print()
    print('%d variants submitted.' % len(manifest['variants']))
    print('Compare with: python -m slm.simreport --sweep %s' % sweep_root)


if __name__ == '__main__':
    main()

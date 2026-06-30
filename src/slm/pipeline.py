"""Local end-to-end orchestrator.

Runs the stages in order, in process, on the current machine. Use this on a
single GPU node or for a smoke test. For queued or multi-node execution use the
Slurm submitter, which runs the same stage entrypoints as dependent jobs.

    python -m slm.pipeline --config configs/poc.yaml
    python -m slm.pipeline --config configs/smoke.yaml --stages tokenizer,data,pretrain
"""

import argparse

from .config import load_config, save_config
from .utils import ensure_directory, get_logger

logger = get_logger('pipeline')

ALL_STAGES = ['generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate']


def run_stage(name, config):
    logger.info('=== stage: %s ===', name)
    if name == 'generate':
        from . import generate

        generate.run(config)
    elif name == 'tokenizer':
        from . import tokenizer

        tokenizer.train(config)
    elif name == 'data':
        from . import data

        data.prepare_pretrain(config)
    elif name == 'pretrain':
        from . import pretrain

        pretrain.run(config)
    elif name == 'finetune':
        from . import finetune

        finetune.run(config)
    elif name == 'evaluate':
        from . import evaluate

        evaluate.run(config, stage='sft')
    else:
        raise ValueError('unknown stage %r' % name)


def run(config, stages):
    ensure_directory(config.out_dir)
    save_config(config, config.out_dir / 'config.resolved.yaml')
    for stage in stages:
        run_stage(stage, config)
    logger.info('pipeline complete, artifacts under %s', config.out_dir)


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline locally')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stages', default=','.join(ALL_STAGES))
    arguments = parser.parse_args()
    stages = [stage.strip() for stage in arguments.stages.split(',') if stage.strip()]
    run(load_config(arguments.config), stages)


if __name__ == '__main__':
    main()

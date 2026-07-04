"""Interactive prompt-and-response loop over a trained checkpoint.

A minimal REPL for probing a built model by hand, complementing slm.sample
(fixed seeds, non-interactive). The pretrain checkpoint is continued in raw
completion style; the sft checkpoint answers in the Question and Answer framing.
Sampling settings are adjustable at runtime with slash commands, and the model
can be switched between stages in place.

    python -m slm.chat --config configs/scale/s1_nano.yaml
    python -m slm.chat --config configs/scale/s1_nano.yaml --stage sft --penalty 1.3
"""

import argparse

from .config import load_config
from .infer import StudentModel
from .utils import get_logger

logger = get_logger('chat')

HELP = """commands:
  /help            show this help
  /stage NAME      switch model: pretrain or sft
  /temp VALUE      set sampling temperature
  /topp VALUE      set nucleus top-p
  /penalty VALUE   set repetition penalty
  /tokens COUNT    set maximum new tokens
  /settings        show current settings
  /exit            leave"""

NUMERIC_COMMANDS = {
    '/temp': ('temperature', float),
    '/topp': ('top_p', float),
    '/penalty': ('penalty', float),
    '/tokens': ('max_new_tokens', int),
}


def _checkpoint_path(config, stage):
    checkpoint_base = config.sft_dir if stage == 'sft' else config.pretrain_dir
    checkpoint_path = checkpoint_base / 'ckpt_last.pt'
    if stage == 'pretrain':
        best = config.pretrain_dir / 'ckpt_best.pt'
        if best.exists():
            checkpoint_path = best
    return checkpoint_path


def _load_student(config, stage, cache):
    """Load and cache the StudentModel for a stage, reusing it across switches."""
    if stage not in cache:
        checkpoint_path = _checkpoint_path(config, stage)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                'no %s checkpoint at %s' % (stage, checkpoint_path)
            )
        logger.info('loading %s checkpoint %s', stage, checkpoint_path)
        cache[stage] = StudentModel(config, checkpoint_path)
    return cache[stage]


def _generate(student, stage, prompt, settings):
    if stage == 'sft':
        return student.respond(
            prompt,
            max_new_tokens=settings['max_new_tokens'],
            temperature=settings['temperature'],
            top_p=settings['top_p'],
            repetition_penalty=settings['penalty'],
        )
    return student.complete(
        prompt,
        max_new_tokens=settings['max_new_tokens'],
        temperature=settings['temperature'],
        top_p=settings['top_p'],
        repetition_penalty=settings['penalty'],
    )


def _describe(stage, settings):
    return 'stage=%s temperature=%s top_p=%s penalty=%s tokens=%s' % (
        stage, settings['temperature'], settings['top_p'],
        settings['penalty'], settings['max_new_tokens'],
    )


def run(config, stage, settings):
    cache = {}
    student = _load_student(config, stage, cache)
    print('interactive %s model. /help for commands, /exit to leave.' % stage)
    while True:
        try:
            line = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ('/exit', '/quit'):
            break
        if line == '/help':
            print(HELP)
            continue
        if line == '/settings':
            print(_describe(stage, settings))
            continue
        if line.startswith('/'):
            parts = line.split()
            name = parts[0]
            value = parts[1] if len(parts) > 1 else None
            if name == '/stage' and value in ('pretrain', 'sft'):
                try:
                    student = _load_student(config, value, cache)
                    stage = value
                except FileNotFoundError as error:
                    print(error)
                continue
            if name in NUMERIC_COMMANDS and value is not None:
                key, caster = NUMERIC_COMMANDS[name]
                try:
                    settings[key] = caster(value)
                except ValueError:
                    print('invalid value for %s: %s' % (name, value))
                continue
            print('unknown command %r; /help for commands' % line)
            continue
        print(_generate(student, stage, line, settings))


def main():
    parser = argparse.ArgumentParser(description='Interactive model prompt loop')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', default='pretrain', choices=['pretrain', 'sft'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--penalty', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=120)
    arguments = parser.parse_args()
    settings = {
        'temperature': arguments.temperature,
        'top_p': arguments.top_p,
        'penalty': arguments.penalty,
        'max_new_tokens': arguments.max_new_tokens,
    }
    run(load_config(arguments.config), arguments.stage, settings)


if __name__ == '__main__':
    main()

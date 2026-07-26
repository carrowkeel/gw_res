"""Interactive prompt-and-response loop over a trained checkpoint.

A minimal REPL for probing a built model by hand, complementing slm.sample
(fixed seeds, non-interactive). Every stage that writes a checkpoint can be
loaded: pretrain, the legacy sft stage, the bridging SFT stages, and the
simulator. Only the legacy sft stage uses the Question and Answer framing;
the bridging stages and the simulator were trained on plain text (a dialogue
ending at a speaker cue, a rendered market block), so their input is
continued verbatim. Sampling settings are adjustable at runtime with slash
commands, and the model can be switched between stages in place.

    python -m slm.chat --config runs/world/pico/config.yaml
    python -m slm.chat --config runs/t1_full-<id>/config.resolved.yaml \\
      --stage mathsft
"""

import argparse

from .config import load_config
from .infer import StudentModel
from .sftstage import resolve_checkpoint
from .utils import get_logger

logger = get_logger('chat')

STAGE_DIRECTORIES = {
    'pretrain': 'pretrain_dir',
    'sft': 'sft_dir',
    'commsft': 'commsft_dir',
    'mathsft': 'mathsft_dir',
    'simtrain': 'simtrain_dir',
}

STAGE_HINTS = {
    'pretrain': 'raw continuation: type any text and it is continued',
    'sft': 'instruction framing: type a question',
    'commsft': 'dialogue continuation: end with a speaker cue, for example '
               '"Renn: How many crates arrived?\\nSela:"',
    'mathsft': 'dialogue continuation: end with a speaker cue, for example '
               '"Renn: The first cart brought 12.\\nSela: The second '
               'brought 7.\\nRenn: How many together?\\nSela:"',
    'simtrain': 'raw continuation of a rendered market block, ending at the '
                'trader cue',
}

HELP = """commands:
  /help            show this help
  /stage NAME      switch model: %s
  /temp VALUE      set sampling temperature
  /topp VALUE      set nucleus top-p
  /penalty VALUE   set repetition penalty
  /tokens COUNT    set maximum new tokens
  /settings        show current settings
  /exit            leave

a literal \\n in the input starts a new line, so multi-turn dialogue
prompts can be entered on one line""" % ', '.join(sorted(STAGE_DIRECTORIES))

NUMERIC_COMMANDS = {
    '/temp': ('temperature', float),
    '/topp': ('top_p', float),
    '/penalty': ('penalty', float),
    '/tokens': ('max_new_tokens', int),
}


def _checkpoint_path(config, stage):
    """Return the stage's best checkpoint, falling back to its last."""
    directory = getattr(config, STAGE_DIRECTORIES[stage])
    return resolve_checkpoint(directory), directory


def _tokenizer_path(config, stage):
    """Return the tokenizer that built a stage's checkpoints.

    The simulator trains into its own run tree but tokenizes with the base
    run's artifact, so its checkpoints are only readable under that one.
    """
    base = config.simtrain.base_run_dir
    if stage == 'simtrain' and base:
        from pathlib import Path

        return Path(base) / 'tokenizer' / 'tokenizer.json'
    return config.tokenizer_path


def _load_student(config, stage, cache):
    """Load and cache the StudentModel for a stage, reusing it across switches."""
    if stage not in cache:
        checkpoint_path, directory = _checkpoint_path(config, stage)
        if checkpoint_path is None:
            raise FileNotFoundError('no %s checkpoint under %s' % (
                stage, directory
            ))
        logger.info('loading %s checkpoint %s', stage, checkpoint_path)
        cache[stage] = StudentModel(
            config, checkpoint_path,
            tokenizer_path=_tokenizer_path(config, stage),
        )
    return cache[stage]


def _generate(student, stage, prompt, settings):
    prompt = prompt.replace('\\n', '\n')
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
    print(STAGE_HINTS[stage])
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
            if name == '/stage' and value in STAGE_DIRECTORIES:
                try:
                    student = _load_student(config, value, cache)
                    stage = value
                    print(STAGE_HINTS[stage])
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
    parser.add_argument(
        '--stage', default='pretrain', choices=sorted(STAGE_DIRECTORIES)
    )
    parser.add_argument(
        '--base-run',
        help='run tree holding the tokenizer for a simtrain checkpoint; '
             'overrides simtrain.base_run_dir',
    )
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
    config = load_config(arguments.config)
    if arguments.base_run:
        config.simtrain.base_run_dir = arguments.base_run
    run(config, arguments.stage, settings)


if __name__ == '__main__':
    main()

"""The math game: arithmetic dialogues graded exactly by the program.

The first single-turn game, built directly on the math SFT's task
authors so the model starts on familiar ground: the same dialogue
format, the same operation kinds, the same exact program-derived
answers. What changes is the data source - the model answers and the
program grades, so training pairs come from the model's own verified
play rather than from a teacher.

The experiment tree lives in the parameters, each branch changing one
thing:

- maximum_value and kinds set difficulty on the trunk.
- distractor_lines inserts extra numbered statements about unrelated
  items, so solving requires attending to the relevant quantities and
  ignoring the rest - the ability the market game assumes.
- score_noise_sigma corrupts the grade with Gaussian noise. At zero a
  sample's training weight is exactly its correctness; above zero the
  weight is the noisy score, and sweeping sigma measures how much
  selection noise verified self-improvement survives - the signal-to-
  noise threshold any outcome-driven game must be designed above.
"""

import random

from .. import mathsft
from ..sfteval import numbers_correct

DISTRACTOR_TEMPLATES = [
    'The %s shelf holds %d %s.',
    'Someone left %d %s by the door.',
    'The old ledger lists %d %s from last month.',
    'A neighbor asked about the %d %s in the yard.',
    'There are still %d %s waiting at the dock.',
]

DISTRACTOR_ADJECTIVES = ['spare', 'unsorted', 'borrowed', 'labeled']

DEFAULT_PARAMETERS = {
    'maximum_value': 200,
    'kinds': sorted(mathsft._KIND_BUILDERS),
    'distractor_lines': 0,
    'score_noise_sigma': 0.0,
}


def resolve_parameters(parameters):
    """Overlay game parameters on the defaults, strict on typos."""
    resolved = dict(DEFAULT_PARAMETERS)
    unknown = set(parameters or {}) - set(resolved)
    if unknown:
        raise ValueError('unknown math game parameters: %s'
                         % sorted(unknown))
    resolved.update(parameters or {})
    unknown_kinds = set(resolved['kinds']) - set(mathsft._KIND_BUILDERS)
    if unknown_kinds:
        raise ValueError('unknown math kinds: %s' % sorted(unknown_kinds))
    return resolved


def _distractor_line(random_generator, maximum_value, used_item, speaker):
    item = random_generator.choice(mathsft.ITEMS)
    while item == used_item:
        item = random_generator.choice(mathsft.ITEMS)
    template = random_generator.choice(DISTRACTOR_TEMPLATES)
    value = random_generator.randint(2, max(3, maximum_value))
    if template.count('%') == 3:
        adjective = random_generator.choice(DISTRACTOR_ADJECTIVES)
        text = template % (adjective, value, item)
    else:
        text = template % (value, item)
    return '%s: %s' % (speaker, text)


def build_task(parameters, random_generator):
    """Author one task: a dialogue prompt, an exact answer, its kind.

    Mirrors mathsft.build_dialogue, with two additions: the operation
    kind is drawn from the game's kind list, and distractor_lines
    numbered statements about other items are woven between the
    operative statements, so the relevant quantities must be selected,
    not just found.
    """
    kind = random_generator.choice(sorted(parameters['kinds']))
    statement_one, statement_two, question, answer = (
        mathsft._KIND_BUILDERS[kind](
            random_generator, parameters['maximum_value']
        )
    )
    used_item = None
    for item in mathsft.ITEMS:
        if item in statement_one:
            used_item = item
            break
    speaker_a = mathsft.seeds.invented_name(random_generator)
    speaker_b = mathsft.seeds.invented_name(random_generator)
    while speaker_b == speaker_a:
        speaker_b = mathsft.seeds.invented_name(random_generator)
    speakers = [speaker_a, speaker_b]
    lines = ['%s: %s' % (speaker_a, statement_one)]
    for _ in range(parameters['distractor_lines']):
        lines.append(_distractor_line(
            random_generator, parameters['maximum_value'], used_item,
            random_generator.choice(speakers),
        ))
    lines.append('%s: %s' % (random_generator.choice(speakers),
                             statement_two))
    if parameters['distractor_lines'] and random_generator.random() < 0.5:
        lines.append(_distractor_line(
            random_generator, parameters['maximum_value'], used_item,
            random_generator.choice(speakers),
        ))
    lines.append('%s: %s' % (speaker_b, question))
    return {
        'prompt': '\n'.join(lines) + '\n%s:' % speaker_a,
        'answer': answer,
        'kind': kind,
    }


def generate_tasks(parameters, random_generator, count):
    return [build_task(parameters, random_generator) for _ in range(count)]


def verify(task, response):
    """Exact correctness: every reference number appears in the response."""
    return bool(numbers_correct(response, task['answer']))


def score(task, response, parameters, random_generator):
    """The sample's training weight: correctness, plus selection noise.

    At sigma zero this is the exact gate (weight one when correct, zero
    otherwise). Above zero the grade the trainer sees is corrupted, so
    wrong answers are sometimes kept and right ones sometimes dropped -
    the controlled version of what an outcome-scored simulator does to
    its training signal.
    """
    value = 1.0 if verify(task, response) else 0.0
    sigma = parameters['score_noise_sigma']
    if sigma > 0:
        value += random_generator.gauss(0.0, sigma)
    return max(0.0, value)

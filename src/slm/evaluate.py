"""Stage five: evaluate the trained model with an existing model as judge.

The primary measures run on the base pretrained model, because that is the
product at this scale. Completions from in-distribution seeds are judged for
fluency and for how free they are of real-world referents, and in-world
instructions are judged for coherence and for whether the answer follows the
request (the model learns to follow instructions during pretraining through the
mixed-in Question and Answer text). A small real-world knowledge probe is kept
but demoted: a tiny model answers such out-of-distribution questions poorly, so
those scores are unreliable and labelled as such.

    python -m slm.evaluate --config runs/world/pico/config.yaml
"""

import argparse
import json
import random
import re

from .config import load_config
from .infer import StudentModel
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('evaluate')

COMPLETION_SEEDS = [
    'The wood stands on the higher ground, and',
    'In one of the valleys there is',
    'The near bank had given way since',
    'Beyond the ridge the ground',
    'There is a pool at the foot of the slope, and',
    'The mist lay over the marsh until',
    'A path ran along the edge of the water, and',
    'Once the frost had gone,',
]

IN_WORLD_INSTRUCTIONS = [
    'Describe how a hill and a stream below it are arranged.',
    'Explain how a marsh changes after heavy water.',
    'Describe a wood on higher ground and the open land below it.',
    'Tell what happens to a bank when the water rises.',
    'Compare a shallow pool and a deeper one nearby.',
    'Describe a ridge and the ground that falls away beyond it.',
]

PROBE_QUESTIONS = [
    'What is the capital of France?',
    'Who was the first president of the United States?',
    'How many days are in a week?',
    'What is two plus two?',
    'In what year did the war end?',
    'What is the largest planet?',
    'Who wrote the play about two feuding families?',
    'What is the chemical symbol for water?',
]


def _keyword_pattern(keyword):
    tokens = re.findall(r'[a-z]+', keyword.lower())
    return r'[ _]+'.join(tokens)


def _extract_score(text, keyword, low=1.0, high=10.0):
    """Pull a numeric score for keyword from a possibly verbose judge reply."""
    lowered = text.lower()
    labeled = re.search(
        _keyword_pattern(keyword) + r'[^0-9]{0,20}(\d+(?:\.\d+)?)', lowered
    )
    if labeled:
        value = float(labeled.group(1))
        if low <= value <= high:
            return value
    out_of_ten = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(?:10|ten)', lowered
    )
    if out_of_ten:
        value = float(out_of_ten.group(1))
        if low <= value <= high:
            return value
    for token in re.findall(r'\d+(?:\.\d+)?', lowered):
        value = float(token)
        if low <= value <= high:
            return value
    return None


def _mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _load_judge(config):
    from vllm import LLM, SamplingParams

    model_name = config.eval.judge_model or config.generate.default_model
    engine = LLM(
        model=model_name,
        gpu_memory_utilization=config.eval.judge_gpu_memory_utilization,
        max_model_len=config.generate.max_model_len,
        dtype=config.generate.dtype,
    )
    sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=300)
    return engine, sampling, model_name


def _judge(engine, sampling, system_prompt, user_prompt):
    tokenizer = engine.get_tokenizer()
    rendered = tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    outputs = engine.generate([rendered], sampling)
    return outputs[0].outputs[0].text.strip()


def _pick(items, count, random_generator):
    return [random_generator.choice(items) for _ in range(count)]


def score_completions(config, student, engine, sampling):
    random_generator = random.Random(config.project.seed)
    eval_config = config.eval
    system_prompt = (
        'You grade text produced by a small language model trained only on '
        'imaginary, referent-free English. First decide if the sample is '
        'well-formed English; if it is garbled or not real sentences, grammar '
        'and coherence must be one or two. Rate three axes from one to ten: '
        'grammar, coherence, and referent_free (how free the text is of any '
        'real-world referent, fact, name, place, date, direction, or specific '
        'species; ten means fully generic and imaginary). Reply with exactly '
        'three lines:\ngrammar: <n>\ncoherence: <n>\nreferent_free: <n>'
    )
    samples = []
    aggregate = {'grammar': [], 'coherence': [], 'referent_free': []}
    seeds = _pick(
        COMPLETION_SEEDS, eval_config.number_of_generation_samples,
        random_generator,
    )
    for seed in seeds:
        text = student.complete(
            seed, eval_config.max_new_tokens, eval_config.temperature,
            eval_config.top_p, eval_config.repetition_penalty,
        )
        output = (seed + ' ' + text).strip()
        verdict = _judge(engine, sampling, system_prompt, 'SAMPLE:\n' + output)
        scores = {axis: _extract_score(verdict, axis) for axis in aggregate}
        for axis in aggregate:
            aggregate[axis].append(scores[axis])
        samples.append({'seed': seed, 'output': output, 'scores': scores})
    means = {axis: _mean(values) for axis, values in aggregate.items()}
    return {'samples': samples, 'means': means}


def score_instructions(config, student, engine, sampling):
    random_generator = random.Random(config.project.seed + 1)
    eval_config = config.eval
    system_prompt = (
        'A small language model was asked an INSTRUCTION and gave an ANSWER, in '
        'an imaginary world with no real-world referents. Rate two axes from '
        'one to ten: coherence (is the answer well-formed, sensible English?) '
        'and followed (does it address the instruction?). Reply with exactly '
        'two lines:\ncoherence: <n>\nfollowed: <n>'
    )
    count = max(1, eval_config.number_of_generation_samples // 2)
    instructions = _pick(IN_WORLD_INSTRUCTIONS, count, random_generator)
    results = []
    aggregate = {'coherence': [], 'followed': []}
    for instruction in instructions:
        answer = student.respond(
            instruction, eval_config.max_new_tokens, eval_config.temperature,
            eval_config.top_p, eval_config.repetition_penalty,
        )
        verdict = _judge(
            engine, sampling, system_prompt,
            'INSTRUCTION: %s\nANSWER: %s' % (instruction, answer),
        )
        scores = {axis: _extract_score(verdict, axis) for axis in aggregate}
        for axis in aggregate:
            aggregate[axis].append(scores[axis])
        results.append(
            {'instruction': instruction, 'answer': answer, 'scores': scores}
        )
    means = {axis: _mean(values) for axis, values in aggregate.items()}
    return {'results': results, 'means': means}


def knowledge_probe(config, student, engine, sampling):
    eval_config = config.eval
    random_generator = random.Random(config.project.seed + 2)
    questions = _pick(
        PROBE_QUESTIONS, eval_config.number_of_probe_questions, random_generator
    )
    system_prompt = (
        'A small model was built to have no real-world knowledge. Given a '
        'factual QUESTION and its ANSWER, score from one to ten how free of '
        'real facts the answer is: ten reveals no correct real fact, one states '
        'the real fact. An empty or nonsensical answer reveals no fact and '
        'scores ten. Reply with exactly one line:\nreferent_free: <n>'
    )
    results = []
    scores = []
    for question in questions:
        answer = student.respond(
            question, max_new_tokens=48, temperature=0.7,
            repetition_penalty=eval_config.repetition_penalty,
        )
        if not answer.strip():
            score = 10.0
        else:
            verdict = _judge(
                engine, sampling, system_prompt,
                'QUESTION: %s\nANSWER: %s' % (question, answer),
            )
            score = _extract_score(verdict, 'referent_free')
        results.append(
            {'question': question, 'answer': answer, 'referent_free': score}
        )
        if score is not None:
            scores.append(score)
    return {'results': results, 'mean_referent_free': _mean(scores)}


def write_report(config, report):
    stage = report['stage']
    output_directory = ensure_directory(config.eval_dir)
    (output_directory / ('report_%s.json' % stage)).write_text(
        json.dumps(report, indent=2)
    )

    completion_means = report['completions']['means']
    instruction_means = report['instructions']['means']
    lines = [
        '# Evaluation report: %s' % config.project.name,
        '',
        '> Stage: %s. Judge model: %s. The completion and instruction scores '
        'are the primary measures; the knowledge probe is out-of-distribution '
        'for a small model and unreliable below larger scales.'
        % (report['stage'], report['judge_model']),
        '',
        '## Completions (base model, one to ten)',
        '- grammar: %s' % completion_means.get('grammar'),
        '- coherence: %s' % completion_means.get('coherence'),
        '- referent_free: %s' % completion_means.get('referent_free'),
        '',
        '### Sample completions',
    ]
    for sample in report['completions']['samples'][:6]:
        lines.append('- %r' % sample['output'])
    lines += [
        '',
        '## Instruction following (one to ten)',
        '- coherence: %s' % instruction_means.get('coherence'),
        '- followed: %s' % instruction_means.get('followed'),
        '',
        '### Sample instruction answers',
    ]
    for result in report['instructions']['results'][:6]:
        lines.append(
            '- Q: %s\n  A: %r' % (result['instruction'], result['answer'])
        )
    lines += [
        '',
        '## Knowledge probe (demoted, out-of-distribution)',
        '- mean referent_free: %s' % report['probe']['mean_referent_free'],
    ]
    report_path = output_directory / ('report_%s.md' % stage)
    report_path.write_text('\n'.join(lines))
    logger.info('wrote evaluation report to %s', report_path)


def _find_checkpoint(base):
    for name in ('ckpt_best.pt', 'ckpt_last.pt'):
        if (base / name).exists():
            return base / name
    return None


def _sft_targets(config):
    """Return (report label, checkpoint directory) for each finetune variant."""
    variants = config.finetune.variants
    if not variants:
        return [('sft', config.sft_dir)]
    return [
        ('sft_%s' % variant['name'], config.sft_dir / variant['name'])
        for variant in variants
    ]


def run(config, stage='pretrain', checkpoint_dir=None):
    set_seed(config.project.seed)
    if checkpoint_dir is None:
        checkpoint_dir = (
            config.pretrain_dir if stage == 'pretrain' else config.sft_dir
        )
    checkpoint_path = _find_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        logger.info('no %s checkpoint found, skipping', stage)
        return None
    logger.info('loading %s checkpoint %s', stage, checkpoint_path)
    student = StudentModel(config, checkpoint_path)
    engine, sampling, judge_model = _load_judge(config)

    report = {
        'stage': stage,
        'judge_model': judge_model,
        'completions': score_completions(config, student, engine, sampling),
        'instructions': score_instructions(config, student, engine, sampling),
        'probe': knowledge_probe(config, student, engine, sampling),
    }
    write_report(config, report)
    return report


def run_all(config):
    """Evaluate the pretrained model and every finetune variant when present."""
    reports = {}
    report = run(config, 'pretrain', config.pretrain_dir)
    if report is not None:
        reports['pretrain'] = report
    for label, checkpoint_dir in _sft_targets(config):
        report = run(config, label, checkpoint_dir)
        if report is not None:
            reports[label] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained model')
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--stage', default='both', choices=['pretrain', 'sft', 'both']
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.stage == 'pretrain':
        run(config, 'pretrain', config.pretrain_dir)
    elif arguments.stage == 'sft':
        for label, checkpoint_dir in _sft_targets(config):
            run(config, label, checkpoint_dir)
    else:
        run_all(config)


if __name__ == '__main__':
    main()

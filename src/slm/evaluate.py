"""Stage five: evaluate the trained model with an existing model as judge.

Three views, all driven by an instruct model served through vLLM: quality
scoring of generated samples, a knowledge probe that confirms the model cannot
supply real facts, and a model-queries-model interrogation. Results are written
as a machine-readable and a human-readable report.

Score parsing is tolerant of verbose judge replies, and the quality rubric
forces low grades for text that is not well-formed English so that fluent-
looking nonsense cannot score high. The judge should be a capable model; small
judge models produce unreliable scores, which the report states explicitly.

    python -m slm.evaluate --config configs/poc.yaml --stage sft
"""

import argparse
import json
import random
import re

from .config import load_config
from .infer import StudentModel
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('evaluate')

_CONTINUATION_SEEDS = [
    'In one of the valleys there is a lake, and',
    'The path turned where the ground began to rise, and',
    'The far slope stood higher than the near one, and',
    'Two of them waited at the edge of the clearing while',
    'A is any region that lies beside a B, and when',
]

_INSTRUCTION_PROMPTS = [
    'Describe how a lake and a nearby hill are arranged, using only relations.',
    'Define an invented term for a kind of passage through its relations.',
    'Continue a short serious exchange between two speakers about which way is shorter.',
    'Write a plain description of a clearing and what lies beside it.',
    'Explain how two areas come to be arranged as they are.',
]

_PROBE_QUESTIONS = [
    'What is the capital of France?',
    'Who was the first president of the United States?',
    'How many days are in a week?',
    'What is two plus two?',
    'In what year did the war end?',
    'What is the largest planet?',
    'Who wrote the play about two feuding families?',
    'What is the chemical symbol for water?',
    'What currency is used across the eastern islands?',
    'How fast does light travel?',
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


def _load_judge(config):
    from vllm import LLM, SamplingParams

    model_name = config.eval.judge_model or config.generate.default_model
    engine = LLM(
        model=model_name,
        gpu_memory_utilization=config.eval.judge_gpu_memory_utilization,
        max_model_len=config.generate.max_model_len,
        dtype=config.generate.dtype,
    )
    sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=400)
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


def score_quality(config, student, engine, sampling):
    random_generator = random.Random(config.project.seed)
    eval_config = config.eval
    samples = []
    for index in range(eval_config.number_of_generation_samples):
        if index % 2 == 0:
            seed_text = random_generator.choice(_CONTINUATION_SEEDS)
            continuation = student.complete(
                seed_text, eval_config.max_new_tokens,
                eval_config.temperature, eval_config.top_p,
            )
            samples.append({
                'kind': 'continuation',
                'prompt': seed_text,
                'output': seed_text + ' ' + continuation,
            })
        else:
            instruction = random_generator.choice(_INSTRUCTION_PROMPTS)
            response = student.respond(
                instruction, eval_config.max_new_tokens,
                eval_config.temperature, eval_config.top_p,
            )
            samples.append({
                'kind': 'response',
                'prompt': instruction,
                'output': response,
            })

    system_prompt = (
        'You grade text written by a small language model trained only on '
        'serious imaginary text with no real-world referents. First judge '
        'whether the sample is well-formed, coherent English. If it is not '
        'clearly well-formed English, for example it is garbled, random, or not '
        'made of real sentences, then grammar and coherence must be one or two '
        'and creativity at most three. Only genuinely fluent text may score '
        'above five. Rate each axis from one to ten and reply with exactly '
        'three lines:\ngrammar: <n>\ncoherence: <n>\ncreativity: <n>'
    )
    aggregate = {'grammar': [], 'coherence': [], 'creativity': []}
    for sample in samples:
        verdict = _judge(
            engine, sampling, system_prompt, 'SAMPLE:\n' + sample['output']
        )
        scores = {
            axis: _extract_score(verdict, axis) for axis in aggregate
        }
        sample['scores'] = scores
        for axis in aggregate:
            if scores[axis] is not None:
                aggregate[axis].append(scores[axis])
    means = {
        axis: (sum(values) / len(values) if values else None)
        for axis, values in aggregate.items()
    }
    return {'samples': samples, 'means': means}


def knowledge_probe(config, student, engine, sampling):
    eval_config = config.eval
    repeated = _PROBE_QUESTIONS * (
        eval_config.number_of_probe_questions // len(_PROBE_QUESTIONS) + 1
    )
    questions = repeated[:eval_config.number_of_probe_questions]
    system_prompt = (
        'A small language model was deliberately trained with no real-world '
        'referents. I give you a factual question and the model answer. Score '
        'from one to ten how free of real facts the answer is: ten means it '
        'reveals no correct real fact, one means it states the real fact. An '
        'empty or nonsensical answer reveals no fact and should score ten. '
        'Reply with exactly one line:\nreferent_free: <n>'
    )
    results = []
    scores = []
    for question in questions:
        answer = student.respond(question, max_new_tokens=64, temperature=0.7)
        if not answer.strip():
            score = 10.0
        else:
            verdict = _judge(
                engine, sampling, system_prompt,
                'QUESTION: %s\nANSWER: %s' % (question, answer),
            )
            score = _extract_score(verdict, 'referent_free')
        results.append({
            'question': question,
            'answer': answer,
            'referent_free': score,
        })
        if score is not None:
            scores.append(score)
    mean = sum(scores) / len(scores) if scores else None
    return {'results': results, 'mean_referent_free': mean, 'scored': len(scores)}


def interrogate(config, student, engine, sampling, rounds=8):
    question_system = (
        'You are interviewing a small language model that knows only a serious '
        'imaginary world with no real referents. Ask one short question to test '
        'its language ability. Output only the question.'
    )
    verdict_system = (
        'Give a one-paragraph qualitative verdict on this transcript of an '
        'interview with a small language model. Comment on its fluency and '
        'whether it correctly shows no real-world referents. Do not invent '
        'abilities the transcript does not show.'
    )
    transcript = []
    for _ in range(rounds):
        question = _judge(
            engine, sampling, question_system, 'Ask your next question.'
        ).strip().split('\n')[0]
        answer = student.respond(question, max_new_tokens=80, temperature=0.8)
        transcript.append({'question': question, 'answer': answer})
    conversation = '\n'.join(
        'Q: %s\nA: %s' % (turn['question'], turn['answer'])
        for turn in transcript
    )
    verdict = _judge(engine, sampling, verdict_system, conversation)
    return {'transcript': transcript, 'verdict': verdict}


def write_report(config, report):
    output_directory = ensure_directory(config.eval_dir)
    (output_directory / 'report.json').write_text(json.dumps(report, indent=2))

    means = report['quality']['means']
    probe = report['probe']
    lines = [
        '# Evaluation report: %s' % config.project.name,
        '',
        '> Judge model: %s. Scores are only meaningful with a capable judge; '
        'small judge models are unreliable and may rate nonsense highly.'
        % report['judge_model'],
        '',
        '## Quality (judge scores, one to ten)',
        '- grammar: %s' % means.get('grammar'),
        '- coherence: %s' % means.get('coherence'),
        '- creativity: %s' % means.get('creativity'),
        '',
        '## Referent-free probe (one to ten, higher means fewer real facts)',
        '- mean referent_free: %s (scored %s of %s)'
        % (probe['mean_referent_free'], probe['scored'], len(probe['results'])),
        '',
        '### Sample probe answers',
    ]
    for result in probe['results'][:8]:
        lines.append(
            '- Q: %s | A: %r (score %s)'
            % (result['question'], result['answer'], result['referent_free'])
        )
    lines += ['', '## Interrogation verdict', '',
              report['interrogation']['verdict'], '', '### Sample exchanges']
    for turn in report['interrogation']['transcript'][:6]:
        lines.append('- Q: %s | A: %r' % (turn['question'], turn['answer']))
    (output_directory / 'report.md').write_text('\n'.join(lines))
    logger.info('wrote evaluation report to %s', output_directory / 'report.md')


def run(config, stage='sft'):
    set_seed(config.project.seed)
    checkpoint_base = config.sft_dir if stage == 'sft' else config.pretrain_dir
    checkpoint_path = checkpoint_base / 'ckpt_last.pt'
    if stage == 'pretrain' and not checkpoint_path.exists():
        checkpoint_path = config.pretrain_dir / 'ckpt_best.pt'
    logger.info('loading student checkpoint %s', checkpoint_path)
    student = StudentModel(config, checkpoint_path)
    engine, sampling, judge_model = _load_judge(config)

    report = {
        'stage': stage,
        'judge_model': judge_model,
        'quality': score_quality(config, student, engine, sampling),
        'probe': knowledge_probe(config, student, engine, sampling),
        'interrogation': interrogate(config, student, engine, sampling),
    }
    write_report(config, report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate the trained model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', default='sft', choices=['pretrain', 'sft'])
    arguments = parser.parse_args()
    run(load_config(arguments.config), arguments.stage)


if __name__ == '__main__':
    main()

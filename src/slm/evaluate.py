"""Stage five: evaluate the trained model, matched to the grounded corpus.

Generation is grounded: every document and pair is anchored in program-known
facts from worldgen (node 46), so the headline evaluation supplies facts the
same way and scores by exact match against program-derived answers, per task
kind (retrieval, comparison, multihop, notstated, and the puzzle kinds). The
notstated sub-score measures whether the model declines when the facts do not
contain the answer or fabricates one in perfect form. Completions are judged
on grammar and coherence over a mix of grounded and generic seeds.

The earlier real-world instruction set and factual accuracy probe remain, but
demoted to an explicitly labeled out-of-distribution generalization section:
the grounded corpus deliberately contains no such knowledge, so low scores
there are expected and the accuracy probe reads as a contamination gauge (how
much real-world fact leaked from the generator), not as a target.

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

GENERIC_COMPLETION_SEEDS = [
    'The clerk had covered for his colleague twice that month, and',
    'By the time the loan came due, the interest had',
    'The recipe called for the dough to rest until',
    'When the two committees finally met, the first disagreement was over',
    'Doctors now agree that the most common cause of the condition is',
    'The bridge had been closed for repair since',
    'After the match, the coach explained that the turning point was',
    'The old trade route between the two cities was abandoned once',
    'A well-kept engine loses most of its power when',
    'The negotiation stalled because neither side would',
    'Farmers in the region rotate their crops every year because',
    'The newest exhibit at the museum traces how the technique of',
    'She had promised to pay back the debt by',
    'The court ruled that the contract was invalid because',
    'What began as a small workshop grew, within a decade, into',
    'The experiment failed the first time because the temperature',
]


def _completion_seeds(config, random_generator):
    """Return completion seeds, half grounded in worldgen and half generic.

    Grounded seeds are openings cut from rendered world documents, so they sit
    inside the corpus distribution; the generic half probes how the model
    handles material from outside its world.
    """
    from . import worldgen

    grounded = []
    for document in worldgen.world_documents(
        config.project.seed + 4, len(GENERIC_COMPLETION_SEEDS)
    ):
        words = document.split()
        cut = random_generator.randint(8, min(14, max(9, len(words) - 1)))
        grounded.append(' '.join(words[:cut]))
    return grounded + list(GENERIC_COMPLETION_SEEDS)


TASK_INSTRUCTIONS = [
    'Explain how compound interest works.',
    'Explain why bread dough needs to rest before baking.',
    'Give step-by-step instructions for changing a bicycle tire.',
    'Give step-by-step instructions for setting up a new email account.',
    'Compare renting an apartment to buying one.',
    'Compare a hand plane and a power sander for finishing wood.',
    'Define the word "collateral" and give an example.',
    'Define the word "tributary" and give an example.',
    'What is the boiling point of water at sea level?',
    'What causes the seasons to change?',
    'A friend is deciding between two job offers with different pay and '
    'commute times. Give them advice on how to decide.',
    'Summarize why a company might choose to lease equipment instead of '
    'buying it.',
    'Rewrite this sentence more clearly: "The thing that was done by the '
    'team was not really finished on the time that it was supposed to be."',
    'List three things to check before signing a rental agreement, with a '
    'reason for each.',
    'A shop sells an item for more than it costs to make, but still loses '
    'money overall. Explain how that can happen.',
    'Two trains leave different stations at different times and speeds, '
    'travelling toward each other. Explain how to find when they meet.',
]

PROBE_QUESTIONS = [
    'What is the capital of France?',
    'Who was the first president of the United States?',
    'How many days are in a week?',
    'What is two plus two?',
    'In what year did the Second World War end?',
    'What is the largest planet in the solar system?',
    'Who wrote the play about two feuding families, Romeo and Juliet?',
    'What is the chemical symbol for water?',
    'What is the freezing point of water in Celsius?',
    'How many sides does a hexagon have?',
    'What organ pumps blood through the body?',
    'What is the currency used in Japan?',
    'Who painted the Mona Lisa?',
    'What is the tallest mountain in the world?',
    'How many players are on a standard soccer team on the field at once?',
    'What is the capital of Italy?',
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
        'You grade text produced by a small language model. First decide if '
        'the sample is well-formed English; if it is garbled or not real '
        'sentences, grammar and coherence must be one or two. Rate two axes '
        'from one to ten: grammar (correct, varied English) and coherence '
        '(does the continuation make sense and follow from the seed, staying '
        'on topic and internally consistent). Reply with exactly two lines:\n'
        'grammar: <n>\ncoherence: <n>'
    )
    samples = []
    aggregate = {'grammar': [], 'coherence': []}
    seeds = _pick(
        _completion_seeds(config, random_generator),
        eval_config.number_of_generation_samples,
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
        'A small language model was asked an INSTRUCTION and gave an ANSWER. '
        'Rate two axes from one to ten: coherence (is the answer well-formed, '
        'sensible English?) and followed (does it address the instruction, '
        'and is it correct where the instruction has a factual or practical '
        'answer?). Reply with exactly two lines:\ncoherence: <n>\nfollowed: <n>'
    )
    count = max(1, eval_config.number_of_generation_samples // 2)
    instructions = _pick(TASK_INSTRUCTIONS, count, random_generator)
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


def accuracy_probe(config, student, engine, sampling):
    """Score factual correctness on fixed real-world questions.

    The grounded corpus deliberately contains no real-world knowledge, so
    this is a contamination gauge, not a target: a score near one means
    essentially nothing leaked from the generator model, and movement upward
    signals leakage (or, at scale, transfer) rather than success or failure
    of training.
    """
    eval_config = config.eval
    random_generator = random.Random(config.project.seed + 2)
    questions = _pick(
        PROBE_QUESTIONS, eval_config.number_of_probe_questions, random_generator
    )
    system_prompt = (
        'Given a factual QUESTION and its ANSWER, score from one to ten how '
        'factually correct the answer is: ten is exactly correct, one is '
        'wrong or empty. Partial or vague but not incorrect answers score in '
        'the middle. Reply with exactly one line:\naccuracy: <n>'
    )
    results = []
    scores = []
    for question in questions:
        answer = student.respond(
            question, max_new_tokens=48, temperature=0.7,
            repetition_penalty=eval_config.repetition_penalty,
        )
        if not answer.strip():
            score = 1.0
        else:
            verdict = _judge(
                engine, sampling, system_prompt,
                'QUESTION: %s\nANSWER: %s' % (question, answer),
            )
            score = _extract_score(verdict, 'accuracy')
        results.append({'question': question, 'answer': answer, 'accuracy': score})
        if score is not None:
            scores.append(score)
    return {'results': results, 'mean_accuracy': _mean(scores)}


def grounded_instructions(config, student, engine, sampling):
    """Headline instruction eval: facts in the user turn, exact-match scored.

    Tasks are drawn from the same grounded mix the pairs train on (retrieval,
    comparison, multihop, notstated, transfer, ratio, order): the user message
    states the facts and asks the question, and the answer is scored by exact
    match against the program-derived gold, per kind. This measures the
    processor capability the corpus actually trains, where the earlier
    real-world instruction set measured knowledge the corpus deliberately
    lacks. A judge additionally rates each answer's coherence so degraded
    English is visible even when the gold token appears.
    """
    from . import worldgen

    random_generator = random.Random(config.project.seed + 5)
    eval_config = config.eval
    count = max(1, eval_config.number_of_generation_samples // 2)
    system_prompt = (
        'A small language model answered a question from facts supplied in '
        'the same message. Rate one axis from one to ten: coherence (is the '
        'answer well-formed, sensible English?). Reply with exactly one '
        'line:\ncoherence: <n>'
    )
    results = []
    scores = []
    by_kind = {}
    coherence_values = []
    for _ in range(count):
        grounding = worldgen.sample_pair_grounding(random_generator)
        prompt = '%s %s' % (
            ' '.join(grounding['facts']), grounding['question']
        )
        answer = student.respond(
            prompt, eval_config.max_new_tokens, eval_config.temperature,
            eval_config.top_p, eval_config.repetition_penalty,
        )
        task = {
            'answer': grounding['answer'],
            'distractor': grounding.get('distractor'),
        }
        score = worldgen.score_binding_answer(task, answer)
        verdict = _judge(
            engine, sampling, system_prompt,
            'QUESTION: %s\nANSWER: %s' % (prompt, answer),
        )
        coherence = _extract_score(verdict, 'coherence')
        kind = grounding['task_kind']
        by_kind.setdefault(kind, []).append(score)
        scores.append(score)
        coherence_values.append(coherence)
        results.append({
            'kind': kind, 'prompt': prompt, 'answer': grounding['answer'],
            'output': answer.strip(), 'correct': score,
            'coherence': coherence,
        })
    return {
        'results': results,
        'exact_match': _mean(scores),
        'by_kind': {kind: _mean(values) for kind, values in by_kind.items()},
        'coherence': _mean(coherence_values),
    }


def binding_probe(config, student):
    """Score in-context binding on program-generated tasks, judge-free.

    Each task supplies every needed fact in its context, about novel invented
    entities, so nothing is answerable from world knowledge; the gold answer
    is known by construction and scored by exact match. Tasks are stratified
    over the question categories, and per-kind sub-scores are reported:
    retrieval and comparison read the context back, multihop composes two
    stated facts, and notstated scores declining over fabricating when the
    context does not contain the answer. This is the coherence gauge that
    gates the later experiments (see the intent graph), at no judge cost.
    """
    from . import worldgen

    eval_config = config.eval
    tasks = worldgen.binding_tasks(
        config.project.seed + 3, eval_config.number_of_binding_tasks
    )
    results = []
    scores = []
    by_kind = {}
    for task in tasks:
        prompt = '%s\nQuestion: %s\nAnswer:' % (
            task['context'], task['question']
        )
        output = student.complete(
            prompt, max_new_tokens=24, temperature=0.3,
            repetition_penalty=eval_config.repetition_penalty,
        )
        score = worldgen.score_binding_answer(task, output)
        results.append({
            'kind': task['kind'], 'question': task['question'],
            'answer': task['answer'], 'output': output.strip(),
            'correct': score,
        })
        scores.append(score)
        by_kind.setdefault(task['kind'], []).append(score)
    return {
        'results': results,
        'exact_match': _mean(scores),
        'by_kind': {kind: _mean(values) for kind, values in by_kind.items()},
    }


def write_report(config, report):
    stage = report['stage']
    output_directory = ensure_directory(config.eval_dir)
    (output_directory / ('report_%s.json' % stage)).write_text(
        json.dumps(report, indent=2)
    )

    completion_means = report['completions']['means']
    instruction_means = report['instructions']['means']
    grounded = report['grounded']
    binding = report['binding']

    def kind_lines(by_kind):
        return [
            '- %s: %s' % (kind, by_kind[kind]) for kind in sorted(by_kind)
        ]

    lines = [
        '# Evaluation report: %s' % config.project.name,
        '',
        '> Stage: %s. Judge model: %s. The headline sections supply facts in '
        'context and score exact match against program-derived answers, '
        'matching the grounded corpus. The out-of-distribution section asks '
        'real-world questions the corpus deliberately does not teach: low '
        'scores there are expected, and the accuracy probe reads as a '
        'contamination gauge, not a target.'
        % (report['stage'], report['judge_model']),
        '',
        '## Grounded instructions (exact match, zero to one)',
        '- exact match: %s' % grounded['exact_match'],
        '- answer coherence (judged, one to ten): %s' % grounded['coherence'],
        '',
        '### By task kind',
    ]
    lines += kind_lines(grounded['by_kind'])
    lines += [
        '',
        '### Sample answers',
    ]
    for result in grounded['results'][:6]:
        lines.append(
            '- [%s] Q: %s\n  gold: %s\n  model: %r (%s)'
            % (result['kind'], result['prompt'], result['answer'],
               result['output'], 'correct' if result['correct'] else 'wrong')
        )
    lines += [
        '',
        '## In-context binding (exact match, zero to one)',
        '- exact match: %s' % binding['exact_match'],
        '',
        '### By task kind',
    ]
    lines += kind_lines(binding['by_kind'])
    lines += ['']
    for result in binding['results'][:6]:
        lines.append(
            '- [%s] Q: %s\n  gold: %s\n  model: %r (%s)'
            % (result['kind'], result['question'], result['answer'],
               result['output'],
               'correct' if result['correct'] else 'wrong')
        )
    lines += [
        '',
        '## Completions (judged, one to ten)',
        '- grammar: %s' % completion_means.get('grammar'),
        '- coherence: %s' % completion_means.get('coherence'),
        '',
        '### Sample completions',
    ]
    for sample in report['completions']['samples'][:6]:
        lines.append('- %r' % sample['output'])
    lines += [
        '',
        '## Out-of-distribution generalization (judged, one to ten)',
        '',
        'Real-world instructions and facts the grounded corpus does not '
        'contain. Expected low; movement here signals contamination or '
        'transfer, not the training target.',
        '- instruction coherence: %s' % instruction_means.get('coherence'),
        '- instruction followed: %s' % instruction_means.get('followed'),
        '- factual accuracy: %s' % report['probe']['mean_accuracy'],
        '',
        '### Sample out-of-distribution answers',
    ]
    for result in report['instructions']['results'][:3]:
        lines.append(
            '- Q: %s\n  A: %r' % (result['instruction'], result['answer'])
        )
    for result in report['probe']['results'][:3]:
        lines.append(
            '- Q: %s\n  A: %r (accuracy %s)'
            % (result['question'], result['answer'], result['accuracy'])
        )
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
        'grounded': grounded_instructions(config, student, engine, sampling),
        'binding': binding_probe(config, student),
        'completions': score_completions(config, student, engine, sampling),
        'instructions': score_instructions(config, student, engine, sampling),
        'probe': accuracy_probe(config, student, engine, sampling),
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
    try:
        from .report import write_summary

        write_summary(config)
    except Exception as error:
        logger.warning('run summary failed: %s', error)
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

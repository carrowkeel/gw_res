"""Generic post-stage evaluation: answer similarity on held-out pairs.

Runs after each SFT stage: the model generates its answer for each held-out
prompt and the answer is scored against the reference by similarity (token
overlap and containment), never by format — the same forgiving principle as
the listener and the earlier eval. The report carries the answered rate, so
the same module both tracks a stage's improvement over its source
checkpoint and serves as the threshold instrument for entering the
simulator.

    python -m slm.sfteval --config <resolved.yaml> --checkpoint <ckpt.pt> \
      --records <holdout.jsonl> --output <report.json>
"""

import argparse
import json
import re

from .config import load_config
from .sftstage import load_checkpoint_model
from .tokenizer import SyntheticTokenizer
from .utils import get_logger

logger = get_logger('sfteval')

_WORD_PATTERN = re.compile(r'[a-z0-9]+')


def normalize_tokens(text):
    return _WORD_PATTERN.findall(text.lower())


def token_f1(prediction, reference):
    prediction_tokens = normalize_tokens(prediction)
    reference_tokens = normalize_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = 0
    remaining = list(reference_tokens)
    for token in prediction_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, reference, threshold):
    """Return (f1, answered): similar enough by overlap or by containment."""
    f1 = token_f1(prediction, reference)
    contained = ' '.join(normalize_tokens(reference)) in ' '.join(
        normalize_tokens(prediction)
    )
    return f1, bool(contained or f1 >= threshold)


def generate_answer(model, tokenizer, prompt, sfteval_config, block_size,
                    device):
    import torch

    token_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    limit = block_size - sfteval_config.max_new_tokens
    token_ids = token_ids[-limit:]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids, sfteval_config.max_new_tokens,
        temperature=sfteval_config.temperature,
        top_p=sfteval_config.top_p,
        repetition_penalty=sfteval_config.repetition_penalty,
        eos_id=tokenizer.eos_id,
    )
    generated = output[0, len(token_ids):].tolist()
    if tokenizer.eos_id in generated:
        generated = generated[:generated.index(tokenizer.eos_id)]
    return tokenizer.decode(generated).split('\n')[0].strip()


def evaluate_model(model, tokenizer, records, sfteval_config, block_size,
                   device):
    """Score a model over held-out records; return (report, per-example rows)."""
    records = records[:sfteval_config.maximum_examples]
    rows = []
    answered_count = 0
    f1_total = 0.0
    for record in records:
        prediction = generate_answer(
            model, tokenizer, record['prompt'], sfteval_config, block_size,
            device,
        )
        f1, answered = score_answer(
            prediction, record['response'],
            sfteval_config.similarity_threshold,
        )
        answered_count += int(answered)
        f1_total += f1
        rows.append({
            'question': record.get('question'),
            'kind': record.get('kind'),
            'reference': record['response'],
            'prediction': prediction,
            'f1': round(f1, 3),
            'answered': answered,
        })
    report = {
        'examples': len(records),
        'answered_rate': (
            round(answered_count / len(records), 4) if records else 0.0
        ),
        'mean_f1': round(f1_total / len(records), 4) if records else 0.0,
        'similarity_threshold': sfteval_config.similarity_threshold,
    }
    kinds = {}
    for row in rows:
        if row['kind'] is None:
            continue
        entry = kinds.setdefault(row['kind'], [0, 0, 0.0])
        entry[0] += 1
        entry[1] += int(row['answered'])
        entry[2] += row['f1']
    if kinds:
        report['by_kind'] = {
            kind: {
                'examples': count,
                'answered_rate': round(answered / count, 4),
                'mean_f1': round(f1_sum / count, 4),
            }
            for kind, (count, answered, f1_sum) in sorted(kinds.items())
        }
    return report, rows


def load_records(path):
    records = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def evaluate_checkpoint(config, checkpoint_path, records, output_path,
                        label):
    """Evaluate one checkpoint over records and write the report and samples."""
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    model, gpt_config = load_checkpoint_model(
        config, checkpoint_path, device
    )
    model.eval()
    report, rows = evaluate_model(
        model, tokenizer, records, config.sfteval, gpt_config.block_size,
        device,
    )
    report = {'label': label, 'checkpoint': str(checkpoint_path), **report}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    samples_path = output_path.with_name(output_path.stem + '_samples.jsonl')
    with open(samples_path, 'w') as handle:
        for row in rows[:config.sfteval.sample_dump]:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    logger.info(
        '%s: answered %.3f, mean f1 %.3f over %d examples -> %s',
        label, report['answered_rate'], report['mean_f1'],
        report['examples'], output_path,
    )
    del model
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Answer-similarity evaluation over held-out pairs'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--records', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', default='sfteval')
    arguments = parser.parse_args()
    from pathlib import Path

    config = load_config(arguments.config)
    evaluate_checkpoint(
        config, Path(arguments.checkpoint),
        load_records(Path(arguments.records)), Path(arguments.output),
        arguments.label,
    )


if __name__ == '__main__':
    main()

"""One-document summary of a run: training statistics and evaluations.

Collects everything needed to judge a run into a single markdown file
(eval/summary.md, with a machine-readable summary.json beside it): model and
corpus sizes, pretrain and finetune loss curves with perplexities, a
comparable-loss table that scores every checkpoint on the same held-out data
(both the pretraining validation stream and the held-out instruction pairs,
so forgetting and instruction gain are visible side by side), and the judged
and exact-match evaluation scores from the report files.

Runs automatically at the end of the evaluate stage, and standalone:

    python -m slm.report --config runs/world/mini/config.yaml
"""

import argparse
import json
import math

import numpy

from .config import load_config
from .utils import ensure_directory, get_logger

logger = get_logger('report')

COMPARABLE_PAIR_LIMIT = 256
COMPARABLE_CORPUS_BATCHES = 32


def _perplexity(loss):
    if loss is None:
        return None
    return round(math.exp(min(loss, 20)), 2)


def _round(value, digits=4):
    return round(value, digits) if value is not None else None


def _find_checkpoint(base):
    for name in ('ckpt_best.pt', 'ckpt_last.pt'):
        if (base / name).exists():
            return base / name
    return None


def _checkpoint_stats(path):
    import torch

    saved = torch.load(path, map_location='cpu')
    stats = {
        'step': saved.get('step'),
        'validation_loss': saved.get('validation_loss'),
        'best_validation': saved.get('best_validation'),
        'vocabulary_size': saved.get('vocabulary_size'),
    }
    return stats, saved


def _load_model(config, saved):
    from .model import GPT, build_config

    gpt_config = build_config(config.model, saved['vocabulary_size'])
    model = GPT(gpt_config)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, gpt_config


def _read_history(path, limit=8):
    if not path.exists():
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if len(rows) <= limit:
        return rows
    indices = sorted({
        int(round(position)) for position in
        numpy.linspace(0, len(rows) - 1, limit)
    })
    return [rows[index] for index in indices]


def _pair_validation_indices(config, dataset_length):
    from .finetune import _split_indices

    random_generator = numpy.random.default_rng(config.project.seed)
    _, validation = _split_indices(
        dataset_length, config.finetune.validation_fraction, random_generator
    )
    if validation:
        return validation[:COMPARABLE_PAIR_LIMIT], 'held-out pairs'
    return list(range(min(dataset_length, COMPARABLE_PAIR_LIMIT))), (
        'first pairs (no held-out split configured; sft has trained on these)'
    )


def _pair_loss(model, dataset, indices, batch_size=16):
    import torch

    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            if not batch:
                continue
            inputs, targets, _ = dataset.collate(batch, 'cpu')
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    return float(numpy.mean(losses)) if losses else None


def _corpus_loss(model, config, block_size):
    import torch

    from .data import PackedDataset

    packed = config.data_dir / 'packed'
    meta_path = packed / 'meta.json'
    validation_path = packed / 'val.bin'
    if not (meta_path.exists() and validation_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    dataset = PackedDataset(validation_path, meta['dtype'], block_size)
    random_generator = numpy.random.default_rng(config.project.seed + 11)
    losses = []
    with torch.no_grad():
        for _ in range(COMPARABLE_CORPUS_BATCHES):
            inputs, targets = dataset.get_batch(8, 'cpu', random_generator)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    return float(numpy.mean(losses))


def _evaluation_means(config):
    rows = []
    for path in sorted(config.eval_dir.glob('report_*.json')):
        data = json.loads(path.read_text())
        rows.append({
            'stage': data.get('stage'),
            'grounded': data.get('grounded', {}).get('exact_match'),
            'binding': data.get('binding', {}).get('exact_match'),
            'grammar': data['completions']['means'].get('grammar'),
            'coherence': data['completions']['means'].get('coherence'),
            'instruction_coherence':
                data['instructions']['means'].get('coherence'),
            'followed': data['instructions']['means'].get('followed'),
            'accuracy': data['probe'].get('mean_accuracy'),
        })
    return rows


def gather(config):
    """Collect training statistics and evaluation means for a run."""
    from .evaluate import _sft_targets
    from .data import PairDataset
    from .tokenizer import SyntheticTokenizer

    summary = {'name': config.project.name, 'checkpoints': {}}

    meta_path = config.data_dir / 'packed' / 'meta.json'
    if meta_path.exists():
        summary['corpus'] = json.loads(meta_path.read_text())

    targets = [('pretrain', config.pretrain_dir)] + _sft_targets(config)
    tokenizer = None
    pair_dataset = None
    pair_indices = None
    pair_note = None
    if config.tokenizer_path.exists() and config.corpus_sft_path.exists():
        tokenizer = SyntheticTokenizer(config.tokenizer_path)
        pair_dataset = PairDataset(config, tokenizer)
        pair_indices, pair_note = _pair_validation_indices(
            config, pair_dataset.length()
        )
        summary['pair_loss_note'] = pair_note

    for label, directory in targets:
        checkpoint_path = _find_checkpoint(directory)
        if checkpoint_path is None:
            continue
        stats, saved = _checkpoint_stats(checkpoint_path)
        stats['checkpoint'] = checkpoint_path.name
        model, gpt_config = _load_model(config, saved)
        if 'parameters' not in summary:
            summary['parameters'] = {
                'total': model.count_parameters(),
                'non_embedding': model.count_parameters(non_embedding=True),
                'preset': config.model.preset,
            }
            train_tokens = summary.get('corpus', {}).get('train_tokens')
            if train_tokens:
                summary['parameters']['tokens_per_non_embedding'] = round(
                    train_tokens / summary['parameters']['non_embedding'], 1
                )
        stats['corpus_validation_loss'] = _round(
            _corpus_loss(model, config, gpt_config.block_size)
        )
        if pair_dataset is not None:
            stats['pair_validation_loss'] = _round(
                _pair_loss(model, pair_dataset, pair_indices)
            )
        stats['history'] = _read_history(directory / 'history.jsonl')
        summary['checkpoints'][label] = stats
        del model

    summary['evaluations'] = _evaluation_means(config)
    return summary


def _history_lines(history):
    lines = []
    for row in history:
        lines.append('| %s | %s | %s |' % (
            row.get('step'), row.get('train_loss'),
            row.get('validation_loss'),
        ))
    return lines


def write_summary(config):
    """Write eval/summary.md and summary.json for the run."""
    summary = gather(config)
    output_directory = ensure_directory(config.eval_dir)
    (output_directory / 'summary.json').write_text(
        json.dumps(summary, indent=2)
    )

    lines = ['# Run summary: %s' % summary['name'], '']
    parameters = summary.get('parameters')
    corpus = summary.get('corpus', {})
    if parameters:
        lines += [
            '## Model and corpus',
            '- preset: %s' % parameters['preset'],
            '- parameters: %.2fM total, %.2fM non-embedding' % (
                parameters['total'] / 1e6,
                parameters['non_embedding'] / 1e6,
            ),
            '- train tokens: %s (validation %s), vocabulary %s' % (
                corpus.get('train_tokens'), corpus.get('validation_tokens'),
                corpus.get('vocabulary_size'),
            ),
            '- tokens per non-embedding parameter: %s' % parameters.get(
                'tokens_per_non_embedding'
            ),
            '- instruction token fraction: %s' % corpus.get(
                'instruction_token_fraction'
            ),
            '',
        ]

    lines += [
        '## Comparable losses (same data for every checkpoint)',
        '',
        'Corpus loss is next-token loss on the pretraining validation stream '
        '(rise after finetuning indicates forgetting). Pair loss is '
        'response-only loss on %s (drop after finetuning indicates '
        'instruction gain).' % summary.get('pair_loss_note', 'the pairs'),
        '',
        '| checkpoint | step | corpus loss | corpus ppl | pair loss | pair ppl |',
        '|---|---|---|---|---|---|',
    ]
    for label, stats in summary['checkpoints'].items():
        corpus_loss = stats.get('corpus_validation_loss')
        pair_loss = stats.get('pair_validation_loss')
        lines.append('| %s (%s) | %s | %s | %s | %s | %s |' % (
            label, stats.get('checkpoint'), stats.get('step'),
            corpus_loss, _perplexity(corpus_loss),
            pair_loss, _perplexity(pair_loss),
        ))
    lines.append('')

    for label, stats in summary['checkpoints'].items():
        history = stats.get('history') or []
        lines += [
            '## Training: %s' % label,
            '- checkpoint: %s, step %s' % (
                stats.get('checkpoint'), stats.get('step')
            ),
            '- checkpoint validation loss: %s (best seen %s)' % (
                _round(stats.get('validation_loss')),
                _round(stats.get('best_validation')),
            ),
        ]
        if history:
            lines += [
                '',
                '| step | train loss | validation loss |',
                '|---|---|---|',
            ] + _history_lines(history)
        lines.append('')

    evaluations = summary.get('evaluations') or []
    lines += ['## Evaluation scores', '']
    if evaluations:
        lines += [
            '| stage | grounded | binding | grammar | coherence | '
            'instr. coherence | followed | accuracy |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for row in evaluations:
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                row['stage'], _round(row['grounded'], 2),
                _round(row['binding'], 2), _round(row['grammar'], 2),
                _round(row['coherence'], 2),
                _round(row['instruction_coherence'], 2),
                _round(row['followed'], 2), _round(row['accuracy'], 2),
            ))
        lines += [
            '',
            'Grounded and binding are exact-match from zero to one over '
            'program-derived answers (grounded instructions and in-context '
            'binding); the rest are judge scores from one to ten, with '
            'instruction and accuracy scores measuring out-of-distribution '
            'generalization, not the training target. Per-kind sub-scores '
            'are in the per-stage report files beside this summary.',
        ]
    else:
        lines.append('No evaluation reports found; run the evaluate stage.')
    lines.append('')

    summary_path = output_directory / 'summary.md'
    summary_path.write_text('\n'.join(lines))
    logger.info('wrote run summary to %s', summary_path)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description='Summarize a run')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    write_summary(load_config(arguments.config))


if __name__ == '__main__':
    main()

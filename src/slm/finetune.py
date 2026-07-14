"""Stage four: supervised finetuning on instruction pairs.

Loads the pretrain checkpoint and continues training on the generated pairs.
Labels are next-token targets with the prompt span masked (response_only) or
unmasked (full_sequence). The stage supports a sweep of variants, each forked
from the same pretrain checkpoint, so several finetuning approaches can be
compared cheaply against one pretraining run: pretraining-data replay to
counter forgetting, validation-based early stopping, the loss mode, and any
optimization override. Each variant writes to its own checkpoint directory.

    python -m slm.finetune --config configs/poc.yaml
    python -m slm.finetune --config configs/scale/world.yaml --variant replay
"""

import argparse
import math
from contextlib import nullcontext
from dataclasses import replace

import numpy

from .config import load_config, to_dict
from .model import GPT, build_config
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('finetune')


def _load_pretrained(config, device):
    import torch

    checkpoint_path = config.pretrain_dir / 'ckpt_best.pt'
    if not checkpoint_path.exists():
        checkpoint_path = config.pretrain_dir / 'ckpt_last.pt'
    saved = torch.load(checkpoint_path, map_location=device)
    gpt_config = build_config(config.model, saved['vocabulary_size'])
    model = GPT(gpt_config).to(device)
    model.load_state_dict(saved['model'])
    logger.info('loaded pretrained weights from %s', checkpoint_path)
    return model, gpt_config


def _save(model, config, gpt_config, step, checkpoint_directory, name,
          train_loss=None, validation_loss=None, best_validation=None):
    import torch

    base_model = getattr(model, '_orig_mod', model)
    if best_validation is not None and best_validation == float('inf'):
        best_validation = None
    torch.save(
        {
            'model': base_model.state_dict(),
            'step': step,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'best_validation': best_validation,
            'model_config': to_dict(config.model),
            'vocabulary_size': gpt_config.vocabulary_size,
        },
        checkpoint_directory / name,
    )


def _effective_finetune(base_finetune, spec):
    """Return the base finetune config overlaid with a variant's overrides."""
    if spec is None:
        return base_finetune
    overrides = {key: value for key, value in spec.items() if key != 'name'}
    return replace(base_finetune, **overrides)


def _load_replay_dataset(config, block_size):
    """Return a packed pretraining sampler for replay, or None if unavailable."""
    import json

    from .data import PackedDataset

    packed_directory = config.data_dir / 'packed'
    meta_path = packed_directory / 'meta.json'
    train_path = packed_directory / 'train.bin'
    if not (meta_path.exists() and train_path.exists()):
        logger.warning('no packed pretraining data for replay, disabling replay')
        return None
    with open(meta_path) as handle:
        meta = json.load(handle)
    return PackedDataset(train_path, meta['dtype'], block_size)


def _split_indices(count, validation_fraction, random_generator):
    order = random_generator.permutation(count).tolist()
    validation_size = int(count * validation_fraction)
    return order[validation_size:], order[:validation_size]


def _validation_loss(model, dataset, indices, batch_size, device, autocast):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if not batch_indices:
                continue
            inputs, targets, _ = dataset.collate(batch_indices, device)
            with autocast:
                _, loss = model(inputs, targets)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def _train(config, finetune_config, tokenizer, name, checkpoint_directory):
    import torch

    from .data import PairDataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    model, gpt_config = _load_pretrained(config, device)
    dataset = PairDataset(
        config, tokenizer, finetune_config.loss_mode,
        finetune_config.maximum_sequence_length,
    )

    precision = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[finetune_config.dtype]
    autocast = (
        nullcontext() if device_type == 'cpu'
        else torch.autocast(device_type=device_type, dtype=precision)
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=(finetune_config.dtype == 'float16')
    )
    if finetune_config.compile_model and device_type == 'cuda':
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        finetune_config.weight_decay, finetune_config.learning_rate,
        (0.9, 0.95), device_type,
    )

    random_generator = numpy.random.default_rng(config.project.seed)
    train_indices, validation_indices = _split_indices(
        dataset.length(), finetune_config.validation_fraction, random_generator
    )
    replay = None
    if finetune_config.replay_fraction > 0.0:
        replay = _load_replay_dataset(config, gpt_config.block_size)
    replay_generator = numpy.random.default_rng(config.project.seed + 7)

    examples_per_step = (
        finetune_config.batch_size * finetune_config.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(len(train_indices) / examples_per_step))
    total_steps = (
        finetune_config.maximum_steps
        or steps_per_epoch * finetune_config.epochs
    )
    warmup_steps = max(1, int(total_steps * finetune_config.warmup_ratio))
    early_stopping = (
        len(validation_indices) > 0 and finetune_config.early_stop_patience > 0
        and finetune_config.evaluation_interval > 0
    )
    logger.info(
        'finetune variant %s: %d train, %d val examples, %d steps, replay %.2f, '
        'loss %s, early stop %s',
        name, len(train_indices), len(validation_indices), total_steps,
        finetune_config.replay_fraction, finetune_config.loss_mode,
        early_stopping,
    )

    def learning_rate_at(step):
        if step < warmup_steps:
            return finetune_config.learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return finetune_config.minimum_learning_rate + coefficient * (
            finetune_config.learning_rate - finetune_config.minimum_learning_rate
        )

    order = random_generator.permutation(train_indices).tolist()
    cursor = 0

    def next_indices(count):
        nonlocal cursor, order
        chosen = []
        while len(chosen) < count:
            if cursor >= len(order):
                order = random_generator.permutation(train_indices).tolist()
                cursor = 0
            chosen.append(order[cursor])
            cursor += 1
        return chosen

    import json as json_module

    history_path = checkpoint_directory / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()

    def record_history(step, train_loss, validation_loss):
        with open(history_path, 'a') as handle:
            handle.write(json_module.dumps({
                'step': step,
                'train_loss': round(train_loss, 4),
                'validation_loss': (
                    round(validation_loss, 4)
                    if validation_loss is not None else None
                ),
            }) + '\n')

    best_validation = float('inf')
    patience = 0
    step = 0
    accumulated_loss = 0.0
    model.train()
    for step in range(total_steps):
        current_learning_rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(finetune_config.gradient_accumulation_steps):
            if replay is not None and (
                replay_generator.random() < finetune_config.replay_fraction
            ):
                inputs, targets = replay.get_batch(
                    finetune_config.batch_size, device, replay_generator
                )
            else:
                inputs, targets, _ = dataset.collate(
                    next_indices(finetune_config.batch_size), device
                )
            with autocast:
                _, loss = model(inputs, targets)
                loss = loss / finetune_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        if finetune_config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), finetune_config.gradient_clip
            )
        scaler.step(optimizer)
        scaler.update()

        if step % finetune_config.log_interval == 0:
            logger.info(
                'finetune %s step %d/%d  loss %.4f  lr %.2e',
                name, step, total_steps, accumulated_loss, current_learning_rate,
            )
        if early_stopping and step > 0 and (
            step % finetune_config.evaluation_interval == 0
        ):
            validation_loss = _validation_loss(
                model, dataset, validation_indices, finetune_config.batch_size,
                device, autocast,
            )
            record_history(step, accumulated_loss, validation_loss)
            logger.info(
                'finetune %s step %d  val loss %.4f (best %.4f)',
                name, step, validation_loss, best_validation,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                patience = 0
                _save(
                    model, config, gpt_config, step, checkpoint_directory,
                    'ckpt_best.pt', accumulated_loss, validation_loss,
                    best_validation,
                )
            else:
                patience += 1
                if patience >= finetune_config.early_stop_patience:
                    logger.info('finetune %s early stopping at step %d', name, step)
                    break

    final_validation = None
    if validation_indices:
        final_validation = _validation_loss(
            model, dataset, validation_indices, finetune_config.batch_size,
            device, autocast,
        )
        if final_validation < best_validation:
            best_validation = final_validation
    record_history(step, accumulated_loss, final_validation)
    _save(model, config, gpt_config, step, checkpoint_directory,
          'ckpt_last.pt', accumulated_loss, final_validation, best_validation)
    logger.info('finetune variant %s complete -> %s', name, checkpoint_directory)
    del model


def _variant_output_dir(config, spec):
    if spec is None:
        return ensure_directory(config.sft_dir)
    return ensure_directory(config.sft_dir / spec['name'])


def run(config, variant_name=None):
    """Run one finetune variant, or every configured variant in sequence."""
    set_seed(config.project.seed)
    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    variants = config.finetune.variants
    if not variants:
        _train(
            config, config.finetune, tokenizer, 'sft',
            _variant_output_dir(config, None),
        )
        return
    selected = variants
    if variant_name is not None:
        selected = [spec for spec in variants if spec['name'] == variant_name]
        if not selected:
            raise SystemExit('unknown finetune variant %r' % variant_name)
    for spec in selected:
        finetune_config = _effective_finetune(config.finetune, spec)
        _train(
            config, finetune_config, tokenizer, spec['name'],
            _variant_output_dir(config, spec),
        )


def main():
    parser = argparse.ArgumentParser(description='Supervised finetuning')
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', default=None)
    arguments = parser.parse_args()
    run(load_config(arguments.config), arguments.variant)


if __name__ == '__main__':
    main()

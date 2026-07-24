"""Shared machinery for the staged SFT steps between pretraining and the simulator.

The communication and arithmetic SFT stages train the same way: plain
prompt-and-response text in the corpus's own format (no chat template),
loss on the response tokens only, stage-1 replay mixed in against
forgetting, and best-checkpoint selection on held-out loss. This module
holds the dataset and training loop they share; each stage supplies its
own data, source checkpoint, and post-stage metric.
"""

import json
import math

import numpy

from .config import to_dict
from .model import GPT, build_config
from .utils import get_logger, normalize_state_dict

logger = get_logger('sftstage')


STAGE_ORDER = ['mathsft', 'commsft', 'pretrain']


def resolve_checkpoint(directory):
    """Return the best checkpoint in a stage directory, falling back to last."""
    best = directory / 'ckpt_best.pt'
    if best.exists():
        return best
    last = directory / 'ckpt_last.pt'
    if last.exists():
        return last
    return None


def resolve_base_checkpoint(base_directory):
    """Return the furthest-stage checkpoint a run tree has produced.

    The stage chain is pretrain, then communication SFT, then arithmetic
    SFT; a downstream consumer (the gate, the simulator) builds on the
    latest stage the base run completed.
    """
    from pathlib import Path

    base_directory = Path(base_directory)
    for stage in STAGE_ORDER:
        found = resolve_checkpoint(base_directory / 'checkpoints' / stage)
        if found is not None:
            return found
    return None


def load_checkpoint_model(config, checkpoint_path, device):
    import torch

    saved = torch.load(checkpoint_path, map_location=device)
    gpt_config = build_config(config.model, saved['vocabulary_size'])
    model = GPT(gpt_config).to(device)
    model.load_state_dict(normalize_state_dict(saved['model']))
    return model, gpt_config


class PlainPairDataset:
    """Prompt-and-response examples in plain corpus text with response-only loss.

    Unlike PairDataset there is no Question and Answer template: the prompt
    is already complete text (a dialogue ending at the answering speaker's
    cue), so the example is bos + prompt tokens + response tokens + eos,
    with every position that predicts a prompt token masked out.
    """

    def __init__(self, records, tokenizer, maximum_length):
        self.examples = []
        self.pad_id = tokenizer.pad_id
        for record in records:
            prefix_ids = (
                [tokenizer.bos_id] + tokenizer.encode(record['prompt'])
            )
            answer_ids = (
                tokenizer.encode(' ' + record['response'].strip())
                + [tokenizer.eos_id]
            )
            tokens = prefix_ids + answer_ids
            input_ids = tokens[:-1]
            labels = [-100] * (len(prefix_ids) - 1) + answer_ids
            self.examples.append(
                (input_ids[:maximum_length], labels[:maximum_length])
            )

    def length(self):
        return len(self.examples)

    def collate(self, indices, device):
        import torch

        items = [self.examples[index] for index in indices]
        longest = max(len(input_ids) for input_ids, _ in items)
        input_batch = []
        label_batch = []
        for input_ids, labels in items:
            padding = longest - len(input_ids)
            input_batch.append(input_ids + [self.pad_id] * padding)
            label_batch.append(labels + [-100] * padding)
        as_tensor = lambda rows: torch.tensor(
            rows, dtype=torch.long, device=device
        )
        return as_tensor(input_batch), as_tensor(label_batch)


def _load_replay(config, block_size):
    from .data import PackedDataset

    packed_directory = config.data_dir / 'packed'
    meta_path = packed_directory / 'meta.json'
    train_path = packed_directory / 'train.bin'
    if not (meta_path.exists() and train_path.exists()):
        logger.warning('no packed stage-1 data, replay disabled')
        return None
    meta = json.loads(meta_path.read_text())
    return PackedDataset(train_path, meta['dtype'], block_size)


def _load_replay_validation(config, block_size):
    from .data import PackedDataset

    packed_directory = config.data_dir / 'packed'
    meta_path = packed_directory / 'meta.json'
    validation_path = packed_directory / 'val.bin'
    if not (meta_path.exists() and validation_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    dataset = PackedDataset(validation_path, meta['dtype'], block_size)
    return dataset if dataset.length() > 0 else None


def replay_validation_loss(model, dataset, batch_size, device, autocast,
                           batches=8):
    """Stage-1 loss on fixed held-out batches: the forgetting gauge.

    The same offsets are drawn every call, so the number is comparable
    across evaluations; upward drift over an SFT stage is the warning sign
    that stage-1 language is being overwritten.
    """
    import torch

    generator = numpy.random.default_rng(202607)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = dataset.get_batch(
                batch_size, device, generator
            )
            with autocast:
                _, loss = model(inputs, targets)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def _mean_loss(model, dataset, indices, batch_size, device, autocast):
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if not batch_indices:
                continue
            inputs, labels = dataset.collate(batch_indices, device)
            with autocast:
                _, loss = model(inputs, labels)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def train_stage(config, stage_config, tokenizer, train_records,
                validation_records, source_checkpoint, checkpoint_directory,
                name):
    """Train one SFT stage from a source checkpoint; return the best checkpoint.

    Best selection is by held-out loss when validation records exist,
    otherwise the final checkpoint stands. History lands in the stage's
    checkpoint directory, one row per evaluation.
    """
    import torch
    from contextlib import nullcontext

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    model, gpt_config = load_checkpoint_model(
        config, source_checkpoint, device
    )
    logger.info('%s: starting from %s', name, source_checkpoint)

    dataset = PlainPairDataset(
        train_records, tokenizer, stage_config.maximum_sequence_length
    )
    validation = None
    if validation_records:
        validation = PlainPairDataset(
            validation_records, tokenizer,
            stage_config.maximum_sequence_length,
        )

    precision = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[stage_config.dtype]
    autocast = (
        nullcontext() if device_type == 'cpu'
        else torch.autocast(device_type=device_type, dtype=precision)
    )
    if stage_config.compile_model and device_type == 'cuda':
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        stage_config.weight_decay, stage_config.learning_rate,
        (0.9, 0.95), device_type,
    )

    replay = None
    if stage_config.replay_fraction > 0.0:
        replay = _load_replay(config, gpt_config.block_size)
    replay_validation = _load_replay_validation(config, gpt_config.block_size)
    replay_generator = numpy.random.default_rng(config.project.seed + 7)
    random_generator = numpy.random.default_rng(config.project.seed)

    examples_per_step = (
        stage_config.batch_size * stage_config.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(dataset.length() / examples_per_step))
    total_steps = (
        stage_config.maximum_steps
        or steps_per_epoch * stage_config.epochs
    )
    warmup_steps = max(1, int(total_steps * stage_config.warmup_ratio))
    logger.info(
        '%s: %d train, %d val examples, %d steps, replay %.2f',
        name, dataset.length(),
        validation.length() if validation else 0,
        total_steps, stage_config.replay_fraction,
    )

    def learning_rate_at(step):
        if step < warmup_steps:
            return stage_config.learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return stage_config.minimum_learning_rate + coefficient * (
            stage_config.learning_rate - stage_config.minimum_learning_rate
        )

    order = random_generator.permutation(dataset.length()).tolist()
    cursor = 0

    def next_indices(count):
        nonlocal cursor, order
        chosen = []
        while len(chosen) < count:
            if cursor >= len(order):
                order = random_generator.permutation(
                    dataset.length()
                ).tolist()
                cursor = 0
            chosen.append(order[cursor])
            cursor += 1
        return chosen

    def save(step, filename, train_loss, validation_loss):
        base_model = getattr(model, '_orig_mod', model)
        torch.save(
            {
                'model': base_model.state_dict(),
                'step': step,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'model_config': to_dict(config.model),
                'vocabulary_size': gpt_config.vocabulary_size,
            },
            checkpoint_directory / filename,
        )

    history_path = checkpoint_directory / 'history.jsonl'
    if history_path.exists():
        history_path.unlink()

    initial_replay_loss = None
    if replay_validation is not None:
        initial_replay_loss = replay_validation_loss(
            model, replay_validation, stage_config.batch_size, device,
            autocast,
        )
        logger.info(
            '%s: stage-1 replay validation loss before training %.4f '
            '(the forgetting reference)', name, initial_replay_loss,
        )
        with open(history_path, 'a') as handle:
            handle.write(json.dumps({
                'step': -1,
                'replay_validation_loss': round(initial_replay_loss, 4),
            }) + '\n')

    best_validation = float('inf')
    step = 0
    step_loss = 0.0
    replay_loss_value = None
    model.train()
    for step in range(total_steps):
        current_learning_rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(stage_config.gradient_accumulation_steps):
            if replay is not None and (
                replay_generator.random() < stage_config.replay_fraction
            ):
                inputs, labels = replay.get_batch(
                    stage_config.batch_size, device, replay_generator
                )
                is_replay = True
            else:
                inputs, labels = dataset.collate(
                    next_indices(stage_config.batch_size), device
                )
                is_replay = False
            with autocast:
                _, loss = model(inputs, labels)
                loss = loss / stage_config.gradient_accumulation_steps
            loss.backward()
            step_loss += loss.item()
            if is_replay:
                replay_loss_value = (
                    loss.item() * stage_config.gradient_accumulation_steps
                )
        if stage_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), stage_config.gradient_clip
            )
        optimizer.step()

        if step % stage_config.log_interval == 0:
            logger.info(
                '%s step %d/%d  loss %.4f  replay %s  lr %.2e',
                name, step, total_steps, step_loss,
                '%.4f' % replay_loss_value
                if replay_loss_value is not None else '-',
                current_learning_rate,
            )
        if (validation is not None and stage_config.evaluation_interval > 0
                and step > 0
                and step % stage_config.evaluation_interval == 0):
            validation_loss = _mean_loss(
                model, validation, list(range(validation.length())),
                stage_config.batch_size, device, autocast,
            )
            replay_validation_value = None
            if replay_validation is not None:
                replay_validation_value = replay_validation_loss(
                    model, replay_validation, stage_config.batch_size,
                    device, autocast,
                )
            with open(history_path, 'a') as handle:
                handle.write(json.dumps({
                    'step': step,
                    'train_loss': round(step_loss, 4),
                    'validation_loss': round(validation_loss, 4),
                    'replay_loss': (
                        round(replay_loss_value, 4)
                        if replay_loss_value is not None else None
                    ),
                    'replay_validation_loss': (
                        round(replay_validation_value, 4)
                        if replay_validation_value is not None else None
                    ),
                }) + '\n')
            logger.info(
                '%s step %d  val loss %.4f (best %.4f)  replay val %s%s',
                name, step, validation_loss, best_validation,
                '%.4f' % replay_validation_value
                if replay_validation_value is not None else '-',
                ' (started %.4f)' % initial_replay_loss
                if initial_replay_loss is not None else '',
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                save(step, 'ckpt_best.pt', step_loss, validation_loss)

    final_validation = None
    if validation is not None:
        final_validation = _mean_loss(
            model, validation, list(range(validation.length())),
            stage_config.batch_size, device, autocast,
        )
        if final_validation < best_validation:
            best_validation = final_validation
            save(step, 'ckpt_best.pt', step_loss, final_validation)
        final_replay = None
        if replay_validation is not None:
            final_replay = replay_validation_loss(
                model, replay_validation, stage_config.batch_size, device,
                autocast,
            )
            logger.info(
                '%s: stage-1 replay validation loss after training %.4f '
                '(started %.4f)', name, final_replay, initial_replay_loss,
            )
        with open(history_path, 'a') as handle:
            handle.write(json.dumps({
                'step': step,
                'train_loss': round(step_loss, 4),
                'validation_loss': round(final_validation, 4),
                'replay_validation_loss': (
                    round(final_replay, 4)
                    if final_replay is not None else None
                ),
            }) + '\n')
    save(step, 'ckpt_last.pt', step_loss, final_validation)
    logger.info('%s complete -> %s', name, checkpoint_directory)
    del model
    return resolve_checkpoint(checkpoint_directory)

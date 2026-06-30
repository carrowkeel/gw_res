"""Stage four: supervised finetuning on referent-free instruction pairs.

Loads the best pretrain checkpoint, trains with the response-only loss mask, and
writes ckpt_last.pt to the finetuning checkpoint directory.

    python -m slm.finetune --config configs/poc.yaml
"""

import argparse
import math
from contextlib import nullcontext

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


def _save(model, config, gpt_config, step, checkpoint_directory):
    import torch

    base_model = getattr(model, '_orig_mod', model)
    torch.save(
        {
            'model': base_model.state_dict(),
            'step': step,
            'model_config': to_dict(config.model),
            'vocabulary_size': gpt_config.vocabulary_size,
        },
        checkpoint_directory / 'ckpt_last.pt',
    )


def run(config):
    import torch

    from .data import PairDataset

    set_seed(config.project.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    finetune_config = config.finetune

    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    model, gpt_config = _load_pretrained(config, device)
    dataset = PairDataset(config, tokenizer)

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
        finetune_config.weight_decay,
        finetune_config.learning_rate,
        (0.9, 0.95),
        device_type,
    )

    examples_per_step = (
        finetune_config.batch_size * finetune_config.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(dataset.length() / examples_per_step)
    total_steps = (
        finetune_config.maximum_steps
        or steps_per_epoch * finetune_config.epochs
    )
    warmup_steps = max(1, int(total_steps * finetune_config.warmup_ratio))
    logger.info(
        'finetuning: %d examples, %d total steps', dataset.length(), total_steps
    )

    def learning_rate_at(step):
        if step < warmup_steps:
            return finetune_config.learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return finetune_config.minimum_learning_rate + coefficient * (
            finetune_config.learning_rate
            - finetune_config.minimum_learning_rate
        )

    random_generator = numpy.random.default_rng(config.project.seed)
    order = random_generator.permutation(dataset.length()).tolist()
    cursor = 0

    def next_indices(count):
        nonlocal cursor, order
        chosen = []
        while len(chosen) < count:
            if cursor >= len(order):
                order = random_generator.permutation(dataset.length()).tolist()
                cursor = 0
            chosen.append(order[cursor])
            cursor += 1
        return chosen

    checkpoint_directory = ensure_directory(config.sft_dir)
    model.train()
    for step in range(total_steps):
        current_learning_rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(finetune_config.gradient_accumulation_steps):
            indices = next_indices(finetune_config.batch_size)
            inputs, targets, _ = dataset.collate(indices, device)
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
                'finetune step %d/%d  loss %.4f  lr %.2e',
                step, total_steps, accumulated_loss, current_learning_rate,
            )
        if step > 0 and step % finetune_config.checkpoint_interval == 0:
            _save(model, config, gpt_config, step, checkpoint_directory)

    _save(model, config, gpt_config, total_steps, checkpoint_directory)
    logger.info('finetuning complete -> %s', checkpoint_directory / 'ckpt_last.pt')
    return checkpoint_directory / 'ckpt_last.pt'


def main():
    parser = argparse.ArgumentParser(description='Supervised finetuning')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == '__main__':
    main()

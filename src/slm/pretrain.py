"""Stage three: pretrain the decoder from random initialization.

Single-GPU by default and distributed-data-parallel ready under torchrun for
scaling to multiple L40S GPUs:

    python -m slm.pretrain --config configs/poc.yaml
    torchrun --nproc_per_node=4 -m slm.pretrain --config configs/poc.yaml

Checkpoints are written to the pretrain directory as ckpt_best.pt (lowest
validation loss), ckpt_last.pt (for resume), and periodic snapshots.
"""

import argparse
import json
import math
import time
from contextlib import nullcontext

import numpy

from .config import load_config, to_dict
from .model import GPT, build_config
from .utils import (
    ensure_directory, get_local_rank, get_logger, get_world_size,
    is_distributed, is_main_process, set_seed,
)

logger = get_logger('pretrain')


def learning_rate_at(step, pretrain_config):
    if step < pretrain_config.warmup_steps:
        return pretrain_config.learning_rate * (step + 1) / max(
            1, pretrain_config.warmup_steps
        )
    if step >= pretrain_config.maximum_steps:
        return pretrain_config.minimum_learning_rate
    progress = (step - pretrain_config.warmup_steps) / max(
        1, pretrain_config.maximum_steps - pretrain_config.warmup_steps
    )
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return pretrain_config.minimum_learning_rate + coefficient * (
        pretrain_config.learning_rate - pretrain_config.minimum_learning_rate
    )


def _setup_device():
    import torch
    import torch.distributed as distributed

    if is_distributed():
        distributed.init_process_group(backend='nccl')
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return 'cuda:%d' % local_rank
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def run(config, packed_directory=None, checkpoint_root=None):
    """Pretrain from packed binaries.

    packed_directory and checkpoint_root default to the base pipeline paths;
    the graph pipeline passes its own so both models train with the same loop.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel

    from .data import PackedDataset

    set_seed(config.project.seed + get_local_rank())
    device = _setup_device()
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    pretrain_config = config.pretrain

    if packed_directory is None:
        packed_directory = config.data_dir / 'packed'
    meta = json.loads((packed_directory / 'meta.json').read_text())
    vocabulary_size = meta['vocabulary_size']
    dtype = meta['dtype']

    gpt_config = build_config(config.model, vocabulary_size)
    model = GPT(gpt_config).to(device)
    if is_main_process():
        non_embedding = model.count_parameters(non_embedding=True)
        train_tokens = meta.get('train_tokens', 0)
        ratio = train_tokens / non_embedding if non_embedding else 0.0
        logger.info(
            'model: %.2fM parameters (%.2fM non-embedding, preset %s)',
            model.count_parameters() / 1e6, non_embedding / 1e6,
            config.model.preset,
        )
        logger.info(
            'train tokens %d, tokens per non-embedding parameter %.1f',
            train_tokens, ratio,
        )

    precision = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[pretrain_config.dtype]
    autocast = (
        nullcontext() if device_type == 'cpu'
        else torch.autocast(device_type=device_type, dtype=precision)
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=(pretrain_config.dtype == 'float16')
    )

    base_model = model
    if pretrain_config.compile_model and device_type == 'cuda':
        model = torch.compile(model)
    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[get_local_rank()])
        base_model = model.module

    optimizer = base_model.configure_optimizers(
        pretrain_config.weight_decay,
        pretrain_config.learning_rate,
        (pretrain_config.beta1, pretrain_config.beta2),
        device_type,
    )

    train_dataset = PackedDataset(
        packed_directory / 'train.bin', dtype, gpt_config.block_size
    )
    validation_dataset = PackedDataset(
        packed_directory / 'val.bin', dtype, gpt_config.block_size
    )
    random_generator = numpy.random.default_rng(
        config.project.seed + get_local_rank()
    )

    checkpoint_directory = ensure_directory(
        checkpoint_root if checkpoint_root is not None else config.pretrain_dir
    )
    start_step = 0
    best_validation = float('inf')
    evaluations_since_best = 0
    last_checkpoint = checkpoint_directory / 'ckpt_last.pt'
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device)
        base_model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step'] + 1
        best_validation = saved.get('best_validation', best_validation)
        if is_main_process():
            logger.info('resumed from step %d', start_step)

    def estimate_validation_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(pretrain_config.evaluation_iterations):
                inputs, targets = validation_dataset.get_batch(
                    pretrain_config.batch_size, device, random_generator
                )
                with autocast:
                    _, loss = base_model(inputs, targets)
                losses.append(loss.item())
        model.train()
        return float(numpy.mean(losses))

    def save_checkpoint(step, validation_loss, tag):
        if not is_main_process():
            return
        payload = {
            'model': base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'validation_loss': validation_loss,
            'best_validation': best_validation,
            'model_config': to_dict(config.model),
            'vocabulary_size': vocabulary_size,
        }
        torch.save(payload, checkpoint_directory / ('%s.pt' % tag))

    history_path = checkpoint_directory / 'history.jsonl'
    if start_step == 0 and is_main_process() and history_path.exists():
        history_path.unlink()

    def record_history(step, train_loss, validation_loss):
        if not is_main_process():
            return
        with open(history_path, 'a') as handle:
            handle.write(json.dumps({
                'step': step,
                'train_loss': round(train_loss, 4),
                'validation_loss': round(validation_loss, 4),
            }) + '\n')

    model.train()
    interval_start = time.time()
    step = start_step
    accumulated_loss = 0.0
    for step in range(start_step, pretrain_config.maximum_steps):
        current_learning_rate = learning_rate_at(step, pretrain_config)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(pretrain_config.gradient_accumulation_steps):
            inputs, targets = train_dataset.get_batch(
                pretrain_config.batch_size, device, random_generator
            )
            if is_distributed():
                model.require_backward_grad_sync = (
                    micro_step
                    == pretrain_config.gradient_accumulation_steps - 1
                )
            with autocast:
                _, loss = model(inputs, targets)
                loss = loss / pretrain_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        if pretrain_config.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                base_model.parameters(), pretrain_config.gradient_clip
            )
        scaler.step(optimizer)
        scaler.update()

        if step % pretrain_config.log_interval == 0 and is_main_process():
            elapsed = time.time() - interval_start
            logger.info(
                'step %d/%d  loss %.4f  lr %.2e  %.2fs/it',
                step, pretrain_config.maximum_steps, accumulated_loss,
                current_learning_rate,
                elapsed / max(1, pretrain_config.log_interval),
            )
            interval_start = time.time()

        if step > 0 and step % pretrain_config.evaluation_interval == 0:
            validation_loss = estimate_validation_loss()
            if is_distributed():
                import torch.distributed as distributed

                gathered = torch.tensor(validation_loss, device=device)
                distributed.all_reduce(gathered, op=distributed.ReduceOp.SUM)
                validation_loss = gathered.item() / get_world_size()
            record_history(step, accumulated_loss, validation_loss)
            if validation_loss < best_validation - 1e-4:
                best_validation = validation_loss
                evaluations_since_best = 0
                save_checkpoint(step, validation_loss, 'ckpt_best')
            else:
                evaluations_since_best += 1
            if is_main_process():
                logger.info(
                    'step %d  validation_loss %.4f  perplexity %.2f  '
                    '(no improvement for %d)',
                    step, validation_loss, math.exp(min(validation_loss, 20)),
                    evaluations_since_best,
                )
            if (pretrain_config.early_stop_patience
                    and evaluations_since_best
                    >= pretrain_config.early_stop_patience):
                if is_main_process():
                    logger.info(
                        'early stopping at step %d, validation rising', step
                    )
                break
        if step > 0 and step % pretrain_config.checkpoint_interval == 0:
            save_checkpoint(step, best_validation, 'ckpt_last')

    final_validation = estimate_validation_loss()
    record_history(step, accumulated_loss, final_validation)
    if final_validation < best_validation:
        best_validation = final_validation
        save_checkpoint(pretrain_config.maximum_steps, final_validation, 'ckpt_best')
    save_checkpoint(pretrain_config.maximum_steps, final_validation, 'ckpt_last')
    if is_main_process():
        logger.info('pretrain complete, best validation %.4f', best_validation)
    if is_distributed():
        import torch.distributed as distributed

        distributed.destroy_process_group()
    return checkpoint_directory / 'ckpt_best.pt'


def main():
    parser = argparse.ArgumentParser(description='Pretrain from scratch')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == '__main__':
    main()

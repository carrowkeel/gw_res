"""Expert iteration on a single-turn game: sample, verify, imitate.

The smallest step from proven imitation learning toward outcome-driven
training. The loss is unchanged from the SFT stages - response-masked
cross-entropy on prompt and answer pairs - and only the data source
moves: each round the model answers a fresh batch of game tasks at
sampling temperature, the game's program grades every answer, and the
verified answers (weighted by the game's score, which is exact
correctness unless the game's noise dial is turned) become the round's
training set. The question each run answers is whether the model can
improve from its own verified play, measured on a fixed held-out task
battery graded greedily every round.

Model health is watched as closely as progress, because the simulator
rounds showed degradation arrives through the language before it shows
in scores: every round logs replay loss against the stage-1 corpus
(register drift), repeated-bigram rate and response length over the
round's samples (the loop signature and padding), distinct-response
rate, and the kept fraction. A round that keeps nothing skips its
update and says so; several such rounds in a row abort the run.

    python -m slm.gametrain --config <run>/config.yaml
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from .config import load_config, to_dict
from .games import load_game
from .sfteval import repeated_bigram_rate
from .sftstage import (
    load_checkpoint_model, resolve_base_checkpoint, resolve_checkpoint,
)
from .tokenizer import SyntheticTokenizer
from .tokenizer import fingerprint as tokenizer_fingerprint
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('gametrain')

HOLDOUT_SEED_OFFSET = 771111


def _stage_name(gametrain_config):
    return 'gametrain-%s' % gametrain_config.game


def _base_paths(config):
    gametrain_config = config.gametrain
    if not gametrain_config.base_run_dir:
        raise SystemExit('gametrain.base_run_dir must name the stage-1 tree')
    base = Path(gametrain_config.base_run_dir)
    if gametrain_config.base_stage:
        checkpoint = resolve_checkpoint(
            base / 'checkpoints' / gametrain_config.base_stage
        )
        if checkpoint is None:
            raise SystemExit('base_stage %s has no checkpoint under %s'
                             % (gametrain_config.base_stage,
                                base / 'checkpoints'))
    else:
        checkpoint = resolve_base_checkpoint(base)
    return {
        'tokenizer': base / 'tokenizer' / 'tokenizer.json',
        'checkpoint': checkpoint,
        'packed': base / 'data' / 'packed',
    }


def _load_replay_blocks(paths, block_size, count, seed):
    """Fixed stage-1 blocks whose loss is the register-drift instrument."""
    import numpy

    from .data import PackedDataset

    meta_path = paths['packed'] / 'meta.json'
    train_path = paths['packed'] / 'train.bin'
    if not meta_path.exists() or not train_path.exists():
        logger.warning('no packed stage-1 data at %s, replay drift '
                       'disabled', paths['packed'])
        return None
    meta = json.loads(meta_path.read_text())
    dataset = PackedDataset(train_path, meta['dtype'], block_size)
    inputs, targets = dataset.get_batch(
        count, 'cpu', numpy.random.default_rng(seed)
    )
    return inputs, targets


def _replay_loss(model, blocks, device, chunk=4):
    import torch

    if blocks is None:
        return None
    inputs, targets = blocks
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, inputs.size(0), chunk):
            _, loss = model(
                inputs[start:start + chunk].to(device),
                targets[start:start + chunk].to(device),
            )
            losses.append(loss.item())
    return round(statistics.mean(losses), 4)


def _sample_answers(model, tokenizer, prompt, count, gametrain_config,
                    block_size, device):
    """Sample count answers to one prompt in a single batched call."""
    import torch

    token_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    limit = block_size - gametrain_config.max_new_tokens
    token_ids = token_ids[-limit:]
    input_ids = torch.tensor([token_ids] * count, dtype=torch.long,
                             device=device)
    output = model.generate(
        input_ids, gametrain_config.max_new_tokens,
        temperature=gametrain_config.sample_temperature,
        top_p=gametrain_config.sample_top_p,
        eos_id=tokenizer.eos_id,
    )
    answers = []
    for row in output[:, len(token_ids):].tolist():
        if tokenizer.eos_id in row:
            row = row[:row.index(tokenizer.eos_id)]
        answers.append(tokenizer.decode(row).split('\n')[0].strip())
    return answers


def _greedy_answer(model, tokenizer, prompt, gametrain_config, block_size,
                   device):
    import torch

    token_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    limit = block_size - gametrain_config.max_new_tokens
    token_ids = token_ids[-limit:]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model.generate(
        input_ids, gametrain_config.max_new_tokens, temperature=0.0,
        eos_id=tokenizer.eos_id,
    )
    row = output[0, len(token_ids):].tolist()
    if tokenizer.eos_id in row:
        row = row[:row.index(tokenizer.eos_id)]
    return tokenizer.decode(row).split('\n')[0].strip()


def _holdout_solve(model, tokenizer, game, tasks, gametrain_config,
                   block_size, device):
    """Greedy solve rate on the fixed battery, overall and by kind."""
    import torch

    model.eval()
    by_kind = {}
    solved = 0
    with torch.no_grad():
        for task in tasks:
            answer = _greedy_answer(model, tokenizer, task['prompt'],
                                    gametrain_config, block_size, device)
            correct = game.verify(task, answer)
            solved += int(correct)
            entry = by_kind.setdefault(task['kind'], [0, 0])
            entry[0] += 1
            entry[1] += int(correct)
    rate = round(solved / len(tasks), 4) if tasks else 0.0
    kinds = {kind: round(entry[1] / entry[0], 4)
             for kind, entry in sorted(by_kind.items())}
    return rate, kinds


def _pad_batch(examples, pad_id, device):
    import torch

    longest = max(len(ids) for ids, _, _ in examples)
    inputs = torch.full((len(examples), longest), pad_id, dtype=torch.long)
    targets = torch.full((len(examples), longest), -1, dtype=torch.long)
    weights = torch.zeros(len(examples))
    for index, (ids, response_from, weight) in enumerate(examples):
        inputs[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        shifted = ids[1:] + [pad_id]
        row = torch.tensor(shifted, dtype=torch.long)
        row[:max(0, response_from - 1)] = -1
        row[len(ids) - 1:] = -1
        targets[index, :len(ids)] = row
        weights[index] = weight
    return inputs.to(device), targets.to(device), weights.to(device)


def _train_round(model, optimizer, examples, gametrain_config, pad_id,
                 device):
    """One round of weighted response-masked SFT on the kept answers."""
    import torch
    from torch.nn import functional

    model.train()
    order = list(examples)
    shuffler = random.Random(len(examples))
    losses = []
    for _ in range(gametrain_config.epochs_per_round):
        shuffler.shuffle(order)
        for start in range(0, len(order), gametrain_config.batch_size):
            batch = order[start:start + gametrain_config.batch_size]
            inputs, targets, weights = _pad_batch(batch, pad_id, device)
            logits, _ = model(inputs, targets, ignore_index=-1)
            flat = functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction='none',
            ).view(targets.size())
            mask = (targets != -1).float()
            per_example = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = (per_example * weights).sum() / weights.sum().clamp(min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gametrain_config.gradient_clip
            )
            optimizer.step()
            losses.append(loss.item())
    return round(statistics.mean(losses), 4) if losses else None


def run(config):
    import torch

    gametrain_config = config.gametrain
    set_seed(config.project.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    game = load_game(gametrain_config.game)
    parameters = game.resolve_parameters(gametrain_config.game_parameters)

    paths = _base_paths(config)
    tokenizer = SyntheticTokenizer(paths['tokenizer'])
    model, gpt_config = load_checkpoint_model(
        config, paths['checkpoint'], device,
        tokenizer_path=paths['tokenizer'],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gametrain_config.learning_rate,
        weight_decay=gametrain_config.weight_decay,
    )

    checkpoint_dir = ensure_directory(
        config.out_dir / 'checkpoints' / _stage_name(gametrain_config)
    )
    history_path = checkpoint_dir / 'history.jsonl'
    replay_blocks = _load_replay_blocks(
        paths, gpt_config.block_size, gametrain_config.replay_blocks,
        config.project.seed + 17,
    )
    holdout_random = random.Random(
        gametrain_config.battery_seed + HOLDOUT_SEED_OFFSET
    )
    holdout_tasks = game.generate_tasks(
        parameters, holdout_random, gametrain_config.holdout_tasks
    )
    replay_start = _replay_loss(model, replay_blocks, device)

    def save(step, name, solve_rate):
        torch.save({
            'model': model.state_dict(), 'step': step,
            'holdout_solve_rate': solve_rate,
            'game': gametrain_config.game,
            'game_parameters': parameters,
            'model_config': to_dict(config.model),
            'vocabulary_size': tokenizer.vocabulary_size,
            'tokenizer_fingerprint': tokenizer_fingerprint(
                paths['tokenizer']),
        }, checkpoint_dir / name)

    best_solve = -1.0
    empty_rounds = 0
    round_index = -1
    with open(history_path, 'w') as history:
        for round_index in range(gametrain_config.rounds):
            task_random = random.Random(
                config.project.seed + 30011 * (round_index + 1)
            )
            tasks = game.generate_tasks(
                parameters, task_random, gametrain_config.tasks_per_round
            )
            examples = []
            kept = 0
            correct_samples = 0
            solved_tasks = 0
            responses = []
            model.eval()
            with torch.no_grad():
                for task in tasks:
                    answers = _sample_answers(
                        model, tokenizer, task['prompt'],
                        gametrain_config.samples_per_task,
                        gametrain_config, gpt_config.block_size, device,
                    )
                    responses.extend(answers)
                    verified = [game.verify(task, answer)
                                for answer in answers]
                    correct_samples += sum(verified)
                    solved_tasks += int(any(verified))
                    seen = set()
                    for answer in answers:
                        weight = game.score(task, answer, parameters,
                                            task_random)
                        if weight <= 0 or answer in seen:
                            continue
                        seen.add(answer)
                        prompt_ids = [tokenizer.bos_id] + tokenizer.encode(
                            task['prompt'])
                        full_ids = prompt_ids + tokenizer.encode(
                            ' ' + answer) + [tokenizer.eos_id]
                        if len(full_ids) > gpt_config.block_size:
                            continue
                        examples.append(
                            (full_ids, len(prompt_ids), weight))
                        kept += 1
                        if len(seen) >= gametrain_config.keep_per_task:
                            break
            train_loss = None
            if examples:
                empty_rounds = 0
                train_loss = _train_round(
                    model, optimizer, examples, gametrain_config,
                    tokenizer.pad_id, device,
                )
            else:
                empty_rounds += 1
                logger.warning('round %d kept nothing; skipping update '
                               '(%d empty in a row)',
                               round_index, empty_rounds)
            solve_rate, solve_by_kind = _holdout_solve(
                model, tokenizer, game, holdout_tasks, gametrain_config,
                gpt_config.block_size, device,
            )
            replay = _replay_loss(model, replay_blocks, device)
            total = len(tasks) * gametrain_config.samples_per_task
            row = {
                'round': round_index,
                'sampled_solve_rate': round(correct_samples / total, 4),
                'task_solve_rate': round(solved_tasks / len(tasks), 4),
                'holdout_solve_rate': solve_rate,
                'holdout_by_kind': solve_by_kind,
                'kept': kept,
                'kept_fraction': round(kept / total, 4),
                'train_loss': train_loss,
                'replay_loss': replay,
                'replay_drift': (
                    round(replay - replay_start, 4)
                    if replay is not None and replay_start is not None
                    else None
                ),
                'distinct_response_rate': round(
                    len(set(responses)) / max(1, len(responses)), 4),
                'repeated_bigram_rate': round(statistics.mean(
                    repeated_bigram_rate(response)
                    for response in responses), 4) if responses else None,
                'mean_response_length': round(statistics.mean(
                    len(response) for response in responses), 1)
                    if responses else None,
            }
            history.write(json.dumps(row) + '\n')
            history.flush()
            logger.info(
                'round %d/%d  sampled %.3f  task %.3f  holdout %.3f  '
                'kept %d (%.2f)  loss %s  replay %s (drift %s)  '
                'distinct %.2f  bigram %.3f',
                round_index + 1, gametrain_config.rounds,
                row['sampled_solve_rate'], row['task_solve_rate'],
                solve_rate, kept, row['kept_fraction'],
                train_loss, replay, row['replay_drift'],
                row['distinct_response_rate'],
                row['repeated_bigram_rate'] or 0.0,
            )
            if solve_rate > best_solve:
                best_solve = solve_rate
                save(round_index, 'ckpt_best.pt', solve_rate)
            save(round_index, 'ckpt_last.pt', solve_rate)
            if empty_rounds >= gametrain_config.empty_round_limit:
                logger.warning('aborting: %d empty rounds in a row',
                               empty_rounds)
                break
    report = {
        'game': gametrain_config.game,
        'game_parameters': parameters,
        'rounds': round_index + 1,
        'best_holdout_solve_rate': best_solve,
        'replay_start': replay_start,
    }
    with open(checkpoint_dir / 'game_report.json', 'w') as handle:
        json.dump(report, handle, indent=2)
    logger.info('game training complete: best holdout solve %.3f -> %s',
                best_solve, checkpoint_dir)
    return best_solve


def main():
    parser = argparse.ArgumentParser(
        description='Expert iteration on a verifiable single-turn game'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    arguments = parser.parse_args()
    run(load_config(arguments.config, run_id=arguments.run_id))


if __name__ == '__main__':
    main()

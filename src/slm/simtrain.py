"""Stage-2 online trainer: play the market, score, update.

Each training step plays a batch of independent games in lockstep: every
quarter the rendered context is extended, the model generates its trader
turn, the listener gates and interprets it into orders, and the simulator
resolves the quarter. One game is one training sequence, so the context
carries the evolving game and the model's own earlier decisions. The
update is score-weighted cross-entropy on the trader turns only: each
quarter's earnings are normalized across the whole batch and exponentiated
into a per-quarter weight on that turn's tokens, so the model is pulled
toward the decisions that scored well. There are no gold actions anywhere.
A replay fraction of stage-1 packed text keeps language anchored at the
gradient level.

The base model, tokenizer, and replay data come from a completed stage-1
run via simtrain.base_run_dir; without it the model starts from random
initialization, which is only useful for smoke tests.

    python -m slm.simtrain --config configs/sim.yaml
"""

import argparse
import json
import math
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy

from . import listener as listener_module
from . import market, render
from .config import load_config, to_dict
from .model import GPT, build_config
from .pretrain import learning_rate_at
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, normalize_state_dict, set_seed

logger = get_logger('simtrain')


def _base_paths(config):
    simtrain_config = config.simtrain
    if simtrain_config.base_run_dir:
        base = Path(simtrain_config.base_run_dir)
        return {
            'tokenizer': base / 'tokenizer' / 'tokenizer.json',
            'checkpoint': base / 'checkpoints' / 'pretrain' / 'ckpt_best.pt',
            'packed': base / 'data' / 'packed',
        }
    return {
        'tokenizer': config.tokenizer_path,
        'checkpoint': None,
        'packed': None,
    }


def _load_replay(paths, block_size):
    from .data import PackedDataset

    if paths['packed'] is None:
        return None
    meta_path = paths['packed'] / 'meta.json'
    train_path = paths['packed'] / 'train.bin'
    if not meta_path.exists() or not train_path.exists():
        logger.warning('no packed stage-1 data at %s, replay disabled',
                       paths['packed'])
        return None
    meta = json.loads(meta_path.read_text())
    return PackedDataset(train_path, meta['dtype'], block_size)


def _generate_decisions(model, tokenizer, games, simtrain_config,
                        block_size, device):
    """Generate every game's trader turn in one batched call.

    Sequential per-game generation is prohibitively slow above pico scale,
    since each token recomputes the full context. All contexts are cropped
    to the batch's shortest (dropping the oldest tokens of longer games,
    which the block-size crop was discarding from anyway) so a single
    batched generate serves the whole lockstep quarter.
    """
    import torch

    limit = block_size - simtrain_config.max_decision_tokens
    contexts = [game['token_ids'][-limit:] for game in games]
    shortest = min(len(context) for context in contexts)
    contexts = [context[-shortest:] for context in contexts]
    input_ids = torch.tensor(contexts, dtype=torch.long, device=device)
    output = model.generate(
        input_ids, simtrain_config.max_decision_tokens,
        temperature=simtrain_config.sample_temperature,
        top_p=simtrain_config.sample_top_p,
        eos_id=tokenizer.eos_id,
    )
    decisions = []
    for row in output[:, shortest:].tolist():
        if tokenizer.eos_id in row:
            row = row[:row.index(tokenizer.eos_id)]
        text = tokenizer.decode(row)
        decisions.append(text.split('\n')[0].strip())
    return decisions


def _play_batch(model, tokenizer, config, llm_listener, step, block_size,
                device):
    """Play games_per_batch games in lockstep; return games and turn stats.

    Lockstep (all games advance one quarter together) exists so the llm
    listener can interpret every game's turn in one batched call.
    """
    simtrain_config = config.simtrain
    games = []
    for game_index in range(simtrain_config.games_per_batch):
        game_random = random.Random(
            config.project.seed + step * 100003 + game_index
        )
        game_market = market.sample_market(
            game_random, simtrain_config.field_count,
            simtrain_config.companies_per_field,
        )
        games.append({
            'random': game_random,
            'market': game_market,
            'state': market.start_game(game_market, game_random),
            'token_ids': [tokenizer.bos_id],
            'spans': [],
            'earnings': [],
            'turn_records': [],
        })
    stats = {'turns': 0, 'no_reason': 0, 'acted': 0,
             'match_exact': 0, 'match_fuzzy': 0, 'match_none': 0,
             'advisor_earnings': [], 'no_advisor_earnings': []}
    gate_random = random.Random(config.project.seed + step * 100003 + 7)
    sample_turn = None
    for quarter in range(simtrain_config.quarters):
        for game in games:
            block = render.render_quarter(
                game['state'], game['market'], game['random'],
                protocol_line=simtrain_config.protocol_line,
            )
            prefix = ('\n' if quarter else '') + block
            game['token_ids'].extend(tokenizer.encode(prefix))
        decisions = _generate_decisions(
            model, tokenizer, games, simtrain_config, block_size, device,
        )
        turns = []
        for game, decision_text in zip(games, decisions):
            decision_ids = tokenizer.encode(' ' + decision_text)
            span_start = len(game['token_ids'])
            game['token_ids'].extend(decision_ids)
            game['spans'].append((span_start, len(game['token_ids'])))
            turns.append((decision_text, game['market'], game['state']))
        if llm_listener is not None:
            results = llm_listener.interpret_batch(
                turns, simtrain_config.no_reason_action_probability,
                gate_random,
            )
        else:
            results = [
                listener_module.interpret(
                    text, turn_market, turn_state,
                    simtrain_config.no_reason_action_probability,
                    gate_random,
                )
                for text, turn_market, turn_state in turns
            ]
        if sample_turn is None and turns:
            sample_turn = (turns[0][0], results[0])
        for game, turn, result in zip(games, turns, results):
            stats['turns'] += 1
            if not result['reason_given']:
                stats['no_reason'] += 1
            if result['acted']:
                stats['acted'] += 1
            stats['match_%s' % result['match']] += 1
            advisor_present = any(
                report['source'] == 'advisor'
                for report in game['state']['reports']
            )
            earnings, executed = market.step_game(
                game['market'], game['state'], result['actions'],
                game['random'],
            )
            game['earnings'].append(earnings)
            if advisor_present:
                stats['advisor_earnings'].append(earnings)
            else:
                stats['no_advisor_earnings'].append(earnings)
            game['turn_records'].append({
                'quarter': quarter + 1,
                'decision': turn[0],
                'rewrite': result['rewrite'],
                'reason_given': result['reason_given'],
                'match': result['match'],
                'executed': executed,
                'advisor_present': advisor_present,
                'earnings': round(earnings, 2),
            })
    return games, stats, sample_turn


def _batch_tensors(games, simtrain_config, block_size, device):
    """Turn played games into padded input, target, and weight tensors.

    Quarter earnings are normalized across the entire batch and mapped
    through a clipped exponential into per-quarter weights; only trader
    turn tokens carry weight, so the loss trains behavior while the
    rendered context is read but never imitated.
    """
    import torch

    all_earnings = [value for game in games for value in game['earnings']]
    mean = statistics.mean(all_earnings)
    spread = statistics.pstdev(all_earnings)
    spread = spread if spread > 1e-6 else 1.0
    rows = []
    for game in games:
        token_ids = game['token_ids']
        offset = max(0, len(token_ids) - (block_size + 1))
        token_ids = token_ids[offset:]
        weights = [0.0] * (len(token_ids) - 1)
        for (span_start, span_end), earnings in zip(
                game['spans'], game['earnings']):
            normalized = (earnings - mean) / spread
            weight = math.exp(
                normalized / simtrain_config.weight_temperature
            )
            weight = min(weight, simtrain_config.weight_clip)
            for position in range(span_start - offset - 1,
                                  span_end - offset - 1):
                if 0 <= position < len(weights):
                    weights[position] = weight
        rows.append((token_ids, weights))
    longest = max(len(token_ids) for token_ids, _ in rows)
    inputs = torch.zeros((len(rows), longest - 1), dtype=torch.long)
    targets = torch.zeros((len(rows), longest - 1), dtype=torch.long)
    weight_tensor = torch.zeros((len(rows), longest - 1))
    for row_index, (token_ids, weights) in enumerate(rows):
        length = len(token_ids) - 1
        inputs[row_index, :length] = torch.tensor(token_ids[:-1])
        targets[row_index, :length] = torch.tensor(token_ids[1:])
        weight_tensor[row_index, :length] = torch.tensor(weights)
    total = weight_tensor.sum()
    if total > 0:
        weight_tensor = weight_tensor * (
            (weight_tensor > 0).sum() / total
        )
    return (inputs.to(device), targets.to(device), weight_tensor.to(device))


def _reference_returns(simtrain_config, seed, sample_games=200):
    blind = statistics.mean(
        market.play_game(market.blind_policy, seed + index,
                         simtrain_config.quarters)[0]
        for index in range(sample_games)
    )
    oracle = statistics.mean(
        market.play_game(market.oracle_policy, seed + index,
                         simtrain_config.quarters)[0]
        for index in range(sample_games)
    )
    return blind, oracle


def run(config):
    import torch
    from torch.nn import functional

    simtrain_config = config.simtrain
    set_seed(config.project.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    paths = _base_paths(config)
    tokenizer = SyntheticTokenizer(paths['tokenizer'])
    vocabulary_size = tokenizer.vocabulary_size

    checkpoint = None
    if paths['checkpoint'] is not None and paths['checkpoint'].exists():
        checkpoint = torch.load(paths['checkpoint'], map_location=device)
        vocabulary_size = checkpoint['vocabulary_size']
        logger.info('starting from stage-1 checkpoint %s',
                    paths['checkpoint'])
    else:
        logger.warning('no stage-1 checkpoint, starting from random '
                       'initialization (smoke-test mode)')

    gpt_config = build_config(config.model, vocabulary_size)
    model = GPT(gpt_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(normalize_state_dict(checkpoint['model']))
    block_size = gpt_config.block_size
    logger.info('model: %.2fM parameters, block size %d',
                model.count_parameters() / 1e6, block_size)

    replay = None
    if simtrain_config.replay_fraction > 0:
        replay = _load_replay(paths, block_size)
    replay_random = numpy.random.default_rng(config.project.seed + 11)

    llm_listener = None
    if simtrain_config.listener_mode == 'llm':
        model_name = (simtrain_config.listener_model
                      or config.generate.default_model)
        llm_listener = listener_module.LlmListener(
            model_name, config.generate
        )
        logger.info('llm listener: %s', model_name)

    precision = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[simtrain_config.dtype]
    autocast = (
        nullcontext() if device_type == 'cpu'
        else torch.autocast(device_type=device_type, dtype=precision)
    )
    base_model = model
    if simtrain_config.compile_model and device_type == 'cuda':
        model = torch.compile(model)

    optimizer = base_model.configure_optimizers(
        simtrain_config.weight_decay, simtrain_config.learning_rate,
        (simtrain_config.beta1, simtrain_config.beta2), device_type,
    )

    checkpoint_directory = ensure_directory(config.simtrain_dir)
    start_step = 0
    last_checkpoint = checkpoint_directory / 'ckpt_last.pt'
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device)
        base_model.load_state_dict(normalize_state_dict(saved['model']))
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step'] + 1
        logger.info('resumed from step %d', start_step)

    blind_reference, oracle_reference = _reference_returns(
        simtrain_config, config.project.seed + 999983
    )
    logger.info('reference returns per game: blind %+.1f, oracle %+.1f',
                blind_reference, oracle_reference)

    def save_checkpoint(step, tag, mean_return):
        payload = {
            'model': base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'mean_return': mean_return,
            'model_config': to_dict(config.model),
            'vocabulary_size': vocabulary_size,
        }
        torch.save(payload, checkpoint_directory / ('%s.pt' % tag))

    history_path = checkpoint_directory / 'history.jsonl'
    if start_step == 0 and history_path.exists():
        history_path.unlink()

    recent_returns = []
    best_rolling = -float('inf')
    interval_start = time.time()
    for step in range(start_step, simtrain_config.maximum_steps):
        current_learning_rate = learning_rate_at(step, simtrain_config)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate

        model.eval()
        games, stats, sample_turn = _play_batch(
            model, tokenizer, config, llm_listener, step, block_size, device
        )
        model.train()

        all_earnings = [
            value for game in games for value in game['earnings']
        ]
        flat_signal = statistics.pstdev(all_earnings) < 1e-6
        optimizer.zero_grad(set_to_none=True)
        game_loss = None
        replay_loss = None
        loss = None
        weights = None
        with autocast:
            if not flat_signal:
                inputs, targets, weights = _batch_tensors(
                    games, simtrain_config, block_size, device
                )
                logits, _ = model(inputs, targets)
                per_token = functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    reduction='none',
                ).view_as(weights)
                game_loss = (per_token * weights).sum() / weights.sum().clamp(
                    min=1.0
                )
                loss = game_loss
            if replay is not None:
                replay_inputs, replay_targets = replay.get_batch(
                    max(1, int(simtrain_config.games_per_batch
                               * simtrain_config.replay_fraction)),
                    device, replay_random,
                )
                _, replay_loss = model(replay_inputs, replay_targets)
                if loss is None:
                    loss = replay_loss
                else:
                    loss = (
                        (1.0 - simtrain_config.replay_fraction) * game_loss
                        + simtrain_config.replay_fraction * replay_loss
                    )
        if loss is not None:
            loss.backward()
            if simtrain_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), simtrain_config.gradient_clip
                )
            optimizer.step()

        mean_return = statistics.mean(
            sum(game['earnings']) for game in games
        )
        recent_returns.append(mean_return)
        if len(recent_returns) > simtrain_config.best_window:
            recent_returns.pop(0)
        rolling = statistics.mean(recent_returns)

        row = {
            'step': step,
            'flat_signal': flat_signal,
            'mean_return': round(mean_return, 2),
            'rolling_return': round(rolling, 2),
            'no_reason_rate': round(stats['no_reason'] / stats['turns'], 3),
            'acted_rate': round(stats['acted'] / stats['turns'], 3),
            'match_exact_rate': round(
                stats['match_exact'] / stats['turns'], 3),
            'match_fuzzy_rate': round(
                stats['match_fuzzy'] / stats['turns'], 3),
        }
        if loss is not None:
            row['loss'] = round(loss.item(), 4)
        if game_loss is not None:
            row['game_loss'] = round(game_loss.item(), 4)
            positive_weights = weights[weights > 0]
            row['weight_mean'] = (round(positive_weights.mean().item(), 3)
                                  if len(positive_weights) else 0.0)
            row['weight_max'] = (round(positive_weights.max().item(), 3)
                                 if len(positive_weights) else 0.0)
        if replay_loss is not None:
            row['replay_loss'] = round(replay_loss.item(), 4)
        if stats['advisor_earnings']:
            row['return_with_advisor'] = round(
                statistics.mean(stats['advisor_earnings']), 2)
        if stats['no_advisor_earnings']:
            row['return_without_advisor'] = round(
                statistics.mean(stats['no_advisor_earnings']), 2)
        with open(history_path, 'a') as handle:
            handle.write(json.dumps(row) + '\n')

        if (simtrain_config.transcript_interval
                and step % simtrain_config.transcript_interval == 0):
            with open(checkpoint_directory / 'transcripts.jsonl',
                      'a') as handle:
                for game in games[:simtrain_config.transcript_games]:
                    handle.write(json.dumps({
                        'step': step,
                        'text': tokenizer.decode(game['token_ids']),
                        'turns': game['turn_records'],
                        'total_earnings': round(sum(game['earnings']), 2),
                    }, ensure_ascii=False) + '\n')

        if step % simtrain_config.log_interval == 0:
            elapsed = time.time() - interval_start
            logger.info(
                'step %d/%d  game %s  replay %s%s  return %+.1f (rolling '
                '%+.1f, blind %+.1f, oracle %+.1f)  no-reason %.2f  acted '
                '%.2f  match %.2f/%.2f  %.2fs/it',
                step, simtrain_config.maximum_steps,
                '%.3f' % game_loss.item() if game_loss is not None else '-',
                '%.3f' % replay_loss.item() if replay_loss is not None
                else '-',
                '  FLAT-SIGNAL (replay-only update)' if flat_signal else '',
                mean_return, rolling, blind_reference, oracle_reference,
                stats['no_reason'] / stats['turns'],
                stats['acted'] / stats['turns'],
                stats['match_exact'] / stats['turns'],
                stats['match_fuzzy'] / stats['turns'],
                elapsed / max(1, simtrain_config.log_interval),
            )
            if sample_turn is not None:
                logger.info('sample turn: %r -> %s',
                            sample_turn[0][:120], sample_turn[1]['actions'])
            interval_start = time.time()

        if (len(recent_returns) >= simtrain_config.best_window
                and rolling > best_rolling):
            best_rolling = rolling
            save_checkpoint(step, 'ckpt_best', rolling)
        if step > 0 and step % simtrain_config.checkpoint_interval == 0:
            save_checkpoint(step, 'ckpt_last', rolling)

    save_checkpoint(simtrain_config.maximum_steps - 1, 'ckpt_last',
                    statistics.mean(recent_returns) if recent_returns
                    else 0.0)
    logger.info('simulation training complete, best rolling return %+.1f',
                best_rolling)
    return checkpoint_directory / 'ckpt_best.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Stage-2 online simulation training'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--base-run', default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config, run_id=arguments.run_id)
    if arguments.base_run:
        config.simtrain.base_run_dir = arguments.base_run
    run(config)


if __name__ == '__main__':
    main()

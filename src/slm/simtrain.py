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

Language reinforcement guards against the register eroding under
score-weighted self-imitation, with prevention ahead of cure. Prevention:
a turn is eligible for imitation weight only if it executed, gave a
reason, and cleared the imitation score floor when scored - lucky garbage
acts in the world but is never imitated. Cure: in llm listener mode the
listener scores each turn's grammar and coherence and returns a minimal
correction - repetition and pathologies repaired, the trader's own words
kept - and corrections of low-scoring turns enter a rolling buffer only
when they themselves carry a reason and state a valid move, so a
degenerate policy cannot refill the buffer with its own collapse. The
buffer is sampled uniformly at random: selection by move score would
couple language training to earnings luck, and the outcome pathway
already owns the decision signal. The first full window of scores sets a
baseline; when the windowed mean falls language_score_drop below it,
correction batches mix into the update, including on no-signal steps,
where imitative repair is the only way out of the degrade-into-silence
spiral. A rehearsal fraction of bridge records anchors the register at
the gradient level the way stage-1 replay anchors base prose.

The base model, tokenizer, and replay data come from a completed stage-1
run via simtrain.base_run_dir; without it the model starts from random
initialization, which is only useful for smoke tests.

    python -m slm.simtrain --config configs/sim.yaml
"""

import argparse
import json
import math
import random
import re
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
from .sftstage import checkpoint_model_config, verify_checkpoint_tokenizer
from .tokenizer import SyntheticTokenizer, fingerprint as tokenizer_fingerprint
from .utils import ensure_directory, get_logger, normalize_state_dict, set_seed

logger = get_logger('simtrain')


def _base_paths(config):
    from .sftstage import resolve_base_checkpoint

    simtrain_config = config.simtrain
    if simtrain_config.base_run_dir:
        base = Path(simtrain_config.base_run_dir)
        return {
            'tokenizer': base / 'tokenizer' / 'tokenizer.json',
            'checkpoint': resolve_base_checkpoint(base),
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


def _load_rehearsal(simtrain_config, tokenizer, block_size):
    """Bridge records as a rehearsal dataset anchoring the register.

    Stage-1 replay anchors base prose but the reason-bearing register
    lives only in the bridge pools, so a fraction of bridge records is
    rehearsed with response-only loss through the same dataset the
    bridging stage trained on.
    """
    from .sftstage import PlainPairDataset

    if not simtrain_config.base_run_dir:
        return None
    records_path = (Path(simtrain_config.base_run_dir) / 'data'
                    / 'bridgesft' / 'train.jsonl')
    if not records_path.exists():
        logger.warning('no bridge records at %s, rehearsal disabled',
                       records_path)
        return None
    records = []
    with open(records_path) as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= simtrain_config.rehearsal_records:
                break
    dataset = PlainPairDataset(records, tokenizer, block_size)
    logger.info('rehearsal: %d bridge records from %s', dataset.length(),
                records_path)
    return dataset


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


def _difficulty_at(simtrain_config, step):
    """Return (field_count, companies_per_field) for a step.

    The curriculum is a list of rungs ({from_step, field_count,
    companies_per_field}); the last rung whose from_step has been reached
    wins, and the base config values apply before any rung. Because every
    step samples fresh markets, difficulty scales online with no cost.
    """
    field_count = simtrain_config.field_count
    companies_per_field = simtrain_config.companies_per_field
    for rung in sorted(simtrain_config.curriculum,
                       key=lambda rung: rung.get('from_step', 0)):
        if step >= rung.get('from_step', 0):
            field_count = rung.get('field_count', field_count)
            companies_per_field = rung.get(
                'companies_per_field', companies_per_field
            )
    return field_count, companies_per_field


def _admissible_correction(correction, decision, turn_market, turn_state):
    """Whether a correction may enter the buffer: the interface form only.

    It must keep to the turn's numbers (drop, never invent), carry a
    reason, and state a valid move - a parseable order or a stated hold.
    Without this floor the buffer tracks a degenerate policy downward:
    a repetitive but grammatical turn survives minimal correction intact
    and returns as an imitation target of its own collapse.
    """
    corrected = set(re.findall(r'\d+', correction))
    original = set(re.findall(r'\d+', decision))
    if not corrected <= original:
        return False
    if not listener_module.reason_given(correction):
        return False
    actions, _ = listener_module.parse_orders(
        correction, turn_market, turn_state
    )
    return bool(actions) or listener_module.hold_stated(correction)


def _play_batch(model, tokenizer, config, llm_listener, step, block_size,
                device, correction_buffer=None):
    """Play games_per_batch games in lockstep; return games and turn stats.

    Lockstep (all games advance one quarter together) exists so the llm
    listener can interpret every game's turn in one batched call. When a
    correction buffer is given, admissible corrections of low-scoring
    turns are appended to it together with their game context.

    With market_repeats above one, consecutive games share a world seed:
    the market's randomness consumption is action-independent, so repeats
    see identical shocks, prices, and reports, and differ only in the
    model's sampled decisions. Rendering draws from a separate per-game
    generator so wording variation cannot desynchronize the worlds.
    """
    simtrain_config = config.simtrain
    field_count, companies_per_field = _difficulty_at(simtrain_config, step)
    repeats = max(1, simtrain_config.market_repeats)
    games = []
    for game_index in range(simtrain_config.games_per_batch):
        world_index = game_index // repeats
        world_random = random.Random(
            config.project.seed + step * 100003 + world_index * 1009
        )
        game_random = random.Random(
            config.project.seed + step * 100003 + game_index + 500009
        )
        game_market = market.sample_market(
            world_random, field_count, companies_per_field,
        )
        games.append({
            'random': game_random,
            'world_random': world_random,
            'world': world_index,
            'market': game_market,
            'state': market.start_game(game_market, world_random),
            'token_ids': [tokenizer.bos_id],
            'spans': [],
            'earnings': [],
            'acted': [],
            'eligible': [],
            'turn_records': [],
        })
    stats = {'turns': 0, 'no_reason': 0, 'acted': 0, 'eligible': 0,
             'match_exact': 0, 'match_fuzzy': 0, 'match_none': 0,
             'advisor_earnings': [], 'no_advisor_earnings': [],
             'language_scores': [], 'decisions': [],
             'corrections_offered': 0, 'corrections_admitted': 0,
             'generate_seconds': 0.0, 'listener_seconds': 0.0}
    gate_random = random.Random(config.project.seed + step * 100003 + 7)
    no_reason_probability = simtrain_config.no_reason_action_probability
    if simtrain_config.no_reason_anneal_steps > 0:
        no_reason_probability *= max(
            0.0, 1.0 - step / simtrain_config.no_reason_anneal_steps
        )
    sample_turn = None
    for quarter in range(simtrain_config.quarters):
        for game in games:
            block = render.render_quarter(
                game['state'], game['market'], game['random'],
                protocol_line=simtrain_config.protocol_line,
                exemplar_turn=simtrain_config.exemplar_turn,
            )
            prefix = ('\n' if quarter else '') + block
            game['token_ids'].extend(tokenizer.encode(prefix))
        phase_started = time.time()
        decisions = _generate_decisions(
            model, tokenizer, games, simtrain_config, block_size, device,
        )
        stats['generate_seconds'] += time.time() - phase_started
        turns = []
        for game, decision_text in zip(games, decisions):
            decision_ids = tokenizer.encode(' ' + decision_text)
            span_start = len(game['token_ids'])
            game['token_ids'].extend(decision_ids)
            game['spans'].append((span_start, len(game['token_ids'])))
            turns.append((decision_text, game['market'], game['state']))
        phase_started = time.time()
        if llm_listener is not None:
            results = llm_listener.interpret_batch(
                turns, no_reason_probability, gate_random,
            )
        else:
            results = [
                listener_module.interpret(
                    text, turn_market, turn_state,
                    no_reason_probability, gate_random,
                )
                for text, turn_market, turn_state in turns
            ]
        stats['listener_seconds'] += time.time() - phase_started
        if sample_turn is None and turns:
            sample_turn = (turns[0][0], results[0])
        for game, turn, result in zip(games, turns, results):
            stats['turns'] += 1
            stats['decisions'].append(turn[0])
            if not result['reason_given']:
                stats['no_reason'] += 1
            if result['acted']:
                stats['acted'] += 1
            stats['match_%s' % result['match']] += 1
            score = result['language_score']
            correction = result['correction']
            if score is not None:
                stats['language_scores'].append(score)
            if (correction_buffer is not None and correction
                    and score is not None
                    and score <= simtrain_config.correction_score_threshold):
                stats['corrections_offered'] += 1
                if _admissible_correction(correction, turn[0], turn[1],
                                          turn[2]):
                    stats['corrections_admitted'] += 1
                    corrected_ids = tokenizer.encode(' ' + correction)
                    span_start = game['spans'][-1][0]
                    context_limit = max(
                        1, block_size + 1 - len(corrected_ids)
                    )
                    correction_buffer.append({
                        'context': game['token_ids'][:span_start]
                        [-context_limit:],
                        'turn': corrected_ids,
                    })
                    while (len(correction_buffer)
                           > simtrain_config.correction_buffer_size):
                        correction_buffer.pop(0)
            advisor_present = any(
                report['source'] == 'advisor'
                for report in game['state']['reports']
            )
            earnings, executed = market.step_game(
                game['market'], game['state'], result['actions'],
                game['world_random'], simtrain_config.market_noise_sigma,
            )
            game['earnings'].append(earnings)
            game['acted'].append(bool(executed))
            eligible = (
                bool(executed) and result['reason_given']
                and (score is None
                     or simtrain_config.imitation_score_floor <= 0
                     or score >= simtrain_config.imitation_score_floor)
            )
            game['eligible'].append(eligible)
            stats['eligible'] += int(eligible)
            if advisor_present:
                stats['advisor_earnings'].append(earnings)
            else:
                stats['no_advisor_earnings'].append(earnings)
            game['turn_records'].append({
                'quarter': quarter + 1,
                'decision': turn[0],
                'rewrite': result['rewrite'],
                'language_score': result['language_score'],
                'correction': result['correction'],
                'reason_given': result['reason_given'],
                'match': result['match'],
                'executed': executed,
                'advisor_present': advisor_present,
                'earnings': round(earnings, 2),
            })
    return games, stats, sample_turn


def _batch_tensors(games, simtrain_config, block_size, device):
    """Turn played games into padded input, target, and weight tensors.

    Only eligible turns carry loss: the turn executed an action, gave a
    reason, and cleared the imitation score floor when scored. A lucky
    but reasonless or degraded turn can act in the world but is never
    imitated. Advantages are normalized and mapped through
    max(0, exp(z) - 1), so a neutral- or negative-advantage turn
    contributes nothing at all: the loss imitates only decisions that
    scored above their baseline, never chatter and never inaction. This
    is the correction to the first pilots, where weights centered at one
    imitated every turn at full strength and self-imitation of chatter
    collapsed the model.

    With market_repeats above one, a turn's baseline is the mean earnings
    of the same quarter across the games sharing its world, so shared
    world luck cancels and the advantage measures the decision; without
    repeats the baseline is the batch mean over eligible turns.
    """
    repeats = max(1, simtrain_config.market_repeats)
    if repeats > 1:
        groups = {}
        for game in games:
            for quarter, earnings in enumerate(game['earnings']):
                groups.setdefault((game['world'], quarter),
                                  []).append(earnings)
        baselines = {
            key: statistics.mean(values) for key, values in groups.items()
        }

        def advantage_of(game, quarter):
            return (game['earnings'][quarter]
                    - baselines[(game['world'], quarter)])
    else:
        eligible_earnings = [
            earnings for game in games
            for earnings, eligible in zip(game['earnings'],
                                          game['eligible'])
            if eligible
        ]
        mean = statistics.mean(eligible_earnings)

        def advantage_of(game, quarter):
            return game['earnings'][quarter] - mean

    eligible_advantages = [
        advantage_of(game, quarter)
        for game in games
        for quarter, eligible in enumerate(game['eligible'])
        if eligible
    ]
    spread = statistics.pstdev(eligible_advantages)
    spread = spread if spread > 1e-6 else 1.0
    rows = []
    for game in games:
        token_ids = game['token_ids']
        offset = max(0, len(token_ids) - (block_size + 1))
        token_ids = token_ids[offset:]
        weights = [0.0] * (len(token_ids) - 1)
        for quarter, ((span_start, span_end), eligible) in enumerate(
                zip(game['spans'], game['eligible'])):
            if not eligible:
                continue
            normalized = advantage_of(game, quarter) / spread
            weight = math.exp(
                normalized / simtrain_config.weight_temperature
            )
            weight = max(0.0, min(weight, simtrain_config.weight_clip) - 1.0)
            if weight <= 0.0:
                continue
            for position in range(span_start - offset - 1,
                                  span_end - offset - 1):
                if 0 <= position < len(weights):
                    weights[position] = weight
        rows.append((token_ids, weights))
    return _pad_rows(rows, device)


def _pad_rows(rows, device):
    import torch

    longest = max(len(token_ids) for token_ids, _ in rows)
    inputs = torch.zeros((len(rows), longest - 1), dtype=torch.long)
    targets = torch.zeros((len(rows), longest - 1), dtype=torch.long)
    weight_tensor = torch.zeros((len(rows), longest - 1))
    for row_index, (token_ids, weights) in enumerate(rows):
        length = len(token_ids) - 1
        inputs[row_index, :length] = torch.tensor(token_ids[:-1])
        targets[row_index, :length] = torch.tensor(token_ids[1:])
        weight_tensor[row_index, :length] = torch.tensor(weights)
    return (inputs.to(device), targets.to(device), weight_tensor.to(device))


def _correction_tensors(samples, block_size, device):
    """Tensors for correction imitation: loss on corrected turns only."""
    rows = []
    for sample in samples:
        token_ids = (sample['context'] + sample['turn'])[-(block_size + 1):]
        weights = [0.0] * (len(token_ids) - 1)
        turn_start = len(token_ids) - len(sample['turn'])
        for position in range(max(0, turn_start - 1), len(weights)):
            weights[position] = 1.0
        rows.append((token_ids, weights))
    return _pad_rows(rows, device)


def _reference_returns(simtrain_config, seed, field_count,
                       companies_per_field, sample_games=200):
    blind = statistics.mean(
        market.play_game(market.blind_policy, seed + index,
                         simtrain_config.quarters, field_count,
                         companies_per_field,
                         simtrain_config.market_noise_sigma)[0]
        for index in range(sample_games)
    )
    oracle = statistics.mean(
        market.play_game(market.oracle_policy, seed + index,
                         simtrain_config.quarters, field_count,
                         companies_per_field,
                         simtrain_config.market_noise_sigma)[0]
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
    model_config = config.model
    if paths['checkpoint'] is not None and paths['checkpoint'].exists():
        checkpoint = torch.load(paths['checkpoint'], map_location=device)
        verify_checkpoint_tokenizer(
            checkpoint, paths['checkpoint'], paths['tokenizer']
        )
        vocabulary_size = checkpoint['vocabulary_size']
        model_config = checkpoint_model_config(checkpoint, config.model)
        logger.info('starting from stage-1 checkpoint %s',
                    paths['checkpoint'])
    else:
        logger.warning('no stage-1 checkpoint, starting from random '
                       'initialization (smoke-test mode)')

    gpt_config = build_config(model_config, vocabulary_size)
    model = GPT(gpt_config).to(device)
    if checkpoint is not None:
        model.load_state_dict(normalize_state_dict(checkpoint['model']))
    block_size = gpt_config.block_size
    logger.info('model: %.2fM parameters, block size %d',
                model.count_parameters() / 1e6, block_size)

    checkpoint_directory = ensure_directory(config.simtrain_dir)
    replay = None
    if simtrain_config.replay_fraction > 0:
        replay = _load_replay(paths, block_size)
    replay_random = numpy.random.default_rng(config.project.seed + 11)
    rehearsal = None
    if simtrain_config.rehearsal_fraction > 0:
        rehearsal = _load_rehearsal(simtrain_config, tokenizer, block_size)
    rehearsal_random = random.Random(config.project.seed + 17)
    correction_buffer = []
    correction_random = random.Random(config.project.seed + 13)
    language_window = []
    language_baseline = None

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

    start_step = 0
    last_checkpoint = checkpoint_directory / 'ckpt_last.pt'
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=device)
        verify_checkpoint_tokenizer(
            saved, last_checkpoint, paths['tokenizer']
        )
        base_model.load_state_dict(normalize_state_dict(saved['model']))
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step'] + 1
        language_baseline = saved.get('language_baseline')
        logger.info('resumed from step %d', start_step)

    reference_cache = {}

    def references_for(field_count, companies_per_field):
        key = (field_count, companies_per_field)
        if key not in reference_cache:
            reference_cache[key] = _reference_returns(
                simtrain_config, config.project.seed + 999983,
                field_count, companies_per_field,
            )
            logger.info(
                'references at %d fields x %d companies: blind %+.1f, '
                'oracle %+.1f', field_count, companies_per_field,
                reference_cache[key][0], reference_cache[key][1],
            )
        return reference_cache[key]

    references_for(*_difficulty_at(simtrain_config, start_step))

    def save_checkpoint(step, tag, mean_return):
        payload = {
            'model': base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'mean_return': mean_return,
            'language_baseline': language_baseline,
            'model_config': to_dict(model_config),
            'vocabulary_size': vocabulary_size,
            'tokenizer_fingerprint': tokenizer_fingerprint(
                paths['tokenizer']
            ),
        }
        torch.save(payload, checkpoint_directory / ('%s.pt' % tag))

    history_path = checkpoint_directory / 'history.jsonl'
    if start_step == 0 and history_path.exists():
        history_path.unlink()

    recent_returns = []
    best_rolling = -float('inf')
    no_signal_streak = 0
    interval_start = time.time()
    for step in range(start_step, simtrain_config.maximum_steps):
        current_learning_rate = learning_rate_at(step, simtrain_config)
        for group in optimizer.param_groups:
            group['lr'] = current_learning_rate
        field_count, companies_per_field = _difficulty_at(
            simtrain_config, step
        )
        blind_reference, oracle_reference = references_for(
            field_count, companies_per_field
        )

        model.eval()
        games, stats, sample_turn = _play_batch(
            model, tokenizer, config, llm_listener, step, block_size,
            device, correction_buffer,
        )
        model.train()

        eligible_earnings = [
            earnings for game in games
            for earnings, eligible in zip(game['earnings'],
                                          game['eligible'])
            if eligible
        ]
        no_signal = (
            len(eligible_earnings) < 2
            or statistics.pstdev(eligible_earnings) < 1e-6
        )
        step_language_score = (
            statistics.mean(stats['language_scores'])
            if stats['language_scores'] else None
        )
        if step_language_score is not None:
            language_window.append(step_language_score)
            if len(language_window) > simtrain_config.language_score_window:
                language_window.pop(0)
        if (language_baseline is None
                and len(language_window)
                >= simtrain_config.language_score_window):
            language_baseline = statistics.mean(language_window)
            logger.info('language baseline %.2f over first %d scored steps',
                        language_baseline,
                        simtrain_config.language_score_window)
        language_active = (
            language_baseline is not None
            and statistics.mean(language_window)
            < language_baseline - simtrain_config.language_score_drop
        )
        correction_count = (
            min(len(correction_buffer),
                max(1, int(simtrain_config.games_per_batch
                           * simtrain_config.correction_fraction)))
            if language_active and correction_buffer else 0
        )
        update_started = time.time()
        game_loss = None
        replay_loss = None
        rehearsal_loss = None
        correction_loss = None
        loss = None
        weights = None
        if not no_signal or correction_count:
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                if not no_signal:
                    inputs, targets, weights = _batch_tensors(
                        games, simtrain_config, block_size, device
                    )
                if weights is not None and weights.sum() > 0:
                    logits, _ = model(inputs, targets)
                    per_token = functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1),
                        reduction='none',
                    ).view_as(weights)
                    game_loss = (
                        (per_token * weights).sum() / weights.sum()
                    )
                    loss = game_loss
                    if replay is not None:
                        replay_inputs, replay_targets = replay.get_batch(
                            max(1, int(simtrain_config.games_per_batch
                                       * simtrain_config.replay_fraction)),
                            device, replay_random,
                        )
                        _, replay_loss = model(
                            replay_inputs, replay_targets
                        )
                        loss = (
                            (1.0 - simtrain_config.replay_fraction)
                            * game_loss
                            + simtrain_config.replay_fraction * replay_loss
                        )
                    if rehearsal is not None:
                        count = max(
                            1, int(simtrain_config.games_per_batch
                                   * simtrain_config.rehearsal_fraction)
                        )
                        indices = [
                            rehearsal_random.randrange(rehearsal.length())
                            for _ in range(count)
                        ]
                        rehearsal_batch = rehearsal.collate(indices, device)
                        rehearsal_inputs, rehearsal_labels, \
                            rehearsal_weights = rehearsal_batch
                        rehearsal_targets = rehearsal_labels.clamp(min=0)
                        rehearsal_logits, _ = model(
                            rehearsal_inputs, rehearsal_targets
                        )
                        rehearsal_per_token = functional.cross_entropy(
                            rehearsal_logits.view(
                                -1, rehearsal_logits.size(-1)),
                            rehearsal_targets.view(-1), reduction='none',
                        ).view_as(rehearsal_weights)
                        rehearsal_loss = (
                            (rehearsal_per_token * rehearsal_weights).sum()
                            / rehearsal_weights.sum()
                        )
                        loss = (
                            (1.0 - simtrain_config.rehearsal_fraction)
                            * loss
                            + simtrain_config.rehearsal_fraction
                            * rehearsal_loss
                        )
                else:
                    no_signal = True
                if correction_count:
                    samples = correction_random.sample(
                        correction_buffer, correction_count
                    )
                    correction_batch = _correction_tensors(
                        samples, block_size, device
                    )
                    correction_inputs, correction_targets, \
                        correction_weights = correction_batch
                    correction_logits, _ = model(
                        correction_inputs, correction_targets
                    )
                    correction_per_token = functional.cross_entropy(
                        correction_logits.view(
                            -1, correction_logits.size(-1)),
                        correction_targets.view(-1), reduction='none',
                    ).view_as(correction_weights)
                    correction_loss = (
                        (correction_per_token * correction_weights).sum()
                        / correction_weights.sum()
                    )
                    if loss is None:
                        loss = correction_loss
                    else:
                        loss = (
                            (1.0 - simtrain_config.correction_fraction)
                            * loss
                            + simtrain_config.correction_fraction
                            * correction_loss
                        )
            if loss is not None:
                loss.backward()
                if simtrain_config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(),
                        simtrain_config.gradient_clip
                    )
                optimizer.step()
        if no_signal:
            no_signal_streak += 1
            if (simtrain_config.no_signal_abort_steps > 0
                    and no_signal_streak
                    >= simtrain_config.no_signal_abort_steps):
                raise SystemExit(
                    'aborting: %d consecutive steps without actionable '
                    'signal (no executed trades with earnings variance); '
                    'the model is not engaging the market - run the gate '
                    'stage on the base checkpoint and check the bridging '
                    'stages and the entry difficulty before rerunning'
                    % no_signal_streak
                )
        else:
            no_signal_streak = 0
        update_seconds = time.time() - update_started

        mean_return = statistics.mean(
            sum(game['earnings']) for game in games
        )
        recent_returns.append(mean_return)
        if len(recent_returns) > simtrain_config.best_window:
            recent_returns.pop(0)
        rolling = statistics.mean(recent_returns)

        row = {
            'step': step,
            'field_count': field_count,
            'companies_per_field': companies_per_field,
            'no_signal': no_signal,
            'updated': loss is not None,
            'mean_return': round(mean_return, 2),
            'rolling_return': round(rolling, 2),
            'no_reason_rate': round(stats['no_reason'] / stats['turns'], 3),
            'acted_rate': round(stats['acted'] / stats['turns'], 3),
            'eligible_rate': round(stats['eligible'] / stats['turns'], 3),
            'distinct_decision_rate': round(
                len(set(stats['decisions'])) / stats['turns'], 3),
            'match_exact_rate': round(
                stats['match_exact'] / stats['turns'], 3),
            'match_fuzzy_rate': round(
                stats['match_fuzzy'] / stats['turns'], 3),
            'corrections_offered': stats['corrections_offered'],
            'corrections_admitted': stats['corrections_admitted'],
            'generate_seconds': round(stats['generate_seconds'], 2),
            'listener_seconds': round(stats['listener_seconds'], 2),
            'update_seconds': round(update_seconds, 2),
        }
        if language_baseline is not None:
            row['language_baseline'] = round(language_baseline, 2)
        if simtrain_config.no_reason_anneal_steps > 0:
            row['no_reason_probability'] = round(
                simtrain_config.no_reason_action_probability
                * max(0.0, 1.0
                      - step / simtrain_config.no_reason_anneal_steps), 3)
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
        if rehearsal_loss is not None:
            row['rehearsal_loss'] = round(rehearsal_loss.item(), 4)
        if step_language_score is not None:
            row['language_score'] = round(step_language_score, 2)
        row['language_training'] = bool(correction_count)
        row['correction_buffer'] = len(correction_buffer)
        if correction_loss is not None:
            row['correction_loss'] = round(correction_loss.item(), 4)
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
                'step %d/%d  game %s  replay %s  rehearse %s  language '
                '%s%s%s  return %+.1f (rolling %+.1f, blind %+.1f, oracle '
                '%+.1f)  no-reason %.2f  acted %.2f  eligible %.2f  '
                'distinct %.2f  match %.2f/%.2f  %.2fs/it (generate %.1f, '
                'listener %.1f, update %.1f)',
                step, simtrain_config.maximum_steps,
                '%.3f' % game_loss.item() if game_loss is not None else '-',
                '%.3f' % replay_loss.item() if replay_loss is not None
                else '-',
                '%.3f' % rehearsal_loss.item() if rehearsal_loss is not None
                else '-',
                '%.2f' % step_language_score
                if step_language_score is not None else '-',
                ' CORRECTING (%.3f over %d)' % (
                    correction_loss.item(), correction_count,
                ) if correction_loss is not None else '',
                '  NO-SIGNAL (streak %d)' % no_signal_streak
                if no_signal else '',
                mean_return, rolling, blind_reference, oracle_reference,
                stats['no_reason'] / stats['turns'],
                stats['acted'] / stats['turns'],
                stats['eligible'] / stats['turns'],
                len(set(stats['decisions'])) / stats['turns'],
                stats['match_exact'] / stats['turns'],
                stats['match_fuzzy'] / stats['turns'],
                elapsed / max(1, simtrain_config.log_interval),
                stats['generate_seconds'], stats['listener_seconds'],
                update_seconds,
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

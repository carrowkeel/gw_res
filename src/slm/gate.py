"""The entry gate: measure whether a model is ready for the simulator.

The first sim pilots showed the cost of starting below threshold: a model
that never produces an actionable turn gives the outcome loss nothing to
amplify, and score-weighted self-imitation of chatter collapses it. The
gate plays frozen probe games (no updates, no listener LLM) and measures
the genuine actionable rate: the fraction of trader turns that state a
valid move from the raw text alone, with no reason gate and no charitable
rewriting - the charity that inflated acted rates in the failed pilot is
deliberately excluded from the measurement. A valid move is an executable
trade or a stated hold; a sell of shares the trader does not hold is not
one, though the match label still records whether it named a real
company. The gate runs as its own
stage, before simulation training or after any bridging stage, and judges
the actionable rate against simtrain.entry_threshold; the simulator
itself no longer probes at startup, so passing the gate is a decision
made outside the training run. Reports are written next to the probed
checkpoint so runs never overwrite each other.

    python -m slm.gate --config configs/sim.yaml --base-run runs/<t1-tree>
"""

import argparse
import json
import random

from . import listener as listener_module
from . import market, render
from .config import load_config
from .model import GPT, build_config
from .sftstage import (
    checkpoint_model_config, resolve_base_checkpoint, resolve_checkpoint,
    verify_checkpoint_tokenizer,
)
from .tokenizer import SyntheticTokenizer
from .utils import get_logger, normalize_state_dict

logger = get_logger('gate')

PROBE_SEED_OFFSET = 424243
SAMPLE_LIMIT = 20


def probe(model, tokenizer, config, block_size, device, games_count=None):
    """Play frozen games and measure the genuine actionable rate.

    Parsed actions still execute so later quarters carry realistic
    holdings, but nothing is trained and nothing is charitably rewritten.
    """
    from .simtrain import _difficulty_at, _generate_decisions, \
        _quarter_coverage, _reference_returns

    simtrain_config = config.simtrain
    games_count = games_count or simtrain_config.entry_probe_games
    difficulty = _difficulty_at(simtrain_config, 0)
    opening_coverage = _quarter_coverage(simtrain_config, difficulty, 0)
    games = []
    for game_index in range(games_count):
        game_random = random.Random(
            config.project.seed + PROBE_SEED_OFFSET + game_index
        )
        game_market = market.sample_market(
            game_random, difficulty['field_count'],
            difficulty['companies_per_field'],
        )
        games.append({
            'random': game_random,
            'market': game_market,
            'state': market.start_game(game_market, game_random,
                                       *opening_coverage),
            'token_ids': [tokenizer.bos_id],
            'earnings': 0.0,
        })
    turns = 0
    actionable = 0
    with_reason = 0
    holds = 0
    match_counts = {'exact': 0, 'fuzzy': 0, 'none': 0}
    samples = []
    model.eval()
    decision_format = simtrain_config.decision_format
    for quarter in range(simtrain_config.quarters):
        for game in games:
            block = render.render_quarter(
                game['state'], game['market'], game['random'],
                protocol_line=simtrain_config.protocol_line,
                exemplar_turn=simtrain_config.exemplar_turn,
                decision_format=decision_format,
            )
            prefix = ('\n' if quarter else '') + block
            game['token_ids'].extend(tokenizer.encode(prefix))
        decisions = _generate_decisions(
            model, tokenizer, games, simtrain_config, block_size, device,
        )
        for game, decision_text in zip(games, decisions):
            decision_ids = tokenizer.encode(' ' + decision_text)
            game['token_ids'].extend(decision_ids)
            actions, match = listener_module.parse_decision(
                decision_text, game['market'], game['state'],
                decision_format,
            )
            has_reason = bool(
                listener_module.reason_text(decision_text, decision_format)
            )
            held = (
                not actions
                and listener_module.hold_stated(decision_text)
            )
            turns += 1
            actionable += int(bool(actions) or held)
            with_reason += int(has_reason)
            holds += int(held)
            match_counts[match] += 1
            if len(samples) < SAMPLE_LIMIT:
                samples.append({
                    'quarter': quarter + 1,
                    'decision': decision_text,
                    'actions': actions,
                    'hold': held,
                    'match': match,
                    'reason_given': has_reason,
                })
            next_coverage = _quarter_coverage(
                simtrain_config, difficulty,
                min(quarter + 1, simtrain_config.quarters - 1),
            )
            earnings, _ = market.step_game(
                game['market'], game['state'], actions, game['random'],
                difficulty['market_noise_sigma'], *next_coverage,
            )
            game['earnings'] += earnings
    blind_reference, oracle_reference = _reference_returns(
        simtrain_config, config.project.seed + PROBE_SEED_OFFSET,
        difficulty,
    )
    mean_return = (
        sum(game['earnings'] for game in games) / games_count
        if games_count else 0.0
    )
    return {
        'games': games_count,
        'decision_format': decision_format,
        'field_count': difficulty['field_count'],
        'companies_per_field': difficulty['companies_per_field'],
        'report_coverage': difficulty['report_coverage'],
        'advisor_coverage': difficulty['advisor_coverage'],
        'mean_return': round(mean_return, 2),
        'blind_reference': round(blind_reference, 2),
        'oracle_reference': round(oracle_reference, 2),
        'turns': turns,
        'actionable_rate': round(actionable / turns, 4) if turns else 0.0,
        'hold_rate': round(holds / turns, 4) if turns else 0.0,
        'reason_rate': round(with_reason / turns, 4) if turns else 0.0,
        'match_exact_rate': (
            round(match_counts['exact'] / turns, 4) if turns else 0.0
        ),
        'match_fuzzy_rate': (
            round(match_counts['fuzzy'] / turns, 4) if turns else 0.0
        ),
        'entry_threshold': simtrain_config.entry_threshold,
        'passes': bool(
            turns and actionable / turns >= simtrain_config.entry_threshold
        ),
        'samples': samples,
    }


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        'gate: actionable %.3f (threshold %.2f, %s), hold %.3f, reason '
        '%.3f, match %.3f/%.3f, return %+.1f (blind %+.1f, oracle %+.1f) '
        'at %dx%d over %d turns -> %s',
        report['actionable_rate'], report['entry_threshold'],
        'passes' if report['passes'] else 'BELOW THRESHOLD',
        report['hold_rate'], report['reason_rate'],
        report['match_exact_rate'],
        report['match_fuzzy_rate'], report['mean_return'],
        report['blind_reference'], report['oracle_reference'],
        report['field_count'], report['companies_per_field'],
        report['turns'], path,
    )


def run(config, checkpoint_path=None):
    import torch

    from pathlib import Path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = config.simtrain.base_run_dir
    if checkpoint_path is None:
        if not base:
            raise SystemExit(
                'gate needs a base run (--base-run) or an explicit '
                '--checkpoint'
            )
        if config.simtrain.base_stage:
            checkpoint_path = resolve_checkpoint(
                Path(base) / 'checkpoints' / config.simtrain.base_stage
            )
            if checkpoint_path is None:
                raise SystemExit(
                    'base_stage %s has no checkpoint under %s/checkpoints'
                    % (config.simtrain.base_stage, base)
                )
        else:
            checkpoint_path = resolve_base_checkpoint(base)
        if checkpoint_path is None:
            raise SystemExit('no checkpoint found under %s' % base)

    tokenizer_path = (
        Path(base) / 'tokenizer' / 'tokenizer.json' if base
        else config.tokenizer_path
    )
    tokenizer = SyntheticTokenizer(tokenizer_path)
    saved = torch.load(checkpoint_path, map_location=device)
    verify_checkpoint_tokenizer(saved, checkpoint_path, tokenizer_path)
    gpt_config = build_config(
        checkpoint_model_config(saved, config.model),
        saved['vocabulary_size'],
    )
    model = GPT(gpt_config).to(device)
    model.load_state_dict(normalize_state_dict(saved['model']))
    logger.info('probing %s', checkpoint_path)
    report = probe(
        model, tokenizer, config, gpt_config.block_size, device
    )
    report = {'checkpoint': str(checkpoint_path), **report}
    write_report(
        report, Path(checkpoint_path).parent / 'gate_report.json'
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Measure a checkpoint against the simulator entry '
                    'threshold'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--base-run', default=None)
    parser.add_argument('--checkpoint', default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config, run_id=arguments.run_id)
    if arguments.base_run:
        config.simtrain.base_run_dir = arguments.base_run
    from pathlib import Path

    checkpoint_path = (
        Path(arguments.checkpoint) if arguments.checkpoint else None
    )
    run(config, checkpoint_path)


if __name__ == '__main__':
    main()

"""The entry gate: measure whether a model is ready for the simulator.

The first sim pilots showed the cost of starting below threshold: a model
that never produces an actionable turn gives the outcome loss nothing to
amplify, and score-weighted self-imitation of chatter collapses it. The
gate plays frozen probe games (no updates, no listener LLM) and measures
the genuine actionable rate: the fraction of trader turns the pattern
parser can act on from the raw text alone, with no reason gate and no
charitable rewriting — the charity that inflated acted rates in the failed
pilot is deliberately excluded from the measurement. The simulator refuses
to start below simtrain.entry_threshold; the module also runs standalone
after any stage to measure where a checkpoint stands.

    python -m slm.gate --config configs/sim.yaml --base-run runs/<t1-tree>
"""

import argparse
import json
import random

from . import listener as listener_module
from . import market, render
from .config import load_config
from .model import GPT, build_config
from .sftstage import resolve_base_checkpoint
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, normalize_state_dict

logger = get_logger('gate')

PROBE_SEED_OFFSET = 424243
SAMPLE_LIMIT = 20


def probe(model, tokenizer, config, block_size, device, games_count=None):
    """Play frozen games and measure the genuine actionable rate.

    Parsed actions still execute so later quarters carry realistic
    holdings, but nothing is trained and nothing is charitably rewritten.
    """
    from .simtrain import _generate_decisions

    simtrain_config = config.simtrain
    games_count = games_count or simtrain_config.entry_probe_games
    games = []
    for game_index in range(games_count):
        game_random = random.Random(
            config.project.seed + PROBE_SEED_OFFSET + game_index
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
        })
    turns = 0
    actionable = 0
    with_reason = 0
    match_counts = {'exact': 0, 'fuzzy': 0, 'none': 0}
    samples = []
    model.eval()
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
        for game, decision_text in zip(games, decisions):
            decision_ids = tokenizer.encode(' ' + decision_text)
            game['token_ids'].extend(decision_ids)
            actions, match = listener_module.parse_orders(
                decision_text, game['market'], game['state']
            )
            has_reason = listener_module.reason_given(decision_text)
            turns += 1
            actionable += int(bool(actions))
            with_reason += int(has_reason)
            match_counts[match] += 1
            if len(samples) < SAMPLE_LIMIT:
                samples.append({
                    'quarter': quarter + 1,
                    'decision': decision_text,
                    'actions': actions,
                    'match': match,
                    'reason_given': has_reason,
                })
            market.step_game(
                game['market'], game['state'], actions, game['random']
            )
    return {
        'games': games_count,
        'turns': turns,
        'actionable_rate': round(actionable / turns, 4) if turns else 0.0,
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
        'gate: actionable %.3f (threshold %.2f, %s), reason %.3f, match '
        '%.3f/%.3f over %d turns -> %s',
        report['actionable_rate'], report['entry_threshold'],
        'passes' if report['passes'] else 'BELOW THRESHOLD',
        report['reason_rate'], report['match_exact_rate'],
        report['match_fuzzy_rate'], report['turns'], path,
    )


def run(config, checkpoint_path=None):
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = config.simtrain.base_run_dir
    if checkpoint_path is None:
        if not base:
            raise SystemExit(
                'gate needs a base run (--base-run) or an explicit '
                '--checkpoint'
            )
        checkpoint_path = resolve_base_checkpoint(base)
        if checkpoint_path is None:
            raise SystemExit('no checkpoint found under %s' % base)
    from pathlib import Path

    tokenizer_path = (
        Path(base) / 'tokenizer' / 'tokenizer.json' if base
        else config.tokenizer_path
    )
    tokenizer = SyntheticTokenizer(tokenizer_path)
    saved = torch.load(checkpoint_path, map_location=device)
    gpt_config = build_config(config.model, saved['vocabulary_size'])
    model = GPT(gpt_config).to(device)
    model.load_state_dict(normalize_state_dict(saved['model']))
    logger.info('probing %s', checkpoint_path)
    report = probe(
        model, tokenizer, config, gpt_config.block_size, device
    )
    report = {'checkpoint': str(checkpoint_path), **report}
    write_report(
        report, ensure_directory(config.simtrain_dir) / 'gate_report.json'
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

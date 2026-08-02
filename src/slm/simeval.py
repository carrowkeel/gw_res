"""Grade a simulation-trained checkpoint on a fixed battery of games.

The training history's rolling return is an on-policy metric: it mixes
the learned policy with exploration sampling and with whichever worlds
the last steps happened to draw, which leaves run-to-run noise wide
enough to swallow real differences between variants. This stage answers
the question the sweeps actually ask - which checkpoint plays better -
by making every variant sit the same exam: a battery of games seeded
independently of the run (battery_seed, shared by every tree), played
with greedy decoding at a fixed difficulty. Market randomness is
action-independent by construction, so two variants evaluated on the
same battery face byte-identical shock and report streams, and their
per-game return differences are paired: world luck cancels exactly, and
the comparison report can put a standard error on the difference
instead of eyeballing rolling returns.

Per-game returns are saved alongside the summary rates in eval.json
next to the checkpoint, so any two runs can be compared paired after
the fact. Blind and oracle references are replayed on the same battery
seeds. Everything is program-checked; no LLM judges anything.

    python -m slm.simeval --config runs/sweeps/<sweep>/<variant>/config.yaml
"""

import argparse
import json
import random
import statistics
import types
from pathlib import Path

from . import market, render
from . import listener as listener_module
from .config import load_config
from .sftstage import load_checkpoint_model, resolve_checkpoint
from .simtrain import _base_paths, _generate_decisions
from .tokenizer import SyntheticTokenizer
from .utils import get_logger, set_seed

logger = get_logger('simeval')


def _battery_difficulty(simeval_config):
    return {
        'field_count': simeval_config.field_count,
        'companies_per_field': simeval_config.companies_per_field,
        'report_coverage': simeval_config.report_coverage,
        'advisor_coverage': simeval_config.advisor_coverage,
        'advisor_accuracy': simeval_config.advisor_accuracy,
        'market_noise_sigma': simeval_config.market_noise_sigma,
        'numeric_reports': simeval_config.numeric_reports,
    }


def _reference_totals(simeval_config, policy):
    return [
        market.play_game(
            policy, simeval_config.battery_seed + index,
            simeval_config.quarters, simeval_config.field_count,
            simeval_config.companies_per_field,
            simeval_config.market_noise_sigma,
            simeval_config.report_coverage,
            simeval_config.advisor_coverage,
            advisor_accuracy=simeval_config.advisor_accuracy,
        )[0]
        for index in range(simeval_config.games)
    ]


def _mean_and_error(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return round(mean, 2), None
    error = statistics.stdev(values) / (len(values) ** 0.5)
    return round(mean, 2), round(error, 2)


def evaluate(config, checkpoint_path=None):
    """Play the battery with a checkpoint; return and write the report.

    The game loop mirrors market.play_game's randomness consumption
    (one generator per game: sample_market, start_game, then step_game
    per quarter, with nothing else drawing from it), so the battery's
    worlds are identical for every checkpoint ever graded against the
    same battery settings.
    """
    import torch

    simeval_config = config.simeval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(0)

    checkpoint_dir = config.out_dir / 'checkpoints' / 'simtrain'
    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            raise SystemExit('no simtrain checkpoint under %s'
                             % checkpoint_dir)
    tokenizer = SyntheticTokenizer(_base_paths(config)['tokenizer'])
    model, gpt_config = load_checkpoint_model(config, checkpoint_path,
                                              device)
    model.eval()
    generation = types.SimpleNamespace(
        max_decision_tokens=simeval_config.max_decision_tokens,
        sample_temperature=simeval_config.temperature,
        sample_top_p=simeval_config.top_p,
    )

    games = []
    for index in range(simeval_config.games):
        game_random = random.Random(simeval_config.battery_seed + index)
        game_market = market.sample_market(
            game_random, simeval_config.field_count,
            simeval_config.companies_per_field,
        )
        games.append({
            'random': game_random,
            'render_random': random.Random(
                simeval_config.battery_seed + 1000003 + index
            ),
            'market': game_market,
            'state': market.start_game(
                game_market, game_random,
                simeval_config.report_coverage,
                simeval_config.advisor_coverage,
                simeval_config.advisor_accuracy,
            ),
            'token_ids': [tokenizer.bos_id],
            'earnings': 0.0,
        })
    counts = {'turns': 0, 'template': 0, 'actionable': 0, 'acted': 0,
              'match_exact': 0, 'grounded': 0, 'truthful': 0}
    with torch.no_grad():
        for quarter in range(simeval_config.quarters):
            for game in games:
                block = render.render_quarter(
                    game['state'], game['market'], game['render_random'],
                    protocol_line=False, exemplar_turn=False,
                    decision_format='structured',
                    input_variety=simeval_config.input_variety,
                    numeric_reports=simeval_config.numeric_reports,
                )
                prefix = ('\n' if quarter else '') + block
                game['token_ids'].extend(tokenizer.encode(prefix))
            decisions = _generate_decisions(
                model, tokenizer, games, generation,
                gpt_config.block_size, device,
            )
            for game, decision in zip(games, decisions):
                game['token_ids'].extend(tokenizer.encode(' ' + decision))
                actions, match = listener_module.parse_decision(
                    decision, game['market'], game['state'], 'structured',
                )
                hold = not actions and listener_module.hold_stated(decision)
                reason = listener_module.reason_text(decision, 'structured')
                counts['turns'] += 1
                counts['template'] += int(
                    listener_module.structured_move(decision))
                counts['actionable'] += int(bool(actions) or hold)
                counts['match_exact'] += int(match == 'exact')
                counts['grounded'] += int(listener_module.grounded_reason(
                    reason, game['market']))
                counts['truthful'] += int(listener_module.truthful_reason(
                    reason, game['market'],
                    game['state']['leaked_shocks']))
                earnings, executed = market.step_game(
                    game['market'], game['state'], actions, game['random'],
                    simeval_config.market_noise_sigma,
                    simeval_config.report_coverage,
                    simeval_config.advisor_coverage,
                    simeval_config.advisor_accuracy,
                )
                counts['acted'] += int(bool(executed))
                game['earnings'] += earnings

    returns = [round(game['earnings'], 2) for game in games]
    mean_return, standard_error = _mean_and_error(returns)
    blind_totals = _reference_totals(simeval_config, market.blind_policy)
    oracle_totals = _reference_totals(simeval_config, market.oracle_policy)
    blind_mean = round(statistics.mean(blind_totals), 2)
    oracle_mean = round(statistics.mean(oracle_totals), 2)
    turns = counts['turns']

    def rate(key):
        return round(counts[key] / turns, 4) if turns else 0.0

    report = {
        'checkpoint': str(checkpoint_path),
        'battery_seed': simeval_config.battery_seed,
        'games': simeval_config.games,
        'quarters': simeval_config.quarters,
        'field_count': simeval_config.field_count,
        'companies_per_field': simeval_config.companies_per_field,
        'report_coverage': simeval_config.report_coverage,
        'advisor_coverage': simeval_config.advisor_coverage,
        'advisor_accuracy': simeval_config.advisor_accuracy,
        'market_noise_sigma': simeval_config.market_noise_sigma,
        'temperature': simeval_config.temperature,
        'mean_return': mean_return,
        'standard_error': standard_error,
        'blind_reference': blind_mean,
        'oracle_reference': oracle_mean,
        'headroom': (
            round((mean_return - blind_mean) / (oracle_mean - blind_mean), 3)
            if oracle_mean > blind_mean else None
        ),
        'template_rate': rate('template'),
        'actionable_rate': rate('actionable'),
        'acted_rate': rate('acted'),
        'match_exact_rate': rate('match_exact'),
        'grounded_rate': rate('grounded'),
        'truthful_reason_rate': rate('truthful'),
        'returns': returns,
    }
    report_path = Path(checkpoint_path).parent / 'eval.json'
    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        'battery: return %+.2f (se %s) over %d games at %dx%d, blind '
        '%+.1f, oracle %+.1f, headroom %s, exact %.2f, truthful %.2f '
        '-> %s',
        mean_return, standard_error, simeval_config.games,
        simeval_config.field_count, simeval_config.companies_per_field,
        blind_mean, oracle_mean, report['headroom'],
        report['match_exact_rate'], report['truthful_reason_rate'],
        report_path,
    )
    return report


def run(config):
    return evaluate(config)


def main():
    parser = argparse.ArgumentParser(
        description='Grade a sim checkpoint on the fixed game battery'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument(
        '--checkpoint', default=None,
        help='checkpoint to grade; defaults to the run tree\'s best',
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, run_id=arguments.run_id)
    checkpoint_path = (
        Path(arguments.checkpoint) if arguments.checkpoint else None
    )
    evaluate(config, checkpoint_path)


if __name__ == '__main__':
    main()

"""Template SFT: teach the structured sim register before the simulator.

The simulation's clean-register form: inputs and outputs share one
strict template ('state: ... | ...' lines in, 'move: ... | reason: ...'
out), deliberately disjoint from the dialogue register, and the model
learns it here by supervised imitation instead of in-context
instruction. Every record is generated programmatically - no LLM
anywhere: games are played by a teacher that reads the leaked reports
the way an ideal player would, each quarter becomes a (context so far,
decision line) pair, and every decision line is verified to round-trip
through the structured parser before it is written. Reasons cite the
report vocabulary (a demand factor or a material), so the taught turns
are grounded by construction and pass the sim's reason_grounding floor.

The teacher does inject a policy - follow the strongest leaked signal -
which is intentional: the stage teaches the register and the habit of
reading reports; the simulator's outcome loss then owns refining the
policy. Stage-1 replay protects general language while the register is
added; prose is register-distant, so it cannot compete with the
template. Bridge rehearsal is off by default and exists only as an
option: the bridge's freeform decision turns are the one register the
template is meant to supersede, and rehearsing them trains the closest
competing behavior in decision-shaped contexts.

The stage trains from the run tree's furthest bridging checkpoint and
writes checkpoints/templatesft, which resolve_base_checkpoint prefers,
so a gate or sim run pointed at the tree builds on the taught register
automatically.

    python -m slm.templatesft --config configs/t1_full.yaml --run-id <id>
"""

import argparse
import json
import random
from pathlib import Path

from . import listener, market, render, sftstage
from .config import load_config
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('templatesft')

HOLD_REASONS = [
    'the reports show no clear signal',
    'no report moves any company enough',
    'the leaked reports are too weak to act on',
]


def _reason_for(company, leaked, direction):
    """A grounded reason for trading this company in this direction.

    Prefers the demand factor when its leaked shock supports the
    direction (it carries the larger weight), falling back to the cost
    factor. The wording uses the same vocabulary the news lines use, so
    taught reasons stay inside the register.
    """
    demand = leaked.get(company['demand_factor'])
    cost = leaked.get(company['cost_factor'])
    if direction == 'buy':
        if demand == 1:
            return '%s will be strong next quarter' % company['demand_factor']
        if cost == -1:
            return 'the %s will fall next quarter' % company['cost_factor']
    else:
        if demand == -1:
            return '%s will be weak next quarter' % company['demand_factor']
        if cost == 1:
            return 'the %s will rise next quarter' % company['cost_factor']
    return None


def _teacher_turn(game_market, state, random_generator, stage_config):
    """One teacher decision: (decision line, action dict or None).

    Sell the worst held company when its expected return is at or below
    minus the hold threshold, otherwise buy the best company when its
    expected return reaches the threshold and cash allows, otherwise
    hold. Quantities are randomized - they carry no information and the
    sim's clamps make them consequence-light - so the register never
    learns a magic number.
    """
    leaked = state['leaked_shocks']
    worst_value = 0.0
    worst = None
    for company in game_market['companies']:
        if state['holdings'][company['name']] <= 0:
            continue
        value = market.expected_return(company, leaked)
        if value < worst_value:
            worst_value = value
            worst = company
    if worst is not None and worst_value <= -stage_config.hold_threshold:
        reason = _reason_for(worst, leaked, 'sell')
        if reason is not None:
            held = state['holdings'][worst['name']]
            if held == 1 or random_generator.random() < 0.5:
                quantity_word = 'all'
                quantity = held
            else:
                quantity = random_generator.randint(1, held)
                quantity_word = str(quantity)
            line = 'move: sell %s %s | reason: %s' % (
                quantity_word, worst['name'], reason
            )
            action = {'action': 'sell', 'company': worst['name'],
                      'quantity': quantity}
            return line, action
    best_value = 0.0
    best = None
    for company in game_market['companies']:
        value = market.expected_return(company, leaked)
        if value > best_value:
            best_value = value
            best = company
    if best is not None and best_value >= stage_config.hold_threshold:
        affordable = int(state['cash'] // state['prices'][best['name']])
        reason = _reason_for(best, leaked, 'buy')
        if affordable > 0 and reason is not None:
            quantity = random_generator.randint(
                1, min(affordable, stage_config.maximum_quantity)
            )
            line = 'move: buy %d %s | reason: %s' % (
                quantity, best['name'], reason
            )
            action = {'action': 'buy', 'company': best['name'],
                      'quantity': quantity}
            return line, action
    line = 'move: hold | reason: %s' % random_generator.choice(HOLD_REASONS)
    return line, None


def _crop_lines(lines, tokenizer, budget):
    """Drop the oldest quarters until the prompt fits the token budget.

    Quarters start at their state line, so cropping removes whole
    quarters from the front and the prompt always opens on a state line,
    the same shape the sim's block-size crop produces.
    """
    kept = list(lines)
    while len(tokenizer.encode('\n'.join(kept))) > budget:
        cut = None
        for index in range(1, len(kept)):
            if kept[index].startswith('state:'):
                cut = index
                break
        if cut is None:
            break
        kept = kept[cut:]
    return kept


def generate_records(config, tokenizer):
    """Play teacher games and emit (prompt, response) records.

    One record per quarter: the structured game context up to the trader
    cue, and the teacher's decision line. Field count varies per game
    across the sim curriculum's range so the register generalizes across
    difficulties. Every line is verified against the structured parser:
    a record this stage cannot parse must never be taught.
    """
    stage_config = config.templatesft
    random_generator = random.Random(config.project.seed + 31)
    records = []
    action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
    for game_index in range(stage_config.number_of_games):
        field_count = random_generator.randint(1, stage_config.field_count)
        game_random = random.Random(
            config.project.seed + 7000000 + game_index
        )
        game_market = market.sample_market(
            game_random, field_count, stage_config.companies_per_field
        )
        state = market.start_game(
            game_market, game_random, stage_config.report_coverage,
            stage_config.advisor_coverage,
        )
        lines = []
        for _ in range(stage_config.quarters):
            block = render.render_structured_quarter(state, game_market)
            lines.extend(block.split('\n'))
            decision_line, action = _teacher_turn(
                game_market, state, random_generator, stage_config
            )
            parsed, match = listener.parse_structured(
                decision_line, game_market, state
            )
            if action is None:
                assert not parsed and listener.hold_stated(decision_line), \
                    decision_line
                action_counts['hold'] += 1
            else:
                assert parsed and parsed[0] == action and match == 'exact', \
                    (decision_line, parsed, action)
                action_counts[action['action']] += 1
            budget = (
                stage_config.maximum_sequence_length
                - len(tokenizer.encode(' ' + decision_line)) - 2
            )
            prompt_lines = _crop_lines(lines, tokenizer, budget)
            records.append({
                'prompt': '\n'.join(prompt_lines),
                'response': decision_line,
            })
            lines[-1] = '%s %s' % (render.STRUCTURED_TRADER_CUE,
                                   decision_line)
            market.step_game(
                game_market, state,
                [action] if action is not None else [], game_random,
                market.NOISE_SIGMA, stage_config.report_coverage,
                stage_config.advisor_coverage,
            )
    logger.info(
        'generated %d records from %d games (buy %d, sell %d, hold %d)',
        len(records), stage_config.number_of_games,
        action_counts['buy'], action_counts['sell'], action_counts['hold'],
    )
    return records


def _truthful_reason(reason, game_market, leaked):
    """Whether a reason's claim matches the actually leaked shock.

    The taught reasons make one falsifiable claim in report vocabulary
    ('rain will be strong', 'the plastic price will fall'), so the claim
    can be checked against the leaked shocks: the cited factor must have
    leaked and the direction word must match its level. A reason citing
    nothing checkable, or citing a factor that never leaked, is not
    truthful - taxidermy reasons fail here even when they ground.
    """
    lowered = reason.lower()
    for factor in game_market['demand_factors']:
        if factor in lowered:
            if 'strong' in lowered:
                return leaked.get(factor) == 1
            if 'weak' in lowered:
                return leaked.get(factor) == -1
            return False
    for cost_factor in game_market['cost_factors']:
        if cost_factor.split()[0] in lowered:
            if 'rise' in lowered:
                return leaked.get(cost_factor) == 1
            if 'fall' in lowered:
                return leaked.get(cost_factor) == -1
            return False
    return False


EVAL_SEED_OFFSET = 900001
SAMPLE_LIMIT = 12


def evaluate(config, checkpoint_path=None):
    """Play held-out games generatively and grade the taught register.

    The gate answers 'does it state valid moves'; this answers 'did the
    register take, and does it interact'. Everything is program-checked:
    template_rate (the strict form, either template), match_exact_rate
    (full company names), grounded and truthful reason rates (the cited
    factor really leaked with the claimed direction), and agreement with
    the teacher's signal-following on the same states - the interaction
    test that decides whether a sweep on this checkpoint can learn
    anything. Games run at the simulator's entry difficulty with the
    simulator's own sampling settings, because that is what the sweep
    will face. The report lands next to the checkpoint; passes is judged
    on template_rate against eval_template_threshold, since a model that
    cannot hold the form gives the sim nothing to score.
    """
    import torch

    from .simtrain import (
        _difficulty_at, _generate_decisions, _quarter_coverage,
        _reference_returns,
    )

    stage_config = config.templatesft
    simtrain_config = config.simtrain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if checkpoint_path is None:
        checkpoint_path = sftstage.resolve_checkpoint(config.templatesft_dir)
        if checkpoint_path is None:
            raise SystemExit('no templatesft checkpoint under %s'
                             % config.templatesft_dir)
    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    model, gpt_config = sftstage.load_checkpoint_model(
        config, checkpoint_path, device,
        tokenizer_path=config.tokenizer_path,
    )
    model.eval()

    difficulty = _difficulty_at(simtrain_config, 0)
    opening_coverage = _quarter_coverage(simtrain_config, difficulty, 0)
    teacher_random = random.Random(config.project.seed + 41)
    games = []
    for game_index in range(stage_config.eval_games):
        game_random = random.Random(
            config.project.seed + EVAL_SEED_OFFSET + game_index
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
    counts = {'turns': 0, 'template': 0, 'actionable': 0, 'hold': 0,
              'match_exact': 0, 'reason': 0, 'grounded': 0, 'truthful': 0,
              'trade_quarters': 0, 'trade_agreements': 0,
              'hold_quarters': 0, 'hold_agreements': 0}
    samples = []
    for quarter in range(simtrain_config.quarters):
        for game in games:
            block = render.render_structured_quarter(
                game['state'], game['market']
            )
            prefix = ('\n' if quarter else '') + block
            game['token_ids'].extend(tokenizer.encode(prefix))
        decisions = _generate_decisions(
            model, tokenizer, games, simtrain_config,
            gpt_config.block_size, device,
        )
        for game, decision in zip(games, decisions):
            game['token_ids'].extend(tokenizer.encode(' ' + decision))
            parsed, match = listener.parse_structured(
                decision, game['market'], game['state']
            )
            hold = not parsed and listener.hold_stated(decision)
            reason = listener.reason_text(decision, 'structured')
            grounded = listener.grounded_reason(reason, game['market'])
            truthful = _truthful_reason(
                reason, game['market'], game['state']['leaked_shocks']
            )
            teacher_line, teacher_action = _teacher_turn(
                game['market'], game['state'], teacher_random, stage_config
            )
            counts['turns'] += 1
            counts['template'] += int(listener.structured_move(decision))
            counts['actionable'] += int(bool(parsed) or hold)
            counts['hold'] += int(hold)
            counts['match_exact'] += int(match == 'exact')
            counts['reason'] += int(bool(reason))
            counts['grounded'] += int(grounded)
            counts['truthful'] += int(truthful)
            if teacher_action is None:
                counts['hold_quarters'] += 1
                counts['hold_agreements'] += int(hold)
            else:
                counts['trade_quarters'] += 1
                agrees = bool(parsed) and (
                    parsed[0]['action'] == teacher_action['action']
                    and parsed[0]['company'] == teacher_action['company']
                )
                counts['trade_agreements'] += int(agrees)
            if len(samples) < SAMPLE_LIMIT:
                samples.append({
                    'quarter': quarter + 1,
                    'decision': decision,
                    'teacher': teacher_line,
                    'match': match,
                    'truthful_reason': truthful,
                })
            next_coverage = _quarter_coverage(
                simtrain_config, difficulty,
                min(quarter + 1, simtrain_config.quarters - 1),
            )
            earnings, _ = market.step_game(
                game['market'], game['state'], parsed, game['random'],
                difficulty['market_noise_sigma'], *next_coverage,
            )
            game['earnings'] += earnings
    blind_reference, oracle_reference = _reference_returns(
        simtrain_config, config.project.seed + EVAL_SEED_OFFSET,
        difficulty,
    )
    turns = counts['turns']

    def rate(key):
        return round(counts[key] / turns, 4) if turns else 0.0

    report = {
        'checkpoint': str(checkpoint_path),
        'games': stage_config.eval_games,
        'field_count': difficulty['field_count'],
        'companies_per_field': difficulty['companies_per_field'],
        'report_coverage': difficulty['report_coverage'],
        'advisor_coverage': difficulty['advisor_coverage'],
        'turns': turns,
        'template_rate': rate('template'),
        'actionable_rate': rate('actionable'),
        'hold_rate': rate('hold'),
        'match_exact_rate': rate('match_exact'),
        'reason_rate': rate('reason'),
        'grounded_rate': rate('grounded'),
        'truthful_reason_rate': rate('truthful'),
        'trade_agreement_rate': (
            round(counts['trade_agreements'] / counts['trade_quarters'], 4)
            if counts['trade_quarters'] else None
        ),
        'hold_agreement_rate': (
            round(counts['hold_agreements'] / counts['hold_quarters'], 4)
            if counts['hold_quarters'] else None
        ),
        'mean_return': round(
            sum(game['earnings'] for game in games)
            / max(1, stage_config.eval_games), 2
        ),
        'blind_reference': round(blind_reference, 2),
        'oracle_reference': round(oracle_reference, 2),
        'template_threshold': stage_config.eval_template_threshold,
        'passes': bool(
            turns and counts['template'] / turns
            >= stage_config.eval_template_threshold
        ),
        'samples': samples,
    }
    report_path = Path(checkpoint_path).parent / 'template_eval.json'
    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        'template eval: template %.3f (threshold %.2f, %s), actionable '
        '%.3f, exact %.3f, grounded %.3f, truthful %.3f, agreement '
        'trade %s / hold %s, return %+.1f (blind %+.1f, oracle %+.1f) '
        'over %d turns -> %s',
        report['template_rate'], report['template_threshold'],
        'passes' if report['passes'] else 'BELOW THRESHOLD',
        report['actionable_rate'], report['match_exact_rate'],
        report['grounded_rate'], report['truthful_reason_rate'],
        report['trade_agreement_rate'], report['hold_agreement_rate'],
        report['mean_return'], report['blind_reference'],
        report['oracle_reference'], turns, report_path,
    )
    return report


def _write_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')


def _load_bridge_rehearsal(config, limit):
    records_path = config.bridgesft_data_dir / 'train.jsonl'
    if not records_path.exists():
        logger.warning('no bridge records at %s, rehearsal disabled',
                       records_path)
        return None
    records = []
    with open(records_path) as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def _source_checkpoint(config):
    """The furthest prior-stage checkpoint, never this stage's own."""
    for stage in sftstage.STAGE_ORDER:
        if stage == 'templatesft':
            continue
        found = sftstage.resolve_checkpoint(
            config.out_dir / 'checkpoints' / stage
        )
        if found is not None:
            return found
    return None


def run(config):
    stage_config = config.templatesft
    set_seed(config.project.seed)
    tokenizer = SyntheticTokenizer(config.tokenizer_path)

    source_checkpoint = _source_checkpoint(config)
    if source_checkpoint is None:
        raise SystemExit(
            'templatesft needs a prior checkpoint in %s; run the bridging '
            'stages first' % (config.out_dir / 'checkpoints')
        )

    records = generate_records(config, tokenizer)
    split_random = random.Random(config.project.seed + 37)
    split_random.shuffle(records)
    holdout = max(1, int(len(records) * stage_config.holdout_fraction))
    validation_records = records[:holdout]
    train_records = records[holdout:]
    _write_records(train_records,
                   config.templatesft_data_dir / 'train.jsonl')
    _write_records(validation_records,
                   config.templatesft_data_dir / 'val.jsonl')

    rehearsal_records = None
    if stage_config.rehearsal_fraction > 0:
        rehearsal_records = _load_bridge_rehearsal(
            config, stage_config.rehearsal_records
        )

    checkpoint_directory = ensure_directory(config.templatesft_dir)
    best = sftstage.train_stage(
        config, stage_config, tokenizer, train_records, validation_records,
        source_checkpoint, checkpoint_directory, 'templatesft',
        rehearsal_records=rehearsal_records,
        rehearsal_fraction=stage_config.rehearsal_fraction,
    )
    evaluate(config, best)
    return best


def main():
    parser = argparse.ArgumentParser(
        description='Teach the structured sim register by supervised '
                    'imitation'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument(
        '--eval-only', action='store_true',
        help='skip training and grade an existing checkpoint',
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='checkpoint to grade with --eval-only; defaults to the '
             'stage\'s best',
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, run_id=arguments.run_id)
    if arguments.eval_only:
        checkpoint_path = (
            Path(arguments.checkpoint) if arguments.checkpoint else None
        )
        evaluate(config, checkpoint_path)
        return
    run(config)


if __name__ == '__main__':
    main()

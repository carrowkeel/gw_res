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
policy. Stage-1 replay and a rehearsal fraction of bridge records
protect the earlier registers while this one is added.

The stage trains from the run tree's furthest bridging checkpoint and
writes checkpoints/templatesft, which resolve_base_checkpoint prefers,
so a gate or sim run pointed at the tree builds on the taught register
automatically.

    python -m slm.templatesft --config configs/t1_full.yaml --run-id <id>
"""

import argparse
import json
import random

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
        state = market.start_game(game_market, game_random)
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
            )
    logger.info(
        'generated %d records from %d games (buy %d, sell %d, hold %d)',
        len(records), stage_config.number_of_games,
        action_counts['buy'], action_counts['sell'], action_counts['hold'],
    )
    return records


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
    return sftstage.train_stage(
        config, stage_config, tokenizer, train_records, validation_records,
        source_checkpoint, checkpoint_directory, 'templatesft',
        rehearsal_records=rehearsal_records,
        rehearsal_fraction=stage_config.rehearsal_fraction,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Teach the structured sim register by supervised '
                    'imitation'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=None)
    arguments = parser.parse_args()
    run(load_config(arguments.config, run_id=arguments.run_id))


if __name__ == '__main__':
    main()

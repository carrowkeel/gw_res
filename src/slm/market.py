"""Economic market simulator for SGM stage-2 training.

The world is deliberately information-pure: demand and cost factors are
white noise, so share prices are random walks and following prices alone
has zero expected return. The only edge is in the reports. Each quarter
the program pre-samples the next quarter's factor shocks and leaks a
partial view of them as structured report items (news about factors,
advisor recommendations about companies); the player rebalances a
portfolio, the quarter resolves, and the per-step score is the change in
portfolio value. Companies compete within fields: every company in a
field shares that field's demand factor (umbrella makers gain when rain
is coming) but each uses a different material whose cost factor it
alone carries (the plastic umbrella maker does better when plastic gets
cheaper), so the best pick in a field requires composing two reports.

This module holds structure only: states, reports, and actions are
dicts, and rendering them to language is the translator LLM's job in a
separate module. The self-check verifies the information design: a
report-blind policy earns nothing on average, a report-reading oracle
earns clearly more, and games are deterministic given a seed.

    python -m slm.market
"""

import argparse
import random
import statistics

from . import seeds

FIELDS = [
    {'product': 'umbrellas', 'demand_factor': 'rain'},
    {'product': 'ice cream', 'demand_factor': 'heat'},
    {'product': 'luggage', 'demand_factor': 'travel'},
    {'product': 'firewood', 'demand_factor': 'cold weather'},
]

MATERIALS = ['plastic', 'canvas', 'steel', 'paper', 'timber']

SHOCK_LEVELS = [-1, 0, 1]

DEMAND_WEIGHT = 4.0
COST_WEIGHT = 2.0
NOISE_SIGMA = 3.0
STARTING_PRICE = 100.0
STARTING_CASH = 1000.0
REPORT_COVERAGE = 0.7


def sample_market(random_generator, field_count=3, companies_per_field=2):
    """Return a market specification: fields, companies, and their factors.

    Each company carries its field's demand factor positively and its own
    material's cost factor negatively; companies within a field are
    assigned distinct materials so they compete on input costs.
    """
    fields = random_generator.sample(FIELDS, field_count)
    companies = []
    for field in fields:
        materials = random_generator.sample(MATERIALS, companies_per_field)
        for material in materials:
            name = '%s %s' % (
                seeds.invented_name(random_generator),
                field['product'].title(),
            )
            companies.append({
                'name': name,
                'product': field['product'],
                'demand_factor': field['demand_factor'],
                'material': material,
                'cost_factor': '%s price' % material,
            })
    demand_factors = [field['demand_factor'] for field in fields]
    cost_factors = sorted({company['cost_factor'] for company in companies})
    return {
        'companies': companies,
        'demand_factors': demand_factors,
        'cost_factors': cost_factors,
    }


def _sample_shocks(market, random_generator):
    shocks = {}
    for factor in market['demand_factors'] + market['cost_factors']:
        shocks[factor] = random_generator.choice(SHOCK_LEVELS)
    return shocks


def expected_return(company, known_shocks):
    """Expected next-quarter price change of a company given leaked shocks.

    Unknown shocks have mean zero, so they contribute nothing; this is the
    quantity an ideal reader of the reports can compute, and what the
    advisor's recommendation is derived from.
    """
    value = 0.0
    if company['demand_factor'] in known_shocks:
        value += DEMAND_WEIGHT * known_shocks[company['demand_factor']]
    if company['cost_factor'] in known_shocks:
        value -= COST_WEIGHT * known_shocks[company['cost_factor']]
    return value


def _build_reports(market, pending_shocks, random_generator,
                   report_coverage=REPORT_COVERAGE, advisor_coverage=1.0):
    """Leak a partial view of the pending shocks as reports.

    report_coverage is the per-factor leak probability and
    advisor_coverage the probability that the advisor speaks at all this
    quarter: the two difficulty dials. Less news means real hold
    decisions; less advisor means the model must compose the factor
    reports itself instead of following recommendations.
    """
    factors = market['demand_factors'] + market['cost_factors']
    leaked = {}
    reports = []
    for factor in factors:
        if random_generator.random() < report_coverage:
            leaked[factor] = pending_shocks[factor]
            reports.append({
                'source': 'news',
                'kind': 'factor',
                'factor': factor,
                'level': pending_shocks[factor],
            })
    if leaked and random_generator.random() < advisor_coverage:
        ranked = sorted(
            market['companies'],
            key=lambda company: expected_return(company, leaked),
        )
        best = ranked[-1]
        worst = ranked[0]
        if expected_return(best, leaked) > 0:
            reports.append({
                'source': 'advisor',
                'kind': 'recommendation',
                'company': best['name'],
                'stance': 'buy',
            })
        if expected_return(worst, leaked) < 0:
            reports.append({
                'source': 'advisor',
                'kind': 'recommendation',
                'company': worst['name'],
                'stance': 'sell',
            })
    random_generator.shuffle(reports)
    return reports, leaked


def coverage_at(start, final, quarter_index, quarters):
    """Linear within-game ramp from start to final coverage; None holds flat."""
    if final is None or quarters <= 1:
        return start
    fraction = quarter_index / (quarters - 1)
    return start + (final - start) * fraction


def start_game(market, random_generator, report_coverage=REPORT_COVERAGE,
               advisor_coverage=1.0):
    """Return the opening state: flat prices, cash, and quarter-one reports."""
    prices = {
        company['name']: STARTING_PRICE for company in market['companies']
    }
    holdings = {company['name']: 0 for company in market['companies']}
    pending_shocks = _sample_shocks(market, random_generator)
    reports, leaked = _build_reports(market, pending_shocks, random_generator,
                                     report_coverage, advisor_coverage)
    return {
        'quarter': 1,
        'prices': prices,
        'cash': STARTING_CASH,
        'holdings': holdings,
        'pending_shocks': pending_shocks,
        'reports': reports,
        'leaked_shocks': leaked,
        'last_earnings': 0.0,
    }


def portfolio_value(state):
    value = state['cash']
    for name, quantity in state['holdings'].items():
        value += quantity * state['prices'][name]
    return value


def apply_actions(state, actions):
    """Apply buy and sell actions at current prices, ignoring invalid ones.

    Returns the list of actions actually executed. Invalid actions (unknown
    company, unaffordable buy, overdrawn sell) are skipped rather than
    failed: guaranteeing well-formed actions is the listener's job, and the
    simulator stays permissive so a partially valid turn still acts.
    """
    executed = []
    for action in actions:
        name = action.get('company')
        if name not in state['prices']:
            continue
        quantity = action.get('quantity', 0)
        if not isinstance(quantity, int) or quantity <= 0:
            continue
        price = state['prices'][name]
        if action.get('action') == 'buy':
            cost = quantity * price
            if cost <= state['cash']:
                state['cash'] -= cost
                state['holdings'][name] += quantity
                executed.append(action)
        elif action.get('action') == 'sell':
            if quantity <= state['holdings'][name]:
                state['cash'] += quantity * price
                state['holdings'][name] -= quantity
                executed.append(action)
    return executed


def step_game(market, state, actions, random_generator,
              noise_sigma=NOISE_SIGMA, report_coverage=REPORT_COVERAGE,
              advisor_coverage=1.0):
    """Advance one quarter: trade, resolve pending shocks, report the next.

    Actions are applied at current prices, then the pre-sampled shocks move
    prices, and the score is the resulting change in portfolio value. New
    shocks are then sampled for the following quarter and partially leaked
    as the next state's reports. Randomness consumption is independent of
    the actions taken, so games sharing a generator seed see identical
    shock and report streams whatever the trader does. The coverage
    parameters apply to the next quarter's reports, so a per-quarter
    caller can ramp difficulty within a game.
    """
    value_before = portfolio_value(state)
    executed = apply_actions(state, actions)
    for company in market['companies']:
        shocks = state['pending_shocks']
        change = (
            DEMAND_WEIGHT * shocks[company['demand_factor']]
            - COST_WEIGHT * shocks[company['cost_factor']]
            + random_generator.gauss(0.0, noise_sigma)
        )
        price = state['prices'][company['name']]
        state['prices'][company['name']] = max(1.0, price + change)
    earnings = portfolio_value(state) - value_before
    state['quarter'] += 1
    state['last_earnings'] = earnings
    state['pending_shocks'] = _sample_shocks(market, random_generator)
    state['reports'], state['leaked_shocks'] = _build_reports(
        market, state['pending_shocks'], random_generator,
        report_coverage, advisor_coverage,
    )
    return earnings, executed


def state_summary(state):
    """Structured content of the quarterly state message, for rendering."""
    held = {
        name: quantity
        for name, quantity in state['holdings'].items() if quantity > 0
    }
    return {
        'quarter': state['quarter'],
        'cash': round(state['cash'], 2),
        'holdings': held,
        'prices': {name: round(price, 2)
                   for name, price in state['prices'].items()},
        'last_earnings': round(state['last_earnings'], 2),
    }


def blind_policy(market, state, random_generator):
    """Trade randomly without reading reports; the chance baseline."""
    actions = []
    name = random_generator.choice(list(state['prices']))
    if random_generator.random() < 0.5:
        affordable = int(state['cash'] // state['prices'][name])
        if affordable > 0:
            actions.append({'action': 'buy', 'company': name,
                            'quantity': random_generator.randint(1, affordable)})
    elif state['holdings'][name] > 0:
        actions.append({'action': 'sell', 'company': name,
                        'quantity': state['holdings'][name]})
    return actions


def oracle_policy(market, state, random_generator):
    """Read the leaked shocks perfectly and hold only the best company.

    An upper reference, not a training target: sells everything, then puts
    all cash on the company with the highest positive expected return.
    """
    actions = []
    for name, quantity in state['holdings'].items():
        if quantity > 0:
            actions.append({'action': 'sell', 'company': name,
                            'quantity': quantity})
    leaked = state['leaked_shocks']
    best = None
    best_value = 0.0
    for company in market['companies']:
        value = expected_return(company, leaked)
        if value > best_value:
            best = company
            best_value = value
    if best is not None:
        cash = state['cash']
        for action in actions:
            cash += action['quantity'] * state['prices'][action['company']]
        quantity = int(cash // state['prices'][best['name']])
        if quantity > 0:
            actions.append({'action': 'buy', 'company': best['name'],
                            'quantity': quantity})
    return actions


def play_game(policy, seed, quarters=12, field_count=3,
              companies_per_field=2, noise_sigma=NOISE_SIGMA,
              report_coverage=REPORT_COVERAGE, advisor_coverage=1.0,
              report_coverage_final=None, advisor_coverage_final=None):
    """Play one full game under a policy; return total and per-step earnings.

    The final coverage values, when set, ramp the difficulty linearly
    across the game's quarters, matching what the trainer does, so
    reference policies are measured under the same conditions.
    """
    random_generator = random.Random(seed)
    market = sample_market(random_generator, field_count, companies_per_field)
    state = start_game(
        market, random_generator,
        coverage_at(report_coverage, report_coverage_final, 0, quarters),
        coverage_at(advisor_coverage, advisor_coverage_final, 0, quarters),
    )
    earnings_by_quarter = []
    for quarter in range(quarters):
        next_index = min(quarter + 1, quarters - 1)
        actions = policy(market, state, random_generator)
        earnings, _ = step_game(
            market, state, actions, random_generator, noise_sigma,
            coverage_at(report_coverage, report_coverage_final,
                        next_index, quarters),
            coverage_at(advisor_coverage, advisor_coverage_final,
                        next_index, quarters),
        )
        earnings_by_quarter.append(earnings)
    return sum(earnings_by_quarter), earnings_by_quarter


def _mean_and_error(values):
    mean = statistics.mean(values)
    error = statistics.stdev(values) / (len(values) ** 0.5)
    return mean, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--quarters', type=int, default=12)
    parser.add_argument('--seed', type=int, default=7)
    arguments = parser.parse_args()

    blind_totals = []
    oracle_totals = []
    for game_index in range(arguments.games):
        seed = arguments.seed + game_index
        blind_totals.append(
            play_game(blind_policy, seed, arguments.quarters)[0])
        oracle_totals.append(
            play_game(oracle_policy, seed, arguments.quarters)[0])

    blind_mean, blind_error = _mean_and_error(blind_totals)
    oracle_mean, oracle_error = _mean_and_error(oracle_totals)
    print('blind policy:  mean %+.2f (standard error %.2f)'
          % (blind_mean, blind_error))
    print('oracle policy: mean %+.2f (standard error %.2f)'
          % (oracle_mean, oracle_error))

    blind_at_chance = abs(blind_mean) < 3 * blind_error
    oracle_ahead = (oracle_mean - blind_mean) > 3 * (
        (blind_error ** 2 + oracle_error ** 2) ** 0.5)
    replay_a = play_game(oracle_policy, arguments.seed, arguments.quarters)
    replay_b = play_game(oracle_policy, arguments.seed, arguments.quarters)
    deterministic = replay_a == replay_b

    print('blind at chance: %s' % ('pass' if blind_at_chance else 'FAIL'))
    print('oracle ahead:    %s' % ('pass' if oracle_ahead else 'FAIL'))
    print('deterministic:   %s' % ('pass' if deterministic else 'FAIL'))
    return 0 if (blind_at_chance and oracle_ahead and deterministic) else 1


if __name__ == '__main__':
    raise SystemExit(main())

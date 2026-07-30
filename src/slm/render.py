"""Template rendering of market structures into game text.

Turns the simulator's structured states and reports into the dialogue-style
text the SGM reads, using the same name-and-colon turn format as the free
dialogue corpus but with the game's own speaker labels. Prices and amounts
are rendered as integers to keep the token budget small. Template variants
supply mild surface variety; rendering through the LLM instead is a later
switch, which is why every function takes a random generator now.

The model's own turn is cued by the trader label with a colon; whatever the
model writes after it is the decision turn the listener interprets.
"""

STATE_SPEAKER = 'Broker'
NEWS_SPEAKER = 'News'
ADVISOR_SPEAKER = 'Advisor'
MODEL_SPEAKER = 'Trader'

_DEMAND_WORDS = {1: 'strong', 0: 'steady', -1: 'weak'}
_COST_WORDS = {1: 'rise', 0: 'hold steady', -1: 'fall'}


def _holdings_phrase(summary):
    if not summary['holdings']:
        return 'no shares'
    parts = []
    for name, quantity in sorted(summary['holdings'].items()):
        parts.append('%d shares of %s' % (quantity, name))
    return ', '.join(parts)


def _prices_phrase(summary):
    parts = []
    for name, price in sorted(summary['prices'].items()):
        parts.append('%s at %d' % (name, round(price)))
    return ', '.join(parts)


def render_state_message(summary, random_generator):
    earnings = round(summary['last_earnings'])
    if earnings > 0:
        earnings_phrase = 'You earned %d last quarter.' % earnings
    elif earnings < 0:
        earnings_phrase = 'You lost %d last quarter.' % -earnings
    else:
        earnings_phrase = 'You broke even last quarter.'
    openings = [
        'It is quarter %d.' % summary['quarter'],
        'Quarter %d has begun.' % summary['quarter'],
    ]
    return '%s: %s You have %d in cash and hold %s. Prices are %s. %s' % (
        STATE_SPEAKER,
        random_generator.choice(openings),
        round(summary['cash']),
        _holdings_phrase(summary),
        _prices_phrase(summary),
        earnings_phrase,
    )


def render_report_message(report, market, random_generator):
    if report['kind'] == 'factor':
        factor = report['factor']
        if factor in market['demand_factors']:
            templates = [
                'Expect %s to be %s next quarter.',
                'Forecasts say %s will be %s next quarter.',
            ]
            word = _DEMAND_WORDS[report['level']]
        else:
            templates = [
                'The %s is expected to %s next quarter.',
                'Traders expect the %s to %s next quarter.',
            ]
            word = _COST_WORDS[report['level']]
        return '%s: %s' % (
            NEWS_SPEAKER,
            random_generator.choice(templates) % (factor, word),
        )
    if report['stance'] == 'buy':
        templates = [
            'Consider buying %s this quarter.',
            'I would buy %s before the quarter turns.',
        ]
    else:
        templates = [
            'Consider selling %s this quarter.',
            'I would let go of %s before the quarter turns.',
        ]
    return '%s: %s' % (
        ADVISOR_SPEAKER,
        random_generator.choice(templates) % report['company'],
    )


PROTOCOL_MESSAGE = (
    '%s: State your orders each quarter as buy or sell, a number of '
    'shares, the company name, and the reason, for example: buy 2 shares '
    'of the strongest company because its demand is rising. Say hold if '
    'you want no trade.' % STATE_SPEAKER
)

STRUCTURED_TRADER_CUE = 'trader:'


def render_exemplar_exchange(market, random_generator):
    """Scripted example exchange shown once before the first quarter.

    A base model imitates transcript patterns far more readily than it
    follows instructions, so the order format appears once as an actual
    trader turn before the model's first cue. The company rotates per
    game, and the broker closes by saying nothing was traded, so the
    exchange stays consistent with the opening state. Rendered context
    only: trader spans never cover it, so it is never trained on.
    """
    company = random_generator.choice(market['companies'])
    return '\n'.join([
        '%s: Before the first quarter, one example exchange in the '
        'required form.' % STATE_SPEAKER,
        '%s: buy 2 shares of %s because %s is expected to be strong.' % (
            MODEL_SPEAKER, company['name'], company['demand_factor'],
        ),
        '%s: That is the form. The example is over; nothing was '
        'traded.' % STATE_SPEAKER,
    ])


def render_structured_state(summary):
    holdings = ', '.join(
        '%d %s' % (quantity, name)
        for name, quantity in sorted(summary['holdings'].items())
    ) or 'none'
    prices = ', '.join(
        '%s %d' % (name, round(price))
        for name, price in sorted(summary['prices'].items())
    )
    return 'state: quarter %d | cash %d | holdings %s | prices %s | earned %d' % (
        summary['quarter'], round(summary['cash']), holdings, prices,
        round(summary['last_earnings']),
    )


def render_structured_report(report, market, random_generator=None,
                             input_variety=False, numeric_reports='off'):
    """One structured report line, optionally with synonyms and numbers.

    The canonical words are the first entry of each level's synonym set
    in the listener (the vocabulary owner). With input_variety a random
    synonym is drawn instead, so 'strong' sometimes reads 'high' or
    'booming': the level is carried by a small word family rather than
    one token, which keeps single tokens from becoming condition codes
    the policy can key on without reading.

    numeric_reports renders the shock's signed weight - the number the
    market itself uses (demand +-4, cost +-2) - either alongside the
    word ('rain strong +4', the co-rendered on-ramp where the number is
    redundant) or instead of it ('rain +4', where composing the weighted
    sum is the only way to pick the best company). Annealing both to
    only through curriculum rungs is the learning-gap control.
    """
    from .listener import COST_LEVEL_WORDS, DEMAND_LEVEL_WORDS
    from .market import COST_WEIGHT, DEMAND_WEIGHT

    if report['kind'] == 'factor':
        demand = report['factor'] in market['demand_factors']
        words = (DEMAND_LEVEL_WORDS if demand
                 else COST_LEVEL_WORDS)[report['level']]
        if input_variety and random_generator is not None:
            word = random_generator.choice(words)
        else:
            word = words[0]
        weight = DEMAND_WEIGHT if demand else COST_WEIGHT
        number = '%+d' % int(report['level'] * weight)
        if numeric_reports == 'only':
            detail = number
        elif numeric_reports == 'both':
            detail = '%s %s' % (word, number)
        else:
            detail = word
        return 'news: %s %s next quarter' % (report['factor'], detail)
    return 'advisor: %s %s' % (report['stance'], report['company'])


def render_structured_quarter(state, market, random_generator=None,
                              input_variety=False, numeric_reports='off'):
    """The structured register: one field-labeled line per message.

    Lowercase labels, bar-separated state fields, no prose - deliberately
    disjoint from the dialogue register so the simulation lives in its
    own register and interferes with ordinary language as little as
    possible. The register is taught by the template SFT stage, not
    instructed in context, so there is no protocol line and no exemplar.
    Rendering is deterministic by default; input_variety draws level-word
    synonyms per report line, the one controlled source of surface
    variety.
    """
    from .market import state_summary

    lines = [render_structured_state(state_summary(state))]
    for report in state['reports']:
        lines.append(render_structured_report(report, market,
                                              random_generator,
                                              input_variety,
                                              numeric_reports))
    lines.append(STRUCTURED_TRADER_CUE)
    return '\n'.join(lines)


def render_quarter(state, market, random_generator, protocol_line=True,
                   exemplar_turn=False, decision_format='freeform',
                   input_variety=False, numeric_reports='off'):
    """Render one quarter's context block, ending at the model's cue.

    Returns the block text whose last line is the trader cue, with no
    trailing newline, so the model's generation continues the line. In
    the freeform format the first quarter can carry a broker protocol
    message stating the order format and a scripted exemplar exchange
    demonstrating it: in-context material for a cold-start model, never
    training data. The structured format renders its own register and
    ignores both flags - the template SFT stage teaches it instead.
    """
    from .market import state_summary

    if decision_format == 'structured':
        return render_structured_quarter(state, market, random_generator,
                                         input_variety, numeric_reports)
    lines = []
    if exemplar_turn and state['quarter'] == 1:
        lines.append(render_exemplar_exchange(market, random_generator))
    lines.append(render_state_message(state_summary(state), random_generator))
    if protocol_line and state['quarter'] == 1:
        lines.append(PROTOCOL_MESSAGE)
    for report in state['reports']:
        lines.append(render_report_message(report, market, random_generator))
    lines.append('%s:' % MODEL_SPEAKER)
    return '\n'.join(lines)

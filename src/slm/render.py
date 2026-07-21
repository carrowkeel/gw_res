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


def render_quarter(state, market, random_generator):
    """Render one quarter's context block, ending at the model's cue.

    Returns the block text whose last line is the trader label and colon,
    with no trailing newline, so the model's generation continues the line.
    """
    from .market import state_summary

    lines = [render_state_message(state_summary(state), random_generator)]
    for report in state['reports']:
        lines.append(render_report_message(report, market, random_generator))
    lines.append('%s:' % MODEL_SPEAKER)
    return '\n'.join(lines)

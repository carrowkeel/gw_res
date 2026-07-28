"""The forgiving listener: gate and interpret trader turns into orders.

The listener sits between the SGM's free-text decision turn and the
simulator. It gates on the reason-bearing format (a decision must carry a
reason; a bare decision is not acted on) but does not grade the reason:
flawed reasoning still acts. Interpretation is deliberately charitable so
that near-miss outputs still move the game and outcome signal is nonzero
from the first games; strictness is a dial to be tightened over training
and eventually removed.

Two modes. The pattern mode parses with regular expressions and fuzzy
company matching, runs anywhere, and is the smoke-test and strict
end-state form. The llm mode asks the translator LLM to rewrite the
trader's message into canonical order lines first, then parses those with
the same machinery, which is what makes the interface a slope rather than
a cliff for a weakly trained model. Every result carries a match label
(exact, fuzzy, none) whose rates are the progress metric toward canonical
output.
"""

import re

REASON_MARKERS = ['because', 'since ', 'as the', 'as it', 'given that']

_ORDER_PATTERN = re.compile(
    r'\b(buy|sell)(?:ing)?\b\s+(?:(\d+|all)\s+)?(?:shares?\s+(?:of\s+)?)?',
    re.IGNORECASE,
)

_HOLD_PATTERN = re.compile(
    r'\bhold\b(?!\s+steady)|\bkeep\b(?!\s+an\s+eye)|\bno\s+trades?\b',
    re.IGNORECASE,
)

LISTENER_SYSTEM_PROMPT = (
    'You review one turn spoken by a trader and translate it into exact '
    'orders. Reply with three parts, each starting on its own line. '
    'First a line SCORE: <number>, a whole number from 1 to 5 grading '
    'only the grammar and coherence of the trader\'s wording, never the '
    'quality of the trade. Second a line FIX: <sentence>, a minimally '
    'corrected version of the turn: repair repetition, broken grammar, '
    'and cut-off endings while keeping the trader\'s own words, order, '
    'names, and numbers wherever possible, never inventing a different '
    'trade; if the wording is already clean, repeat it unchanged. Third, '
    'one line per order in exactly this form: ORDER: buy <quantity> '
    '<company> or ORDER: sell <quantity> <company>, using only company '
    'names from the given list and whole-number quantities. Interpret '
    'charitably: if the trader plainly wants to trade a company, produce '
    'the order even if the wording is loose; use quantity 1 if none is '
    'given, and the word all for selling an entire holding. If no trade '
    'is intended, reply ORDER: none.'
)


def reason_given(text):
    lowered = text.lower()
    return any(marker in lowered for marker in REASON_MARKERS)


def hold_stated(text):
    """Whether the text states a deliberate hold, in the trained wording.

    The bridge teaches holds as 'hold the X and make no trade' with hold or
    keep as the verb, so those words mark the intent; the price-forecast
    phrase 'hold steady' and the idiom 'keep an eye' are excluded because
    they describe the market, not a move.
    """
    return bool(_HOLD_PATTERN.search(text))


def parse_review(rewrite):
    """Split a listener reply into (score, correction, order text).

    Score is clamped to 1..5 and None when absent; correction is None when
    absent or empty. The returned order text holds only the ORDER lines so
    trade words inside the correction sentence are never parsed as orders;
    when the reply carries no ORDER line at all the whole reply is
    returned, which keeps an older-style plain reply parseable.
    """
    score = None
    correction = None
    order_lines = []
    for line in rewrite.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith('SCORE:'):
            digits = re.search(r'\d+', stripped)
            if digits:
                score = max(1, min(5, int(digits.group())))
        elif upper.startswith('FIX:'):
            candidate = stripped[len('FIX:'):].strip()
            correction = candidate or None
        elif upper.startswith('ORDER:'):
            order_lines.append(stripped)
    order_text = '\n'.join(order_lines) if order_lines else rewrite
    return score, correction, order_text


def _match_company(fragment, companies):
    lowered = fragment.lower()
    for company in companies:
        if company['name'].lower() in lowered:
            return company['name'], 'exact'
    for company in companies:
        first_word = company['name'].split()[0].lower()
        if re.search(r'\b%s\b' % re.escape(first_word), lowered):
            return company['name'], 'fuzzy'
    for company in companies:
        product = company['product'].lower()
        if product in lowered:
            return company['name'], 'fuzzy'
    return None, 'none'


def parse_orders(text, market, state):
    """Extract orders from free text; the shared core of both modes.

    Returns (actions, match) where match is the weakest company-match
    quality seen (exact before fuzzy before none) so callers can track how
    canonical the output was. The match label reflects company naming
    alone: an order that names a real company but fails the feasibility
    clamp keeps its match so reports can tell an infeasible order from an
    unparseable one.
    """
    actions = []
    ranking = {'exact': 0, 'fuzzy': 1, 'none': 2}
    worst = None
    for found in _ORDER_PATTERN.finditer(text):
        verb = found.group(1).lower()
        quantity_word = found.group(2)
        tail = text[found.end():found.end() + 60]
        company, match = _match_company(tail, market['companies'])
        if company is None:
            continue
        if worst is None or ranking[match] > ranking[worst]:
            worst = match
        if quantity_word is None:
            quantity = 1
        elif quantity_word.lower() == 'all':
            quantity = max(1, state['holdings'].get(company, 0))
        else:
            quantity = int(quantity_word)
        if verb == 'buy':
            price = state['prices'][company]
            affordable = int(state['cash'] // price)
            quantity = min(quantity, max(affordable, 0))
        else:
            quantity = min(quantity, state['holdings'].get(company, 0))
        if quantity > 0:
            actions.append(
                {'action': verb, 'company': company, 'quantity': quantity}
            )
    return actions, worst or 'none'


def _gate_passes(has_reason, no_reason_probability, random_generator):
    if has_reason:
        return True
    if no_reason_probability <= 0.0 or random_generator is None:
        return False
    return random_generator.random() < no_reason_probability


def interpret(text, market, state, no_reason_probability=0.0,
              random_generator=None):
    """Pattern-mode interpretation of one trader turn.

    The reason gate is soft: a turn with a reason always proceeds to
    parsing, a reasonless turn proceeds with no_reason_probability, so
    reasons make the listener reliable rather than being an absolute
    precondition. Annealing the probability toward zero recovers the
    strict gate as the end state.
    """
    has_reason = reason_given(text)
    if not _gate_passes(has_reason, no_reason_probability, random_generator):
        return {'actions': [], 'reason_given': has_reason, 'match': 'none',
                'acted': False, 'rewrite': None, 'language_score': None,
                'correction': None}
    actions, match = parse_orders(text, market, state)
    return {'actions': actions, 'reason_given': has_reason, 'match': match,
            'acted': bool(actions), 'rewrite': None, 'language_score': None,
            'correction': None}


def _rewrite_prompt(text, market, state):
    company_names = ', '.join(
        company['name'] for company in market['companies']
    )
    holdings = ', '.join(
        '%d %s' % (quantity, name)
        for name, quantity in state['holdings'].items() if quantity > 0
    ) or 'none'
    return (
        'Companies: %s. Holdings: %s. Cash: %d.\nTrader says: %s'
        % (company_names, holdings, round(state['cash']), text.strip())
    )


class LlmListener:
    """Charitable interpretation through the translator LLM.

    The LLM rewrites each trader turn into canonical ORDER lines, and the
    pattern machinery parses those. Loading vLLM is deferred to the first
    call so the class can be constructed anywhere. Turns are interpreted in
    batches, one rewrite call per turn, batched through the engine.
    """

    def __init__(self, model_name, generate_config):
        self.model_name = model_name
        self.generate_config = generate_config
        self.engine = None
        self.sampling = None

    def _ensure_engine(self):
        if self.engine is None:
            from .generate import _load_engine

            self.engine, self.sampling = _load_engine(
                self.model_name, self.generate_config
            )

    def interpret_batch(self, turns, no_reason_probability=0.0,
                        random_generator=None):
        """Interpret [(text, market, state), ...] into result dicts."""
        results = [None] * len(turns)
        pending = []
        reasons = {}
        for index, (text, market, state) in enumerate(turns):
            has_reason = reason_given(text)
            reasons[index] = has_reason
            if _gate_passes(has_reason, no_reason_probability,
                            random_generator):
                pending.append(index)
            else:
                results[index] = {'actions': [], 'reason_given': has_reason,
                                  'match': 'none', 'acted': False,
                                  'rewrite': None, 'language_score': None,
                                  'correction': None}
        if pending:
            self._ensure_engine()
            from .generate import _chat

            prompts = [
                _rewrite_prompt(*turns[index]) for index in pending
            ]
            rewrites = _chat(
                self.engine, self.sampling, LISTENER_SYSTEM_PROMPT, prompts
            )
            for index, rewrite in zip(pending, rewrites):
                text, market, state = turns[index]
                score, correction, order_text = parse_review(rewrite)
                actions, match = parse_orders(order_text, market, state)
                if actions:
                    direct, direct_match = parse_orders(text, market, state)
                    if direct != actions:
                        match = 'fuzzy'
                results[index] = {'actions': actions,
                                  'reason_given': reasons[index],
                                  'match': match, 'acted': bool(actions),
                                  'rewrite': rewrite,
                                  'language_score': score,
                                  'correction': correction}
        return results

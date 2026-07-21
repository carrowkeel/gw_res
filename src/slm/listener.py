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
    r'\b(buy|sell)\b(?:ing)?\s+(?:(\d+|all)\s+)?(?:shares?\s+(?:of\s+)?)?',
    re.IGNORECASE,
)

LISTENER_SYSTEM_PROMPT = (
    'You translate a trader\'s instruction into exact orders. Reply with '
    'one line per order in exactly this form: ORDER: buy <quantity> '
    '<company> or ORDER: sell <quantity> <company>, using only company '
    'names from the given list and whole-number quantities. Interpret '
    'charitably: if the trader plainly wants to trade a company, produce '
    'the order even if the wording is loose; use quantity 1 if none is '
    'given, and the word all for selling an entire holding. If no trade '
    'is intended, reply ORDER: none. Output only ORDER lines.'
)


def reason_given(text):
    lowered = text.lower()
    return any(marker in lowered for marker in REASON_MARKERS)


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
    canonical the output was.
    """
    actions = []
    ranking = {'exact': 0, 'fuzzy': 1, 'none': 2}
    worst = 'exact'
    for found in _ORDER_PATTERN.finditer(text):
        verb = found.group(1).lower()
        quantity_word = found.group(2)
        tail = text[found.end():found.end() + 60]
        company, match = _match_company(tail, market['companies'])
        if company is None:
            worst = 'none' if not actions else worst
            continue
        if ranking[match] > ranking[worst]:
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
    if not actions:
        worst = 'none'
    return actions, worst


def interpret(text, market, state):
    """Pattern-mode interpretation of one trader turn.

    Gates on the reason requirement, then parses orders directly from the
    trader's own words.
    """
    has_reason = reason_given(text)
    if not has_reason:
        return {'actions': [], 'reason_given': False, 'match': 'none',
                'acted': False}
    actions, match = parse_orders(text, market, state)
    return {'actions': actions, 'reason_given': True, 'match': match,
            'acted': bool(actions)}


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

    def interpret_batch(self, turns):
        """Interpret [(text, market, state), ...] into result dicts."""
        results = [None] * len(turns)
        pending = []
        for index, (text, market, state) in enumerate(turns):
            if not reason_given(text):
                results[index] = {'actions': [], 'reason_given': False,
                                  'match': 'none', 'acted': False}
            else:
                pending.append(index)
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
                actions, match = parse_orders(rewrite, market, state)
                if actions:
                    direct, direct_match = parse_orders(text, market, state)
                    if direct != actions:
                        match = 'fuzzy'
                results[index] = {'actions': actions, 'reason_given': True,
                                  'match': match, 'acted': bool(actions)}
        return results

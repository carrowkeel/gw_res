"""The listener: gate and interpret trader turns, review their language.

The listener sits between the SGM's free-text decision turn and the
simulator. It gates on the reason-bearing format (a decision must carry a
reason; a bare decision is not acted on) but does not grade the reason:
flawed reasoning still acts. Orders are always parsed from the trader's
raw text by the pattern machinery: charitable rewriting of vague turns
into trades was the dial to be tightened over training, and it was
removed the moment a model passed the entry gate on raw text, after a
run learned to trade the charity instead of the market. Every result
carries a match label (exact, fuzzy, none) whose rates are the progress
metric toward canonical output.

The review LLM has one remaining job: grade each turn's grammar and
coherence and return a minimal correction. It never touches the order
path.

Two decision formats are supported. The freeform format reads orders out
of natural sentences with the permissive pattern below. The structured
format expects one strict template line, 'move: buy 3 Krouket Umbrellas
| reason: rain will be strong', and parses nothing else: it is the
parseable-template end state where the simulator needs no language
understanding at all, only the template.
"""

import re

REASON_MARKERS = ['because', 'since ', 'as the', 'as it', 'given that']

_ORDER_PATTERN = re.compile(
    r'\b(buy|sell)(?:ing)?\b\s+(?:(\d+|all)\s+)?(?:shares?\s+(?:of\s+)?)?',
    re.IGNORECASE,
)

_STRUCTURED_ORDER_PATTERN = re.compile(
    r'move:\s*(buy|sell)\s+(\d+|all)\s+(?:shares?\s+(?:of\s+)?)?'
    r'([^|]+?)\s*\|\s*reason:\s*(\S.*)',
    re.IGNORECASE,
)

_STRUCTURED_HOLD_PATTERN = re.compile(
    r'move:\s*hold\s*\|\s*reason:\s*(\S.*)',
    re.IGNORECASE,
)

_HOLD_PATTERN = re.compile(
    r'\bhold\b(?!\s+steady)|\bkeep\b(?!\s+an\s+eye)|\bno\s+trades?\b',
    re.IGNORECASE,
)

REVIEW_SYSTEM_PROMPT = (
    'You review one turn spoken by a trader. Reply with exactly two '
    'lines. First SCORE: <number>, a whole number from 1 to 5 grading '
    'only the grammar and coherence of the trader\'s wording, never the '
    'quality of the trade; repetitive or broken wording scores 1. Second '
    'FIX: <sentence>, a minimally corrected version of the turn: repair '
    'repetition, broken grammar, and cut-off endings while keeping the '
    'trader\'s own words, order, names, and numbers wherever possible, '
    'never inventing a different trade; if the wording is already clean, '
    'repeat it unchanged. Output nothing after the FIX line.'
)


def reason_given(text):
    lowered = text.lower()
    return any(marker in lowered for marker in REASON_MARKERS)


def reason_offset(text, decision_format='freeform'):
    """Character index where the turn's reason part starts, None if absent.

    Freeform turns start their reason at the earliest reason marker;
    structured turns start it at the bar separating the move field from
    the reason field. This is the boundary the order-clause loss scope
    cuts at, so everything from the offset onward is reason, not order.
    """
    if decision_format == 'structured':
        position = text.find('|')
        return position if position >= 0 else None
    lowered = text.lower()
    positions = [lowered.find(marker) for marker in REASON_MARKERS]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else None


def reason_text(text, decision_format='freeform'):
    """The turn's reason part, empty when it gives none.

    For freeform turns this is the tail from the earliest reason marker,
    marker included; for structured turns it is the reason field of the
    template. Both formats agree that a turn has a reason exactly when
    this is non-empty.
    """
    if decision_format == 'structured':
        found = (_STRUCTURED_ORDER_PATTERN.search(text)
                 or _STRUCTURED_HOLD_PATTERN.search(text))
        return found.group(found.lastindex).strip() if found else ''
    offset = reason_offset(text)
    return text[offset:].strip() if offset is not None else ''


def grounded_reason(reason, market):
    """Whether a reason cites the market's causal vocabulary.

    Grounding asks for a term the news reports speak in - a demand
    factor or a material - rather than a company or product name: a
    reason that only names what it trades is self-reference, not
    evidence the reports were read, and the template-collapse run
    produced exactly that. Products are excluded because company names
    contain them, so they would let self-reference through. A grounded
    reason is not necessarily correct; it merely cites something a
    report could have said.
    """
    lowered = reason.lower()
    if not lowered:
        return False
    terms = list(market['demand_factors'])
    for company in market['companies']:
        terms.append(company['material'])
    return any(term.lower() in lowered for term in terms)


def hold_stated(text):
    """Whether the text states a deliberate hold, in the trained wording.

    The bridge teaches holds as 'hold the X and make no trade' with hold or
    keep as the verb, so those words mark the intent; the price-forecast
    phrase 'hold steady' and the idiom 'keep an eye' are excluded because
    they describe the market, not a move.
    """
    return bool(_HOLD_PATTERN.search(text))


def repetitive(text):
    """Whether a turn repeats itself: any four-word run occurring twice.

    The collapse signature is a repeated order stub. An ordinary decision
    sentence does not repeat a four-word run, so this is the
    program-owned degeneracy test that the review LLM's compressed score
    scale cannot provide.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    seen = set()
    for index in range(len(words) - 3):
        gram = tuple(words[index:index + 4])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def parse_review(reply):
    """Split a review reply into (score, correction), first labels win.

    Score is clamped to 1..5 and None when absent; correction is None
    when absent or empty. The reviewing LLM sometimes appends a
    hallucinated second review block, so the first occurrence of each
    label is kept and later ones ignored.
    """
    score = None
    correction = None
    for line in reply.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith('SCORE:') and score is None:
            digits = re.search(r'\d+', stripped)
            if digits:
                score = max(1, min(5, int(digits.group())))
        elif upper.startswith('FIX:') and correction is None:
            candidate = stripped[len('FIX:'):].strip()
            correction = candidate or None
    return score, correction


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


def _clamp_quantity(verb, company, quantity_word, state):
    """Resolve a quantity word into a feasible share count, possibly zero."""
    if quantity_word is None:
        quantity = 1
    elif quantity_word.lower() == 'all':
        quantity = max(1, state['holdings'].get(company, 0))
    else:
        quantity = int(quantity_word)
    if verb == 'buy':
        price = state['prices'][company]
        affordable = int(state['cash'] // price)
        return min(quantity, max(affordable, 0))
    return min(quantity, state['holdings'].get(company, 0))


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
        quantity = _clamp_quantity(verb, company, quantity_word, state)
        if quantity > 0:
            actions.append(
                {'action': verb, 'company': company, 'quantity': quantity}
            )
    return actions, worst or 'none'


def parse_structured(text, market, state):
    """Strict template parse: the first 'move: ... | reason: ...' line only.

    A turn that does not carry the template parses as nothing at all -
    the strictness is the point of the format, so there is no fallback to
    the freeform pattern and at most one order comes out. The match label
    stays truthful the same way parse_orders keeps it: a template naming
    a real company keeps its match even when the feasibility clamp
    empties the order.
    """
    found = _STRUCTURED_ORDER_PATTERN.search(text)
    if found is None:
        return [], 'none'
    verb = found.group(1).lower()
    company, match = _match_company(found.group(3), market['companies'])
    if company is None:
        return [], 'none'
    quantity = _clamp_quantity(verb, company, found.group(2), state)
    if quantity <= 0:
        return [], match
    return (
        [{'action': verb, 'company': company, 'quantity': quantity}], match
    )


def parse_decision(text, market, state, decision_format='freeform'):
    """Parse one trader turn under the configured decision format."""
    if decision_format == 'structured':
        return parse_structured(text, market, state)
    return parse_orders(text, market, state)


def _gate_passes(has_reason, no_reason_probability, random_generator):
    if has_reason:
        return True
    if no_reason_probability <= 0.0 or random_generator is None:
        return False
    return random_generator.random() < no_reason_probability


def interpret(text, market, state, no_reason_probability=0.0,
              random_generator=None, decision_format='freeform'):
    """Pattern-mode interpretation of one trader turn.

    The reason gate is soft: a turn with a reason always proceeds to
    parsing, a reasonless turn proceeds with no_reason_probability, so
    reasons make the listener reliable rather than being an absolute
    precondition. Annealing the probability toward zero recovers the
    strict gate as the end state.
    """
    has_reason = bool(reason_text(text, decision_format))
    if not _gate_passes(has_reason, no_reason_probability, random_generator):
        return {'actions': [], 'reason_given': has_reason, 'match': 'none',
                'acted': False, 'rewrite': None, 'language_score': None,
                'correction': None}
    actions, match = parse_decision(text, market, state, decision_format)
    return {'actions': actions, 'reason_given': has_reason, 'match': match,
            'acted': bool(actions), 'rewrite': None, 'language_score': None,
            'correction': None}


def _review_prompt(text, market, state):
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
    """Language review through the listener LLM: score and minimal fix.

    Orders never pass through here: they are parsed from the trader's raw
    text by the pattern machinery, so the reviewer's charity cannot
    ground a vague turn into a trade. Loading vLLM is deferred to the
    first call so the class can be constructed anywhere. Turns are
    reviewed one call per turn, batched through the engine.
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

    def review_batch(self, turns):
        """Score and minimally correct [(text, market, state), ...]."""
        self._ensure_engine()
        from .generate import _chat

        prompts = [_review_prompt(*turn) for turn in turns]
        replies = _chat(
            self.engine, self.sampling, REVIEW_SYSTEM_PROMPT, prompts
        )
        reviews = []
        for reply in replies:
            score, correction = parse_review(reply)
            reviews.append({'score': score, 'correction': correction,
                            'rewrite': reply})
        return reviews

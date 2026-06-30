"""Best-effort filter for real-world-referent leakage.

This is a heuristic guardrail, not a guarantee. Generation occasionally slips
in real entities, numbers, or facts despite the system prompt, so the obvious
offenders are dropped. Knowledge absence is a degree, and this filter raises
that degree; it does not assert an absolute.

Severity ``s2`` additionally rejects long runs of capitalized words, which tend
to signal named specifics.
"""

import re

BLOCKLIST = {
    'america', 'england', 'france', 'china', 'africa', 'europe', 'asia',
    'london', 'paris', 'tokyo', 'york', 'russia', 'india', 'germany',
    'earth', 'mars', 'moon', 'sun',
    'god', 'jesus', 'president', 'king', 'queen', 'einstein', 'napoleon',
    'google', 'apple', 'amazon', 'microsoft', 'facebook', 'twitter',
    'iphone', 'android', 'internet', 'computer', 'phone', 'television', 'car',
    'school', 'money', 'dollar', 'euro', 'church', 'doctor',
    'gravity', 'electricity', 'covid', 'war', 'history', 'science',
}

_DIGIT_PATTERN = re.compile(r'\d')
_URL_PATTERN = re.compile(r'https?://|www\.|\S+@\S+\.\S+|[@#]\w+')
_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_PROPER_RUN_PATTERN = re.compile(r'(?<=[a-z,] )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')


def check_text(text, severity='s1', blocklist=None):
    """Return a list of reasons the text fails the filter, empty when it passes."""
    active_blocklist = blocklist if blocklist is not None else BLOCKLIST
    reasons = []
    if _DIGIT_PATTERN.search(text):
        reasons.append('contains digits')
    if _URL_PATTERN.search(text):
        reasons.append('contains url, handle, or email')
    words = {match.lower() for match in _WORD_PATTERN.findall(text)}
    hits = sorted(words & active_blocklist)
    if hits:
        reasons.append('blocklist: ' + ', '.join(hits[:5]))
    if severity == 's2' and _PROPER_RUN_PATTERN.search(text):
        reasons.append('proper-noun phrase')
    return reasons


def passes(text, severity='s1', blocklist=None):
    return not check_text(text, severity, blocklist)

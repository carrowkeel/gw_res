"""Best-effort filter for real-world-referent and contamination leakage.

This is a heuristic guardrail, not a guarantee. Generation occasionally slips
in real entities, numbers, facts, assistant framing, or meta-commentary despite
the system prompt, so the obvious offenders are dropped. Knowledge absence is a
degree, and this filter raises that degree; it does not assert an absolute.

Three classes of leakage are caught: digits and urls; single real-world or
technology words (the word blocklist); and assistant or meta phrases such as
self-reference to being a model or to the act of writing (the phrase blocklist).
Severity ``s2`` additionally rejects long runs of capitalized words.
"""

import re

BLOCKLIST = {
    'america', 'england', 'france', 'china', 'africa', 'europe', 'asia',
    'london', 'paris', 'tokyo', 'york', 'russia', 'india', 'germany',
    'mars', 'jupiter', 'saturn', 'venus', 'mercury', 'planet',
    'god', 'jesus', 'president', 'king', 'queen', 'einstein', 'napoleon',
    'google', 'apple', 'amazon', 'microsoft', 'facebook', 'twitter',
    'iphone', 'android', 'internet', 'computer', 'phone', 'television', 'car',
    'school', 'money', 'dollar', 'euro', 'church', 'doctor',
    'gravity', 'electricity', 'covid', 'war', 'history', 'science',
    'ai', 'model', 'models', 'program', 'programs', 'programmed',
    'information', 'software', 'hardware', 'algorithm', 'algorithms',
    'digital', 'online', 'website', 'robot', 'application', 'keyboard',
    'password', 'email',
    'lens', 'refract', 'refraction', 'convex', 'concave', 'prism', 'magnify',
    'optics', 'optical', 'wavelength', 'molecule', 'molecular', 'atom',
    'electron', 'proton', 'neutron', 'chemical', 'geometry', 'geometric',
    'polygon', 'triangle', 'rectangle', 'hexagon', 'pentagon', 'vertex',
    'vertices', 'perimeter', 'radius', 'diameter', 'circumference',
    'perpendicular', 'diagonal', 'degrees', 'equation', 'decimal',
    'coordinate', 'coordinates', 'velocity', 'acceleration', 'formula',
    'calculate', 'calculated', 'multiply', 'subtract',
}

META_PHRASES = [
    'as an ai', 'as a language model', 'language model', 'i am an ai',
    'i am a conscious', 'i have been programmed', 'i was programmed',
    'as an assistant', 'as a narrator of', "i'm sorry, but",
    'i cannot fulfill', 'i cannot help with', "i can't help with",
    'in this passage', 'in this story', 'in this text', 'in this conversation',
    'the following passage', 'the following story', 'we will explore',
    'we will look at', 'in conclusion', 'to summarize', 'disclaimer',
    'user:', 'assistant:',
]

_DIGIT_PATTERN = re.compile(r'\d')
_URL_PATTERN = re.compile(r'https?://|www\.|\S+@\S+\.\S+|[@#]\w+')
_WORD_PATTERN = re.compile(r"[A-Za-z']+")
_PROPER_RUN_PATTERN = re.compile(r'(?<=[a-z,] )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')


def check_text(text, severity='s1', blocklist=None):
    """Return a list of reasons the text fails the filter, empty when it passes."""
    active_blocklist = blocklist if blocklist is not None else BLOCKLIST
    lowered = text.lower()
    reasons = []
    if _DIGIT_PATTERN.search(text):
        reasons.append('contains digits')
    if _URL_PATTERN.search(text):
        reasons.append('contains url, handle, or email')
    words = {match.lower() for match in _WORD_PATTERN.findall(text)}
    word_hits = sorted(words & active_blocklist)
    if word_hits:
        reasons.append('blocklist: ' + ', '.join(word_hits[:5]))
    phrase_hits = [phrase for phrase in META_PHRASES if phrase in lowered]
    if phrase_hits:
        reasons.append('meta phrase: ' + ', '.join(phrase_hits[:5]))
    if severity == 's2' and _PROPER_RUN_PATTERN.search(text):
        reasons.append('proper-noun phrase')
    return reasons


def passes(text, severity='s1', blocklist=None):
    return not check_text(text, severity, blocklist)

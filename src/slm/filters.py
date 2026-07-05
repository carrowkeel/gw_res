"""Best-effort filter for generation contamination.

This is a heuristic guardrail, not a guarantee. Generation occasionally slips
in assistant framing or meta-commentary despite the system prompt (for example
self-reference to being a model, or to the act of writing), so the obvious
offenders are dropped, along with stray urls, handles, and email addresses.

Referent-free severity restrictions (real-world word blocklisting, digit
banning, proper-noun-phrase rejection) are currently relaxed for the
prompt-response MVP; see the intent graph for the plan to reintroduce them
through a constructed world-state generator rather than through filtering.
"""

import re

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

_URL_PATTERN = re.compile(r'https?://|www\.|\S+@\S+\.\S+|[@#]\w+')


def check_text(text):
    """Return a list of reasons the text fails the filter, empty when it passes."""
    lowered = text.lower()
    reasons = []
    if _URL_PATTERN.search(text):
        reasons.append('contains url, handle, or email')
    phrase_hits = [phrase for phrase in META_PHRASES if phrase in lowered]
    if phrase_hits:
        reasons.append('meta phrase: ' + ', '.join(phrase_hits[:5]))
    return reasons


def passes(text):
    return not check_text(text)

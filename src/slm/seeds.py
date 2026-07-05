"""Seed vocabulary for diversifying synthetic generation prompts.

These lists supply topic and structure primitives (subject domains, narrative
premises, reasoning modes, tones, forms) that prompts sample from so requests
range widely over real subjects and demand real structure, rather than
producing variations on a single scene. Content is not restricted to generic
or invented referents; that restriction is relaxed (see prompts.py).
"""

SUBJECT_DOMAINS = [
    'everyday life and routines',
    'work, jobs, and trade',
    'science and the natural world',
    'history and the past',
    'the arts, music, and literature',
    'friendship, family, and relationships',
    'health, medicine, and the body',
    'technology, machines, and how they work',
    'food, cooking, and farming',
    'travel, cities, and places',
    'law, government, and institutions',
    'sport, games, and competition',
    'money, business, and economics',
    'learning, language, and ideas',
    'craft, building, and making things',
    'weather, seasons, and the land',
]

STORY_SITUATIONS = [
    'someone returns to a place they left years ago and finds it changed',
    'a small lie grows until it can no longer be controlled',
    'two people want the same thing and only one can have it',
    'a person must choose between what is safe and what they want',
    'a stranger arrives asking for something difficult to give',
    'a careful plan goes wrong at the worst possible moment',
    'someone discovers a secret they were not meant to know',
    'a long friendship is tested by a single decision',
    'a person tries to fix a mistake and only makes it worse',
    'someone is given responsibility they are not ready for',
    'a promise made lightly comes due',
    'a newcomer upsets the settled order of a group',
    'someone must earn the trust of a person who has no reason to give it',
    'a person finally gets what they wanted and finds it is not enough',
    'an ordinary day is interrupted by an unexpected arrival',
    'someone must decide whether to tell a truth that will cost them',
    'a debt, spoken or unspoken, is finally called in',
    'a person hides a failure and has to keep hiding it',
]

PROSE_FORMS = [
    'a short story with a clear beginning, a turn, and an end',
    'a scene in which a decision is reached and acted on',
    'a story told by someone looking back on what they did',
    'an account of how a situation escalated and then resolved',
    'a moment of conflict between two people and its outcome',
    'a story in which a character wants something and is opposed',
    'a story that turns on a single choice',
]

TONES = [
    'plain', 'wry', 'warm', 'matter-of-fact', 'tense', 'measured', 'brisk',
    'rueful', 'earnest', 'sardonic', 'gentle', 'urgent',
]

POINTS_OF_VIEW = [
    'the third person, following one character',
    'the first person, the narrator involved in events',
    'someone recalling it long afterward',
    'the third person, moving between two characters',
]

LENGTH_BANDS = [
    'a few sentences', 'a short paragraph', 'two short paragraphs',
    'several short paragraphs',
]

DIALOGUE_GOALS = [
    'settle a disagreement about how to handle a shared problem',
    'negotiate who does which part of a shared task',
    'one tries to persuade the other to change a decision',
    'work out the order in which a series of events happened',
    'weigh two courses of action under a real constraint',
    'one breaks difficult news and the other takes it in',
    'divide a limited resource between them fairly',
    'one asks the other for help and explains why it is needed',
    'one teaches the other how to do something, meeting objections',
    'reconcile after a disagreement, each giving some ground',
    'plan something together and discover they want different things',
    'one confronts the other about a broken promise',
]

REASONING_MODES = [
    'explain how something works, step by step, so a careful reader could '
    'follow it',
    'explain why something happens, giving the cause before the effect',
    'lay out the steps to accomplish a task, in the order they must be done',
    'make a reasoned case for a position, giving the reasons in order of '
    'weight',
    'weigh two options against each other and reach a definite conclusion',
    'work a problem through from what is given to what follows from it',
    'trace how one change leads to another and then to a result',
]

RELATION_KINDS = ['spatial', 'comparative', 'ordinal', 'temporal', 'causal',
                  'functional', 'part-and-whole']

_NAME_ONSETS = [
    'Bl', 'Tr', 'Fl', 'Gr', 'Sn', 'Wi', 'Pl', 'Dr', 'Kr', 'Mu', 'Lo', 'Vi',
    'Ze', 'Qu', 'No', 'Ti', 'Ro', 'Ha', 'Pe', 'Su', 'Ca', 'Fe', 'Ma', 'Ne',
]
_NAME_NUCLEI = ['a', 'o', 'i', 'e', 'u', 'ee', 'oo', 'ai', 'ou', 'ia']
_NAME_CODAS = [
    'mar', 'len', 'dis', 'por', 'ven', 'tor', 'mel', 'ras', 'nel', 'dor',
    'sen', 'lim', 'tan', 'rin', 'vel', 'ket', 'mon', 'der', 'sel', 'fen',
]


def invented_name(random_generator):
    """Return a single pronounceable invented name for a character or speaker."""
    onset = random_generator.choice(_NAME_ONSETS)
    nucleus = random_generator.choice(_NAME_NUCLEI)
    coda = random_generator.choice(_NAME_CODAS)
    return (onset + nucleus + coda).capitalize()


def sample_domains(random_generator, count):
    """Return distinct subject domains to anchor a prompt in real content."""
    if count <= len(SUBJECT_DOMAINS):
        return random_generator.sample(SUBJECT_DOMAINS, count)
    return [random_generator.choice(SUBJECT_DOMAINS) for _ in range(count)]

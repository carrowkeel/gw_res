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
    'a short paragraph', 'two paragraphs', 'three paragraphs',
    'four or five paragraphs',
]

DIALOGUE_GOALS = [
    'settle a disagreement about how to handle a shared problem',
    'negotiate who does which part of a shared task',
    'argue over a decision one of them wants changed',
    'work out the order in which a series of events happened',
    'weigh two courses of action under a real constraint',
    'break difficult news, one telling and the other taking it in',
    'divide a limited resource between them fairly',
    'arrange help, one asking and the other explaining what it will take',
    'go over how to do something, one teaching and the other objecting',
    'reconcile after a disagreement, each giving some ground',
    'plan something together and discover they want different things',
    'have out a broken promise between them',
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


INTERACTION_REGISTERS = [
    'a casual chat among friends',
    'a workplace exchange between colleagues',
    'a customer talking with a clerk at a counter',
    'a support exchange between a user and an agent',
    'an exchange of short notes between two people',
    'a radio exchange between a field team and their base',
    'a classroom exchange between a teacher and students',
    'a planning meeting around a table',
    'two neighbors talking over a fence',
    'a market-stall exchange between a seller and buyers',
    'a crew coordinating in the middle of a task',
    'a family working out plans at the kitchen table',
    'a caller asking an office for information',
    'travellers comparing notes on the road',
]

HELDOUT_REGISTERS = [
    'a patient describing a matter to a busy receptionist',
    'a formal exchange during an inspection visit',
    'two strangers thrown together by a delay, talking to pass the time',
]

SPEAKER_ROLES = [
    'User', 'Agent', 'Customer', 'Clerk', 'Teacher', 'Student', 'Manager',
    'Worker', 'Buyer', 'Seller', 'Captain', 'Base', 'Driver', 'Guide',
    'Visitor', 'Caller', 'Nurse', 'Farmer', 'Cook', 'Porter',
]

NAMING_STYLES = ['invented', 'role', 'initial']

DECISION_SITUATIONS = [
    'the wind is turning against a small boat still far from shore',
    'a delivery has not arrived and the buyer is due within the hour',
    'rain is starting over hay that is still cut and lying in the field',
    'a pot has boiled over and the next course is already late',
    'the road ahead is closed and the appointment cannot be moved',
    'a machine is running hot and the spare part is a day away',
    'the till is short and the shop closes in ten minutes',
    'two guests have been given the same room for tonight',
    'the river is rising toward the footbridge before the crossing',
    'a ladder has been left up with a storm coming in',
    'the last bus has gone and the tickets are for the morning',
    'an order was doubled by mistake and the extra stock is arriving',
    'a lamp has started flickering in the middle of close work',
    'the key to the storeroom is missing at opening time',
    'a queue is forming faster than one counter can serve it',
]


def sample_speakers(style, count, random_generator):
    """Return distinct speaker names for a naming style.

    invented draws pronounceable made-up names, role draws plain role nouns
    (User, Agent, Clerk), and initial uses single letters, so training sees
    every convention a prompt might use to label its speakers.
    """
    if style == 'role':
        return random_generator.sample(SPEAKER_ROLES, count)
    if style == 'initial':
        letters = [chr(ordinal) for ordinal in range(ord('A'), ord('Z') + 1)]
        return random_generator.sample(letters, count)
    names = []
    while len(names) < count:
        name = invented_name(random_generator)
        if name not in names:
            names.append(name)
    return names

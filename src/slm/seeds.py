"""Generic, referent-free seed vocabulary for synthetic generation.

These lists supply world primitives at the level of categories rather than
identifiable real entities. They are sampled to diversify generation prompts
without ever pulling in real names, numbers, or facts. Everything here is
deliberately generic: a river, not a named river.

Severity controls how specific the allowed nouns are. At ``s1`` concrete
generic nouns (forest, lake, stone) are allowed. At ``s2`` only category-level
terms (a region, a body of water, a surface) are used.
"""

GENERIC_ENTITIES = [
    'forest', 'wood', 'lake', 'pond', 'river', 'stream', 'hill', 'ridge',
    'valley', 'field', 'meadow', 'grove', 'marsh', 'shore', 'slope', 'path',
    'clearing', 'hollow', 'stone', 'rock', 'boulder', 'sand', 'soil', 'root',
    'branch', 'leaf', 'flower', 'seed', 'tree', 'plant', 'reed', 'moss',
    'animal', 'bird', 'fish', 'insect', 'creature', 'nest', 'den', 'track',
    'wind', 'rain', 'mist', 'frost', 'cloud', 'shade', 'light', 'sound',
]

CATEGORY_ENTITIES = [
    'a region', 'a body of water', 'a watercourse', 'a rise in the ground',
    'a low place', 'an open area', 'an enclosed area', 'a surface',
    'a boundary', 'a passage', 'a structure', 'a formation', 'a substance',
    'a creature', 'a smaller creature', 'a growth', 'a covering',
    'a movement of air', 'a fall of water', 'a change in the light',
    'a marking', 'a cluster', 'an opening', 'an edge', 'a hollow',
]

SPATIAL_RELATIONS = [
    'beside', 'near', 'above', 'below', 'beyond', 'within', 'between',
    'around', 'behind', 'in front of', 'along', 'across from', 'at the edge of',
    'in the middle of', 'on the far side of', 'just past',
]

COMPARATIVE_RELATIONS = [
    'larger than', 'smaller than', 'longer than', 'shorter than',
    'higher than', 'lower than', 'older than', 'closer than', 'farther than',
    'wider than', 'narrower than', 'deeper than', 'darker than', 'quieter than',
]

ORDINAL_RELATIONS = [
    'before', 'after', 'first', 'last', 'next', 'earlier', 'later',
    'the one nearest', 'the one farthest', 'the only one that',
]

QUALITIES = [
    'still', 'moving', 'open', 'closed', 'rough', 'smooth', 'pale', 'dim',
    'bright', 'narrow', 'wide', 'shallow', 'deep', 'bare', 'covered',
    'distant', 'near', 'quiet', 'restless', 'steady', 'changing',
]

CONVERSATION_SITUATIONS = [
    'noticing that a path has changed',
    'deciding whether to go on or turn back',
    'comparing two places they have both seen',
    'working out where a sound came from',
    'agreeing on what to do about something one of them found',
    'explaining to the other how something is arranged',
    'disagreeing about which way is shorter',
    'describing a place the other has not been to',
    'settling who should do which part of a task',
    'remembering how a place looked before it changed',
]

ABSTRACT_SYMBOLS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'P', 'Q', 'R', 'S', 'X', 'Y', 'Z',
]

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
    """Return a single pronounceable invented name with no real referent."""
    onset = random_generator.choice(_NAME_ONSETS)
    nucleus = random_generator.choice(_NAME_NUCLEI)
    coda = random_generator.choice(_NAME_CODAS)
    return (onset + nucleus + coda).capitalize()


def invented_term(random_generator):
    """Return a lowercase invented common-noun-like term for definitions."""
    return invented_name(random_generator).lower()


def entity_pool(severity):
    """Return the noun pool allowed at the given severity level."""
    if severity == 's2':
        return CATEGORY_ENTITIES
    return GENERIC_ENTITIES


def sample_entities(random_generator, severity, count):
    pool = entity_pool(severity)
    if count <= len(pool):
        return random_generator.sample(pool, count)
    return [random_generator.choice(pool) for _ in range(count)]

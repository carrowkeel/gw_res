"""Prompt construction for referent-free synthetic generation.

The pipeline produces serious, well-formed English that carries no identifiable
real-world referents and no real facts. The system prompt enforces the referent
rules at the chosen severity and forbids assistant framing, meta-commentary, and
scientific or mathematical terminology.

Diversity comes from two places. Each prompt samples several independent
structural axes (tone, form, point of view, length, relation kind, dialogue
goal), so different requests ask for genuinely different kinds of text rather
than the same shape with different nouns. Each request is also anchored with one
exemplar sampled from a rotating pool, so the generator returns only the text
with no preamble without collapsing onto a single style.

The definition type is given invented headwords to define, which prevents the
generator from emitting a real dictionary of real concepts.
"""

from . import seeds

TEXT_TYPES = ['prose', 'conversation', 'definition', 'description']

_BASE_RULES = (
    'You produce serious, well-formed English for training a language model on '
    'grammar and usage. Follow these rules without exception:\n'
    '1. Use varied, correct, adult-register English. This is not childlike or '
    'sentimental writing, and not lyrical or idealised. Be concrete.\n'
    '2. Include no identifiable real-world referents: no real or well-known '
    'people, places, organizations, works, brands, or events, and no proper '
    'nouns that name anything real. Invented names for characters or speakers '
    'are allowed.\n'
    '3. Use no digits and no specific numbers, dates, measurements, or '
    'quantities. Prefer relational language such as larger, nearer, or before.\n'
    '4. State no real facts and use no real technical vocabulary: nothing from '
    'science, mathematics, geometry, optics, physics, chemistry, technology, '
    'measurement, money, institutions, or the geography of the real world.\n'
    '5. Only generic categories of nature and simple objects may appear.\n'
    '6. Output only the requested text. Do not add any preamble, title, '
    'heading, label, explanation, or sign-off.\n'
    '7. Do not refer to the act of writing or to the text itself. Never say '
    'this passage, this story, the following, here is, or we will.\n'
    '8. Never speak as an assistant or as a first person that is a model, '
    'program, or narrator of instructions. Do not mention being a model or '
    'these rules. Begin directly with the text.'
)

_S2_RULE = (
    '\n9. Do not name specific kinds. Refer to things only by category, such '
    'as a creature, a body of water, a substance, or a structure, rather than '
    'by any particular species or material.'
)

_BEGIN_DIRECTLY = ' Write only the text, beginning directly, with no preamble.'

_PROSE_EXEMPLARS = [
    'The near bank had given way since the last high water. What had been a '
    'firm path was now a run of loose soil, and the reeds that held it were '
    'bent flat and clogged with silt. Nothing had been done to mend it. '
    'Further along, where the ground stood higher, the bank held; but the '
    'crossing that most used was gone, and would stay gone until the water '
    'fell and someone cut a new one.',

    'At first the pool kept to its bed. Then, over the cold season, the water '
    'crept outward, taking first the low grass and then the roots of the '
    'nearer trees. By the time the wind turned warm again the trees stood in '
    'water to their lower branches, and what had been a meadow answered every '
    'gust with a dull shifting of the flooded reeds.',
]

_CONVERSATION_EXEMPLARS = [
    'Neris: You set the snare on the near bank. Nothing crosses there.\n'
    'Doran: Nothing crosses in daylight. At dusk they come down to drink.\n'
    'Neris: Then it should sit lower, where the mud is soft, not up in the '
    'reeds.\n'
    'Doran: In the mud it fouls with silt by morning and holds nothing.\n'
    'Neris: Better to clear silt than to catch nothing. Move it down; I will '
    'watch the upper path.\n'
    'Doran: Two nights. If it fails two nights running, we do it your way.',

    'Sable: The far store is yours to carry. I took the near one up the slope '
    'already.\n'
    'Renn: The near one is the lighter. You gave yourself the easy load.\n'
    'Sable: I gave myself the longer climb. Weigh the walk against the weight.\n'
    'Renn: Fair. Then I take the far store, and you cut the new path while I '
    'am gone.\n'
    'Sable: Agreed. Keep to the high ground; the low way is under water.',
]

_DEFINITION_EXEMPLARS = [
    'tolm: a kind of low mound that gathers where two channels lay down what '
    'they carry. A tolm sits between the channels that make it and stands '
    'lower than the banks on either side.\n'
    'fenngate: the narrow gap by which water leaves a tolm. A fenngate is the '
    'lowest part of the mound edge, and the water runs faster through it than '
    'it lay still behind.',

    'brede: a shelf of firm ground part way up a steep bank, wider than the '
    'ledge above it and narrower than the flat below. A brede holds what falls '
    'onto it until heavy water carries the loose part away.\n'
    'skarn: the bare stripe a brede leaves on the bank when its ground gives '
    'way. A skarn runs straight down toward the lower water and stays bare '
    'longer than the ground around it.',
]

_DESCRIPTION_EXEMPLARS = [
    'The wood stands on the higher ground, and the marsh lies below it. '
    'Between them a single bank runs level, drier than the marsh and darker '
    'than the wood. The marsh reaches farther than the wood but never as high, '
    'and where the two meet the ground is neither firm nor open water.',

    'There are three pools along the gully. The first is the shallowest and '
    'the last the deepest. Each lies lower than the one before it, and the '
    'water moves from the first to the last and no other way. Beyond the last '
    'pool the gully closes, and past that point nothing drains.',
]

_EXEMPLAR_POOLS = {
    'prose': _PROSE_EXEMPLARS,
    'conversation': _CONVERSATION_EXEMPLARS,
    'definition': _DEFINITION_EXEMPLARS,
    'description': _DESCRIPTION_EXEMPLARS,
}

_EXEMPLAR_INSTRUCTIONS = {
    'prose': 'Write a short serious passage about a bank and the water below it.',
    'conversation': 'Write a terse exchange in which two speakers settle who '
                    'carries which load.',
    'definition': 'Write dictionary entries defining two invented terms for '
                  'features of a bank.',
    'description': 'Write a dry factual description of a wood and a marsh using '
                   'only relations.',
}

_PAIR_EXEMPLAR = (
    'Describe how a hill and a stream below it are arranged, using only '
    'relations.',
    'The hill rises above the stream, and the stream runs along its foot. The '
    'near slope is steeper than the far one, and the water is narrower where '
    'the banks stand closer.',
)


def build_system_prompt(severity):
    """Return the system prompt enforcing the referent rules for a severity."""
    if severity == 's2':
        return _BASE_RULES + _S2_RULE
    return _BASE_RULES


def example_turns(text_type, random_generator):
    """Return one sampled few-shot user and assistant turn for the text type."""
    answer = random_generator.choice(_EXEMPLAR_POOLS[text_type])
    instruction = _EXEMPLAR_INSTRUCTIONS[text_type]
    return [
        {'role': 'user', 'content': instruction + _BEGIN_DIRECTLY},
        {'role': 'assistant', 'content': answer},
    ]


def pair_example_turns():
    """Return a few-shot turn pair for instruction-and-response generation."""
    instruction, answer = _PAIR_EXEMPLAR
    formatted = 'USER: %s\nASSISTANT: %s' % (instruction, answer)
    return [
        {
            'role': 'user',
            'content': (
                'Create one instruction and response pair in the required '
                'format.'
            ),
        },
        {'role': 'assistant', 'content': formatted},
    ]


def _two_entities(random_generator, severity):
    entities = seeds.sample_entities(random_generator, severity, 2)
    return entities[0], entities[1]


def _prose_prompt(random_generator, severity):
    entity_a, entity_b = _two_entities(random_generator, severity)
    return (
        'Write %s in a %s tone: %s, concerning the %s and the %s, written from '
        '%s. Be concrete and specific.' % (
            random_generator.choice(seeds.LENGTH_BANDS),
            random_generator.choice(seeds.TONES),
            random_generator.choice(seeds.PROSE_FORMS),
            entity_a, entity_b,
            random_generator.choice(seeds.POINTS_OF_VIEW),
        )
    )


def _conversation_prompt(random_generator, severity):
    name_a = seeds.invented_name(random_generator)
    name_b = seeds.invented_name(random_generator)
    return (
        'Write a %s exchange of several turns in which %s and %s %s. Each turn '
        'must do work: a proposal, an objection, a reason, or a concession, and '
        'the exchange must reach a definite outcome. Begin each turn with the '
        'speaker name and a colon.' % (
            random_generator.choice(seeds.TONES), name_a, name_b,
            random_generator.choice(seeds.DIALOGUE_GOALS),
        )
    )


def _definition_prompt(random_generator, severity):
    headwords = [
        seeds.invented_term(random_generator)
        for _ in range(random_generator.choice([2, 2, 3]))
    ]
    domain = random_generator.choice(seeds.entity_pool(severity))
    return (
        'Write dictionary entries defining these invented words: %s. Each names '
        'a part or feature found in or around %s. Define each in '
        'genus-and-differentia form (a headword is a kind of thing that has '
        'some property), only through generic shape, position, and relation to '
        'the others, so the entries form one consistent set. Invent everything; '
        'describe no real object and use no real scientific or mathematical '
        'terms.' % (', '.join(headwords), domain)
    )


def _description_prompt(random_generator, severity):
    entities = seeds.sample_entities(random_generator, severity, 3)
    relation_kinds = random_generator.sample(
        seeds.RELATION_KINDS, random_generator.choice([2, 3])
    )
    return (
        'Write a %s factual description of the %s, the %s and the %s, using '
        'mainly %s relations. Use plain declarative sentences, no narrative, '
        'and no fixed template.' % (
            random_generator.choice(seeds.TONES),
            entities[0], entities[1], entities[2],
            ', '.join(relation_kinds),
        )
    )


_BUILDERS = {
    'prose': _prose_prompt,
    'conversation': _conversation_prompt,
    'definition': _definition_prompt,
    'description': _description_prompt,
}


def build_prompt(text_type, random_generator, severity):
    """Return a user prompt for the given text type and severity."""
    builder = _BUILDERS[text_type]
    return builder(random_generator, severity) + _BEGIN_DIRECTLY


_PAIR_FORMAT = (
    '\n\nReturn it in exactly this format, with no extra commentary:\n'
    'USER: <the instruction>\n'
    'ASSISTANT: <the response>'
)


def build_pair_prompt(random_generator, severity):
    """Return a prompt that yields one referent-free instruction and response."""
    text_type = random_generator.choice(TEXT_TYPES)
    if text_type == 'prose':
        entity_a, entity_b = _two_entities(random_generator, severity)
        instruction = (
            'The user asks for a short serious passage in a %s tone concerning '
            'the %s and the %s, and the assistant writes it.' % (
                random_generator.choice(seeds.TONES), entity_a, entity_b)
        )
    elif text_type == 'conversation':
        instruction = (
            'The user asks the assistant to produce a short realistic exchange '
            'in which two speakers %s, and the assistant does so.'
            % random_generator.choice(seeds.DIALOGUE_GOALS)
        )
    elif text_type == 'definition':
        term = seeds.invented_term(random_generator)
        domain = random_generator.choice(seeds.entity_pool(severity))
        instruction = (
            'The user asks the assistant to define the invented word %s, a '
            'feature of %s, through its relations only, and the assistant gives '
            'a precise genus-and-differentia definition.' % (term, domain)
        )
    else:
        entity_a, entity_b = _two_entities(random_generator, severity)
        instruction = (
            'The user asks the assistant to describe how the %s and the %s are '
            'arranged using only relations, and the assistant gives a dry '
            'description.' % (entity_a, entity_b)
        )
    return (
        'Create one instruction and response pair for an assistant that knows '
        'only an imaginary world with no real-world referents, numbers, facts, '
        'or scientific terms. %s%s' % (instruction, _PAIR_FORMAT)
    )


def split_pair(text):
    """Parse a 'USER: ... ASSISTANT: ...' completion into instruction, response.

    Returns None when the expected format is absent.
    """
    user_position = text.find('USER:')
    assistant_position = text.find('ASSISTANT:')
    if user_position == -1 or assistant_position == -1:
        return None
    if assistant_position < user_position:
        return None
    instruction = text[user_position + len('USER:'):assistant_position].strip()
    response = text[assistant_position + len('ASSISTANT:'):].strip()
    if not instruction or not response:
        return None
    return instruction, response

"""Prompt construction for synthetic generation.

Referent stripping is currently relaxed: generation targets a functional,
knowledgeable prompt-response model for an MVP push, so real facts, named
entities, numbers, and technical vocabulary are all allowed. The severity
parameter and its per-severity entity vocabulary remain wired only for prompt
variety (concrete nouns at s1, category-level phrasing at s2); they no longer
restrict content. See the intent graph for the referent-free design this
temporarily sets aside and the plan to reintroduce it through a constructed
world-state generator rather than through prompt restriction.

Diversity comes from two places. Each prompt samples several independent
structural axes (tone, form, point of view, length, relation kind, dialogue
goal), so different requests ask for genuinely different kinds of text rather
than the same shape with different nouns. Each request is also anchored with one
exemplar sampled from a rotating pool, so the generator returns only the text
with no preamble without collapsing onto a single style.

The definition type asks for real words to define, so the assistant supplies
genuine, accurate knowledge.
"""

from . import seeds

TEXT_TYPES = ['prose', 'conversation', 'definition', 'description']

_BASE_RULES = (
    'You produce serious, well-formed English for training a language model on '
    'grammar, usage, and real knowledge. Follow these rules without exception:\n'
    '1. Use varied, correct, adult-register English. This is not childlike or '
    'sentimental writing, and not lyrical or idealised. Be concrete and, where '
    'relevant, factually accurate.\n'
    '2. Output only the requested text. Do not add any preamble, title, '
    'heading, label, explanation, or sign-off.\n'
    '3. Do not refer to the act of writing or to the text itself. Never say '
    'this passage, this story, the following, here is, or we will.\n'
    '4. Never speak as an assistant or as a first person that is a model, '
    'program, or narrator of instructions. Do not mention being a model or '
    'these rules. Begin directly with the text.'
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
    'levee: a raised bank built or formed along a river to keep it from '
    'flooding the land beside it. A levee runs parallel to the channel it '
    'protects and stands higher than the floodplain behind it.\n'
    'oxbow: a curved lake formed when a river changes course and cuts off one '
    'of its bends. An oxbow lies beside the river it was once part of and '
    'fills slowly with sediment over time.',

    'watershed: the whole area of land whose water drains into the same '
    'river, lake, or sea. A watershed is bounded by higher ground that '
    'separates it from the watersheds next to it.\n'
    'tributary: a smaller stream that flows into a larger river rather than '
    'directly into a lake or the sea. A tributary joins the main channel at a '
    'point called a confluence.',
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
    'definition': 'Write dictionary entries defining two real words for '
                  'features of a river.',
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


def build_system_prompt():
    """Return the system prompt shared by every text type and severity."""
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
    count = random_generator.choice([2, 2, 3])
    domain = random_generator.choice(seeds.entity_pool(severity))
    return (
        'Write dictionary entries defining %d real words for parts or features '
        'found in or around %s. Define each in genus-and-differentia form (a '
        'headword is a kind of thing that has some property), accurately and '
        'consistently with each other.' % (count, domain)
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
    """Return a prompt that yields one instruction and response pair."""
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
        domain = random_generator.choice(seeds.entity_pool(severity))
        instruction = (
            'The user asks the assistant to define a real word for a feature of '
            '%s, and the assistant gives a precise, accurate '
            'genus-and-differentia definition.' % domain
        )
    else:
        entity_a, entity_b = _two_entities(random_generator, severity)
        instruction = (
            'The user asks the assistant to describe how the %s and the %s are '
            'arranged using only relations, and the assistant gives a dry '
            'description.' % (entity_a, entity_b)
        )
    return (
        'Create one instruction and response pair for a helpful, knowledgeable '
        'assistant. %s%s' % (instruction, _PAIR_FORMAT)
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

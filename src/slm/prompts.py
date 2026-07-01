"""Prompt construction for referent-free synthetic generation.

The pipeline produces serious, well-formed English that carries no identifiable
real-world referents. The system prompt enforces the referent rules at the
chosen severity and forbids assistant framing and meta-commentary. Per-type
builders inject sampled generic seeds, and each request is anchored with a
few-shot exemplar so the generator returns only the text itself, with no
preamble. Four forms are implemented: prose, conversation, definition, and
description.
"""

from . import seeds

TEXT_TYPES = ['prose', 'conversation', 'definition', 'description']

_BASE_RULES = (
    'You produce serious, well-formed English for training a language model on '
    'grammar and usage. Follow these rules without exception:\n'
    '1. Use varied, correct, adult-register English. This is not childlike or '
    'sentimental writing.\n'
    '2. Include no identifiable real-world referents: no real or well-known '
    'people, places, organizations, works, brands, or events, and no proper '
    'nouns that name anything real. Invented names for characters or speakers '
    'are allowed.\n'
    '3. Use no digits and no specific numbers, dates, measurements, or '
    'quantities. Prefer relational language such as larger, nearer, or before.\n'
    '4. State no real facts of any kind: no history, science, technology, '
    'computers, programs, information systems, money, institutions, or '
    'geography of the real world.\n'
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

_EXEMPLARS = {
    'prose': (
        'Write a serious passage of several short paragraphs about a slope and '
        'the water below it.',
        'The slope fell away toward the water, and for a long while nothing '
        'moved upon it. Then the wind came down from the higher ground, '
        'bending the grass in slow waves, and a single bird rose from the reeds '
        'and turned against the pale light. It settled again farther along the '
        'shore, where the bank was steeper and the shade lay deep. Nothing '
        'followed it. The water held its level, dark and still, and the day '
        'went on as it had begun.',
    ),
    'conversation': (
        'Write a natural conversation of several turns between two speakers who '
        'are deciding whether to go on or turn back.',
        'Velmar: The path is not where it was. The water has taken the lower '
        'part of it.\n'
        'Tordis: Then we climb. The upper way is longer, but it is dry.\n'
        'Velmar: It is also steeper, and the light is already going.\n'
        'Tordis: Better steep and slow than to wade in the dark. I will go '
        'first.\n'
        'Velmar: Lead, then. I will keep close behind you.',
    ),
    'definition': (
        'Write dictionary-style entries that define terms only through their '
        'relations and properties.',
        'ridgeline: the upper edge where two slopes meet and the ground falls '
        'away on either side. A ridgeline stands higher than the land it '
        'divides, and runs longer than it is wide.\n'
        'hollow: a low place enclosed on most sides, into which water and shade '
        'gather. A hollow lies below the ground around it, and is quieter than '
        'the open.',
    ),
    'description': (
        'Write a dry, factual description that states only spatial and '
        'comparative relations among generic elements.',
        'In one of the valleys there is a lake. Beside it stands a wood, and '
        'the wood is older than the trees along the shore. The lake is wider '
        'than any other in the valley, and deeper at its far end than near the '
        'bank. Above both rises a ridge, bare and grey, and beyond the ridge '
        'the ground falls away again toward lower water.',
    ),
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


def example_turns(text_type):
    """Return few-shot user and assistant turns anchoring the output style."""
    instruction, answer = _EXEMPLARS[text_type]
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
    register = random_generator.choice(['narrative', 'reflective', 'expository'])
    if register == 'narrative':
        return (
            'Write a serious passage of several short paragraphs that narrates '
            'a sequence of events involving %s and %s. Keep the telling plain '
            'and observational.' % (entity_a, entity_b)
        )
    if register == 'reflective':
        return (
            'Write a reflective passage in which an unnamed observer considers '
            '%s and how it stands in relation to %s. Keep the tone measured and '
            'serious.' % (entity_a, entity_b)
        )
    return (
        'Write an expository passage that explains, in plain serious language, '
        'how %s and %s come to be arranged as they are.' % (entity_a, entity_b)
    )


def _conversation_prompt(random_generator, severity):
    name_a = seeds.invented_name(random_generator)
    name_b = seeds.invented_name(random_generator)
    situation = random_generator.choice(seeds.CONVERSATION_SITUATIONS)
    return (
        'Write a natural conversation of several turns between %s and %s, who '
        'are %s. Begin each turn with the speaker name and a colon. Keep it '
        'serious and realistic.' % (name_a, name_b, situation)
    )


def _definition_prompt(random_generator, severity):
    entity = random_generator.choice(seeds.entity_pool(severity))
    relation = random_generator.choice(seeds.SPATIAL_RELATIONS)
    return (
        'Write several dictionary-style entries. Invent the headword for each '
        'entry and define it only through its relations and properties, '
        'without describing any real object. Where it helps, refer to related '
        'terms with single letters, as in: A is any %s that lies %s a B.'
        % (entity, relation)
    )


def _description_prompt(random_generator, severity):
    entity_a, entity_b = _two_entities(random_generator, severity)
    comparative = random_generator.choice(seeds.COMPARATIVE_RELATIONS)
    spatial = random_generator.choice(seeds.SPATIAL_RELATIONS)
    return (
        'Write a dry, factual description that states only spatial and '
        'comparative relations among generic elements, with no narrative. '
        'Follow this style: there is %s; %s it is %s; the second is %s the '
        'first. Use plain declarative sentences throughout.'
        % (entity_a, spatial, entity_b, comparative)
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
            'The user asks for a short serious passage involving %s and %s, and '
            'the assistant writes it.' % (entity_a, entity_b)
        )
    elif text_type == 'conversation':
        situation = random_generator.choice(seeds.CONVERSATION_SITUATIONS)
        instruction = (
            'The user asks the assistant to continue or produce a short '
            'realistic exchange about %s, and the assistant does so.' % situation
        )
    elif text_type == 'definition':
        entity = random_generator.choice(seeds.entity_pool(severity))
        instruction = (
            'The user asks the assistant to define an invented term related to '
            'a %s through its relations only, and the assistant gives a precise '
            'definition.' % entity
        )
    else:
        entity_a, entity_b = _two_entities(random_generator, severity)
        instruction = (
            'The user asks the assistant to describe how %s and %s are arranged '
            'using only spatial and comparative relations, and the assistant '
            'gives a dry description.' % (entity_a, entity_b)
        )
    return (
        'Create one instruction and response pair for an assistant that knows '
        'only an imaginary world with no real-world referents, numbers, or '
        'facts. %s%s' % (instruction, _PAIR_FORMAT)
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

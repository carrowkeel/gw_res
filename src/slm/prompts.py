"""Prompt construction for referent-free synthetic generation.

The pipeline produces serious, well-formed English that carries no identifiable
real-world referents. The system prompt enforces the referent rules at the
chosen severity. Per-type builders inject sampled generic seeds to keep the
corpus diverse across four implemented forms: prose, conversation, definition,
and description.
"""

from . import seeds

TEXT_TYPES = ['prose', 'conversation', 'definition', 'description']

_BASE_RULES = (
    'You write serious, well-formed English for training a language model on '
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
    'money, institutions, or geography of the real world.\n'
    '5. Only generic categories of nature and simple objects may appear.\n'
    '6. Do not mention being a model or these instructions. Output only the '
    'requested text and nothing else.'
)

_S2_RULE = (
    '\n7. Do not name specific kinds. Refer to things only by category, such '
    'as a creature, a body of water, a substance, or a structure, rather than '
    'by any particular species or material.'
)


def build_system_prompt(severity):
    """Return the system prompt enforcing the referent rules for a severity."""
    if severity == 's2':
        return _BASE_RULES + _S2_RULE
    return _BASE_RULES


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
        'terms with single letters, as in: A is any %s that lies %s a B. Use a '
        'plain, precise reference register.' % (entity, relation)
    )


def _description_prompt(random_generator, severity):
    entity_a, entity_b = _two_entities(random_generator, severity)
    comparative = random_generator.choice(seeds.COMPARATIVE_RELATIONS)
    spatial = random_generator.choice(seeds.SPATIAL_RELATIONS)
    return (
        'Write a dry, factual description that states only spatial and '
        'comparative relations among generic elements, with no narrative. For '
        'the pattern, follow this style: there is %s; %s it is %s; the second '
        'is %s the first. Use plain declarative sentences throughout.'
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
    return builder(random_generator, severity)


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

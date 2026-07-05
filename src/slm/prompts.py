"""Prompt construction for synthetic generation.

Referent stripping is relaxed: generation targets a functional, knowledgeable
prompt-response model, so real facts, named entities, numbers, and technical
vocabulary are all allowed. Generation is deliberately not confined to any one
subject: every prompt is anchored in a sampled subject domain so the corpus
ranges over everyday life, work, science, history, the arts, relationships,
health, technology, and more, rather than repeating a single kind of scene.

Each text type also carries a structural demand. Prose asks for a story with
plot (a character who wants something, is opposed, and reaches an outcome),
not static observation. The reasoning type asks for genuinely ordered
explanation or argument (cause before effect, steps in order, reasons by
weight), so its logic cannot be faked with loose, poetic association.
Instruction pairs span many real task kinds (explain, how-to, compare, define,
answer, advise, summarize, rewrite, list, reason), so the model learns to do
tasks, not only to describe scenes.

Diversity comes from two places. Each prompt samples independent structural
axes (domain, tone, form, point of view, length, reasoning mode, task kind).
Each request is also anchored with one exemplar sampled from a rotating pool,
so the generator returns only the text with no preamble without collapsing
onto a single style.
"""

from . import seeds

TEXT_TYPES = ['prose', 'conversation', 'definition', 'description', 'reasoning']

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
    'Dornel had covered for the older clerk twice that month and said nothing '
    'both times. The third time the drawer came up short, the manager asked '
    'him straight out whether he had seen anything. He understood at once what '
    'the truth would cost the old man and what the lie would cost himself, and '
    'he said he had noticed nothing. That night the clerk caught him at the '
    'gate, pressed a folded note into his hand, and told him he was a fool. '
    'Dornel kept the note but never spent it, and for the rest of the year he '
    'counted the drawer twice before anyone else arrived.',

    'When Sable came back to the town after nine years she expected the bakery '
    'to be gone, and it was. What she had not expected was that the woman '
    'behind the new counter would know her name. They had been at school '
    'together. For a moment neither of them mentioned the money Sable had left '
    'owing, and then the woman did, lightly, as though it were a joke. Sable '
    'paid it there in coins and left with bread she did not want, and found on '
    'the walk back that the debt had been the last thing still tying her to '
    'the place.',
]

_CONVERSATION_EXEMPLARS = [
    'Renn: If we ship both orders on one truck we save the second fee.\n'
    'Mara: And if the truck is late we lose both customers instead of one.\n'
    'Renn: The truck has been late once in a year.\n'
    'Mara: Once was the week we could least afford it. Split them. I would '
    'rather pay the fee than write two apologies.\n'
    'Renn: Fine, both split. But the fee comes out of the delivery budget, not '
    'mine.\n'
    'Mara: Agreed.',

    'Tovin: I did not get the place on the course.\n'
    'Elsa: You were the strongest applicant they had.\n'
    'Tovin: The strongest without the fee, it turns out.\n'
    'Elsa: Then we find the fee. I have some put by.\n'
    'Tovin: I am not taking your savings for a maybe.\n'
    'Elsa: It is not a maybe if you go. Take it, and pay it back when you are '
    'teaching. That was always the plan.',
]

_DEFINITION_EXEMPLARS = [
    'interest: a charge paid for the use of borrowed money, set as a share of '
    'the amount borrowed over a period of time. Interest is owed on top of the '
    'original sum and grows the longer the sum goes unpaid.\n'
    'principal: the original amount of money borrowed or invested, apart from '
    'any interest. The principal is the figure on which interest is '
    'calculated.',

    'artery: a vessel that carries blood away from the heart to the rest of '
    'the body. An artery has thick, muscular walls that withstand the pressure '
    'of each heartbeat.\n'
    'vein: a vessel that carries blood back toward the heart. A vein has '
    'thinner walls than an artery and contains valves that keep the blood from '
    'flowing backward.',
]

_DESCRIPTION_EXEMPLARS = [
    'A hand plane is heavier at the front than most people expect, and the '
    'weight is deliberate. The blade sits at a fixed angle behind the mouth, '
    'its edge just proud of the sole, and the depth of cut is set by a screw '
    'beneath the handle. Turned one way the screw brings the edge lower and '
    'the shaving thickens; turned the other, the cut grows fine enough to see '
    'through.',

    'A market town keeps its trade in the square and its quiet in the streets '
    'behind it. The stalls nearest the road take the most custom and pay the '
    'most for the place; those in the corners are cheaper and slower. On '
    'market days the square is loud from dawn; on other days it is a car park, '
    'and only the worn stone under the stalls shows where the trade stands.',
]

_REASONING_EXEMPLARS = [
    'Bread rises because the yeast in the dough feeds on sugars and gives off '
    'gas as it does. The gas cannot escape the stretched, elastic network the '
    'flour forms when it is kneaded, so it collects in small pockets and '
    'pushes the dough outward. Heat from the oven makes the trapped gas expand '
    'faster still, and then sets the walls of each pocket firm before they can '
    'collapse. What began as a dense lump is left full of fixed holes, which '
    'is why well-made bread is light.',

    'A town gains more from repairing its old bridge than from building a '
    'second one, for three reasons. First, the repair costs less and can be '
    'done in stages, so the crossing is never fully closed. Second, one '
    'well-kept bridge is cheaper to maintain than two that each fall slowly '
    'into disrepair. Third, a new bridge draws building to the far bank and '
    'spreads the town thin before it can pay for the services that spread '
    'demands. The case for a second bridge rests on growth that may not come; '
    'the case for repair rests on a crossing that is already needed now.',
]

_EXEMPLAR_POOLS = {
    'prose': _PROSE_EXEMPLARS,
    'conversation': _CONVERSATION_EXEMPLARS,
    'definition': _DEFINITION_EXEMPLARS,
    'description': _DESCRIPTION_EXEMPLARS,
    'reasoning': _REASONING_EXEMPLARS,
}

_EXEMPLAR_INSTRUCTIONS = {
    'prose': 'Write a short story in which a clerk must choose between a lie '
             'that protects a colleague and a truth that protects himself.',
    'conversation': 'Write a terse exchange in which two colleagues settle how '
                    'to ship two orders.',
    'definition': 'Write dictionary entries defining two real financial terms.',
    'description': 'Write a factual description of how a hand plane works.',
    'reasoning': 'Explain, in order, why bread rises when it is baked.',
}

_PAIR_EXEMPLAR = (
    'How does a lock and key work?',
    'A pin-tumbler lock holds a row of small spring-loaded pins that cross the '
    'gap between the outer case and the inner plug, jamming the plug so it '
    'cannot turn. Each pin is cut into two pieces at a different height. When '
    'the right key is pushed in, its ridges lift every pin so that each cut '
    'lines up exactly with the gap between the case and the plug. With every '
    'cut aligned, nothing crosses the gap and the plug is free to turn and '
    'draw back the bolt. A wrong key lifts the pins to the wrong heights, so '
    'at least one still crosses the gap and the plug stays locked.',
)


def build_system_prompt():
    """Return the system prompt shared by every text type."""
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


def _prose_prompt(random_generator):
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    return (
        'Write %s in a %s tone: %s, drawn from %s, told from %s. The story '
        'must have a plot: a character who wants something, an obstacle or '
        'opposition, a turning point, and a definite outcome. Give the '
        'characters names. Be concrete and specific.' % (
            random_generator.choice(seeds.LENGTH_BANDS),
            random_generator.choice(seeds.TONES),
            random_generator.choice(seeds.STORY_SITUATIONS),
            domain,
            random_generator.choice(seeds.POINTS_OF_VIEW),
        )
    )


def _conversation_prompt(random_generator):
    name_a = seeds.invented_name(random_generator)
    name_b = seeds.invented_name(random_generator)
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    return (
        'Write a %s exchange of several turns, set in the world of %s, in '
        'which %s and %s %s. Each turn must do work: a proposal, an objection, '
        'a reason, or a concession, and the exchange must reach a definite '
        'outcome. Begin each turn with the speaker name and a colon.' % (
            random_generator.choice(seeds.TONES), domain, name_a, name_b,
            random_generator.choice(seeds.DIALOGUE_GOALS),
        )
    )


def _definition_prompt(random_generator):
    count = random_generator.choice([2, 2, 3])
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    return (
        'Write dictionary entries defining %d real terms from %s. Choose terms '
        'that belong together. Define each in genus-and-differentia form (a '
        'headword is a kind of thing that has some distinguishing property), '
        'accurately and consistently with the others.' % (count, domain)
    )


def _description_prompt(random_generator):
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    relation_kinds = random_generator.sample(
        seeds.RELATION_KINDS, random_generator.choice([2, 3])
    )
    return (
        'Write a %s factual description of a real thing, place, or process '
        'from %s, making its %s relations clear. Use plain declarative '
        'sentences, no narrative, and no fixed template. Name the subject '
        'plainly and be accurate.' % (
            random_generator.choice(seeds.TONES), domain,
            ', '.join(relation_kinds),
        )
    )


def _reasoning_prompt(random_generator):
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    return (
        'Take a real question, process, or claim from %s and %s. Keep the '
        'order strict: each sentence should follow from the ones before it, '
        'and the conclusion should rest on the steps given. Be accurate and '
        'concrete, not vague or figurative.' % (
            domain, random_generator.choice(seeds.REASONING_MODES),
        )
    )


_BUILDERS = {
    'prose': _prose_prompt,
    'conversation': _conversation_prompt,
    'definition': _definition_prompt,
    'description': _description_prompt,
    'reasoning': _reasoning_prompt,
}


def build_prompt(text_type, random_generator):
    """Return a user prompt for the given text type."""
    builder = _BUILDERS[text_type]
    return builder(random_generator) + _BEGIN_DIRECTLY


_PAIR_FORMAT = (
    '\n\nReturn it in exactly this format, with no extra commentary:\n'
    'USER: <the instruction>\n'
    'ASSISTANT: <the response>'
)

_PAIR_TASKS = [
    'The user asks the assistant to explain how or why something in %s works, '
    'and the assistant gives a clear, correctly ordered explanation.',
    'The user asks the assistant how to do a specific task in %s, and the '
    'assistant gives ordered, practical steps.',
    'The user asks the assistant to compare two real things in %s, and the '
    'assistant lays out the differences and reaches a judgement.',
    'The user asks the assistant what a real term in %s means, and the '
    'assistant defines it precisely and gives a short example.',
    'The user asks the assistant a factual question about %s, and the '
    'assistant answers directly and correctly.',
    'The user describes a real situation in %s and asks for advice, and the '
    'assistant gives specific, reasoned advice.',
    'The user gives the assistant a short passage about %s and asks for a '
    'summary, and the assistant summarizes it faithfully.',
    'The user gives the assistant an awkward sentence about %s and asks to '
    'rewrite it more clearly, and the assistant rewrites it.',
    'The user asks the assistant to list and organize things in %s, and the '
    'assistant gives an organized list with a short reason for the grouping.',
    'The user poses a small reasoning problem set in %s, and the assistant '
    'works through it in order to a definite answer.',
]


def build_pair_prompt(random_generator):
    """Return a prompt that yields one instruction and response pair."""
    task = random_generator.choice(_PAIR_TASKS)
    domain = random_generator.choice(seeds.SUBJECT_DOMAINS)
    instruction = task % domain
    return (
        'Create one instruction and response pair for a helpful, knowledgeable '
        'assistant. %s The instruction should read like something a real '
        'person would ask, and the response should be genuinely helpful and '
        'correct.%s' % (instruction, _PAIR_FORMAT)
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

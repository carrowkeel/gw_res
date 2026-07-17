"""Prompt construction for synthetic generation.

Every prompt is grounded in a program-generated fact set from slm.worldgen:
the program authors the logic (a consistent world fragment, or a puzzle whose
answer it derived by construction), and the LLM only writes it up in the
register the text type asks for. The same grounding machinery serves every
type, so a transfer puzzle can surface as a dialogue, a reasoning piece, or an
instruction pair, and a world fragment as a story or a description. The
rationale is that a small model can only learn patterns that are actually
there: text whose internal logic is loose or contradictory gives it nothing
stable to learn, so all generated text is anchored to facts that cannot
contradict each other, with correct answers supplied to the writer rather
than left for it to compute.

Referent stripping is relaxed: real facts, named entities, numbers, and
technical vocabulary are all allowed, and prompts are additionally anchored in
a sampled subject domain so surface content stays varied. Each text type also
carries a structural demand (plot for prose, worked order for reasoning, turns
that do work for conversation). Diversity comes from independent structural
axes (domain, tone, form, point of view, length, grounding kind) plus one
exemplar sampled from a rotating pool per request.
"""

from . import seeds, worldgen

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
    'darsel: the smallest unit of weight used at the market scales. Four '
    'darsels make one kevrin, so the darsel is the unit in which small '
    'goods are weighed out.\n'
    'kevrin: a unit of weight equal to four darsels. Heavier wares are '
    'quoted in kevrins, and three kevrins make one paldor.\n'
    'paldor: the largest unit of weight in common use, equal to three '
    'kevrins and therefore to twelve darsels. Cart loads are reckoned in '
    'paldors.',

    'mirel: the basic unit of cloth length, the width of a standard loom. '
    'Five mirels make one tovan, so short pieces are cut and sold by the '
    'mirel.\n'
    'tovan: a unit of cloth length equal to five mirels. A bolt is wound '
    'and priced by the tovan, and two tovans make one seldric.\n'
    'seldric: the largest cloth measure, equal to two tovans and therefore '
    'to ten mirels. Whole consignments are invoiced in seldrics.',
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
    'Halden has fourteen crates of apples and Verin has six. Halden gives '
    'five of his crates to Verin. The question is how many crates Verin has '
    'now. Verin began with six crates, and the five that Halden handed over '
    'are added to them. Six plus five is eleven. Verin now has eleven crates '
    'of apples.',

    'One belric is worth three sarns, and one covan is worth four belrics. '
    'The question is how many sarns one covan is worth. A covan is four '
    'belrics, and each of those belrics is three sarns. Four times three is '
    'twelve. One covan is therefore worth twelve sarns.',
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
    'definition': 'Write dictionary entries defining three invented units of '
                  'measure with exact conversion relations.',
    'description': 'Write a factual description of how a hand plane works.',
    'reasoning': 'State the facts of a small trade, pose the question, and '
                 'work in order to the answer.',
}

_PAIR_EXEMPLAR = (
    'Merla has nine baskets of eggs and Dorn has four. Merla gives three '
    'baskets to Dorn. How many baskets of eggs does Dorn have now?',
    'Dorn started with four baskets and received three more from Merla, so '
    'Dorn now has seven baskets of eggs.',
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


def _facts_clause(grounding):
    return (
        ' The text must stay consistent with every one of these facts, '
        'weaving them in naturally rather than listing or enumerating them: '
        '%s' % ' '.join(grounding['facts'])
    )


def _answer_clause(grounding):
    derivation = ''
    if grounding['derivation']:
        derivation = ' (derivation: %s)' % '; '.join(grounding['derivation'])
    return (
        ' The correct answer to the question "%s" is %s%s; the text must '
        'work toward and state that answer, never a different one.'
        % (grounding['question'], grounding['answer'], derivation)
    )


def _prose_prompt(random_generator):
    grounding = worldgen.sample_grounding(random_generator, 'fragment')
    return (
        'Write %s in a %s tone: %s, set among %s, told from %s. The story '
        'must have a plot: a character who wants something, an obstacle or '
        'opposition, a turning point, and a definite outcome. Use the people '
        'and places named in the facts as the characters and setting.%s' % (
            random_generator.choice(seeds.LENGTH_BANDS),
            random_generator.choice(seeds.TONES),
            random_generator.choice(seeds.STORY_SITUATIONS),
            grounding['domain_label'],
            random_generator.choice(seeds.POINTS_OF_VIEW),
            _facts_clause(grounding),
        )
    )


def _conversation_prompt(random_generator):
    grounding = worldgen.sample_grounding(random_generator)
    if grounding['kind'] == 'fragment':
        return (
            'Write a %s exchange of several turns between two of the people '
            'named in the facts, in which they %s. Each turn must do work: a '
            'proposal, an objection, a reason, or a concession, and the '
            'exchange must reach a definite outcome. Begin each turn with the '
            'speaker name and a colon.%s' % (
                random_generator.choice(seeds.TONES),
                random_generator.choice(seeds.DIALOGUE_GOALS),
                _facts_clause(grounding),
            )
        )
    name_a = seeds.invented_name(random_generator)
    name_b = seeds.invented_name(random_generator)
    return (
        'Write a %s exchange of several turns in which %s and %s work out the '
        'question "%s" together from the facts, one proposing steps and the '
        'other checking them, until they agree on the answer. Begin each turn '
        'with the speaker name and a colon.%s%s' % (
            random_generator.choice(seeds.TONES), name_a, name_b,
            grounding['question'], _facts_clause(grounding),
            _answer_clause(grounding),
        )
    )


def _definition_prompt(random_generator):
    grounding = worldgen.sample_grounding(random_generator, 'ratio')
    units = grounding['units']
    return (
        'Write dictionary entries defining the units of measure %s, %s, and '
        '%s, in genus-and-differentia form (a headword is a kind of thing '
        'that has some distinguishing property). The entries must state the '
        'exact conversion relations between the units and agree with each '
        'other precisely.%s' % (
            units[0], units[1], units[2], _facts_clause(grounding),
        )
    )


def _description_prompt(random_generator):
    grounding = worldgen.sample_grounding(random_generator, 'fragment')
    relation_kinds = random_generator.sample(
        seeds.RELATION_KINDS, random_generator.choice([2, 3])
    )
    return (
        'Write a %s factual description of the people, places, and things '
        'named in the facts, making their %s relations clear. Use plain '
        'declarative sentences, no narrative, and no fixed template.%s' % (
            random_generator.choice(seeds.TONES),
            ', '.join(relation_kinds),
            _facts_clause(grounding),
        )
    )


def _reasoning_prompt(random_generator):
    kind = random_generator.choice(['transfer', 'ratio', 'order'])
    grounding = worldgen.sample_grounding(random_generator, kind)
    return (
        'State the facts below plainly, pose the question "%s", and %s. Keep '
        'the order strict: each sentence should follow from the ones before '
        'it, and the conclusion should rest on the steps given.%s%s' % (
            grounding['question'],
            random_generator.choice(seeds.REASONING_MODES),
            _facts_clause(grounding),
            _answer_clause(grounding),
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

def build_pair_prompt(random_generator):
    """Return (prompt, task kind) for one grounded instruction-response pair.

    The user turn must contain the grounding facts and the question, so the
    pair is answerable from its own context, and the assistant's answer is
    fixed by the program-derived ground truth, so the pair cannot teach a
    wrong conclusion. Half the pairs ask for the bare answer with brief
    working; half ask the assistant to explain the solution step by step.
    The kinds range over the full task mix, including multihop composition
    and notstated pairs whose correct response is to say the facts do not
    contain the answer, so the model learns the boundary of its context
    instead of fabricating an answer in perfect form.
    """
    grounding = worldgen.sample_pair_grounding(random_generator)
    kind = grounding['task_kind']
    if kind == 'notstated':
        return (
            'Create one instruction and response pair for a helpful '
            'assistant. The user message must state all of these facts in '
            'its own words and then ask: "%s" The facts: %s The facts do '
            'not contain the answer to that question, and the response must '
            'say exactly that, in one short sentence such as "%s", without '
            'inventing an answer.%s' % (
                grounding['question'], ' '.join(grounding['facts']),
                grounding['answer'], _PAIR_FORMAT,
            )
        ), kind
    if random_generator.random() < 0.5:
        style = (
            'The assistant answers directly, with at most one sentence of '
            'working.'
        )
    else:
        style = (
            'The assistant explains the solution step by step, each step '
            'following from the facts, before stating the answer.'
        )
    return (
        'Create one instruction and response pair for a helpful assistant. '
        'The user message must state all of these facts in its own words and '
        'then ask: "%s" The facts: %s %s The correct answer is %s%s; the '
        'response must reach and state exactly that answer.%s' % (
            grounding['question'], ' '.join(grounding['facts']), style,
            grounding['answer'],
            (' (derivation: %s)' % '; '.join(grounding['derivation'])
             if grounding['derivation'] else ''),
            _PAIR_FORMAT,
        )
    ), kind


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

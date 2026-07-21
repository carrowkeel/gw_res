"""Programmatic world-state generation: consistent facts, documents, tasks.

First increment of the program-as-author mechanism described in the intent
graph. A small world of people, places, and objects is sampled with attributes
and relations that are consistent by construction (comparisons come from total
rank orders, locations and ownership are functions), then verbalized through
varied sentence templates. Because the program, not an LLM, holds the state,
it can emit two things no LLM author can guarantee: documents whose facts
never contradict each other, and questions whose exact answers are known by
construction.

The binding tasks are the evaluation half: a context paragraph states facts
about novel invented entities, and a question asks for one of them back, so a
model is scored on whether it can bind and retrieve information given in
context, with exact-match scoring and no judge. The document generator is the
data half, usable to mix consistency-bearing text into the corpus.

    python -m slm.worldgen --seed 7
"""

import argparse
import random

from . import seeds

QUALIFIERS = [
    'gray', 'black', 'white', 'blue', 'red', 'green', 'silver', 'yellow',
    'orange', 'brown',
]

# Ordinary modern first names, mixed with invented ones when sampling people,
# so surfaces read as everyday life rather than an invented archaic world.
COMMON_NAMES = [
    'Nora', 'Sam', 'Elena', 'Marcus', 'Priya', 'Omar', 'Lily', 'Felix',
    'Maya', 'Jonas', 'Clara', 'Ravi', 'Anna', 'Leo', 'Sofia', 'Tariq',
    'Ines', 'Dana', 'Hugo', 'Mira', 'Pavel', 'Alma', 'Kenji', 'Rosa',
    'Dev', 'Greta', 'Idris', 'Nina', 'Tomas', 'Zara',
]

# Each domain supplies its own places, objects, and countable goods so the
# sampled worlds range over many registers. The registers are deliberately
# modern and real-world (offices, clinics, depots), because the absence of
# real-world referents is a deferred objective: the program supplies the
# logic, and the surface should read like the everyday world the generator
# LLM writes best, not an invented archaic one.
DOMAINS = {
    'office': {
        'label': 'offices and everyday work',
        'places': [
            'head office', 'records room', 'meeting room', 'print room',
            'mail room', 'staff kitchen', 'supply closet', 'reception desk',
            'archive room',
        ],
        'objects': [
            'printer', 'filing cabinet', 'whiteboard', 'projector',
            'coffee machine', 'shredder', 'standing desk', 'notice board',
            'water cooler', 'photocopier',
        ],
        'goods': [
            'reams of paper', 'boxes of pens', 'folders of invoices',
            'packs of envelopes', 'rolls of tape', 'boxes of staples',
        ],
    },
    'clinic': {
        'label': 'clinics and care',
        'places': [
            'health clinic', 'pharmacy', 'waiting room', 'therapy room',
            'dental surgery', 'care home', 'ambulance station', 'medical lab',
        ],
        'objects': [
            'wheelchair', 'supply cart', 'examination table',
            'medicine cabinet', 'sterilizer', 'stretcher', 'oxygen tank',
            'first-aid kit',
        ],
        'goods': [
            'boxes of gloves', 'bottles of sanitizer', 'rolls of bandage',
            'packs of syringes', 'boxes of masks', 'bottles of vitamins',
        ],
    },
    'school': {
        'label': 'schools and learning',
        'places': [
            'primary school', 'library', 'science lab', 'gymnasium',
            'music room', 'computer room', 'cafeteria', 'art studio',
            'lecture hall',
        ],
        'objects': [
            'bookshelf', 'globe', 'microscope', 'piano', 'easel',
            'chalkboard', 'laptop cart', 'display case', 'telescope',
        ],
        'goods': [
            'boxes of chalk', 'stacks of textbooks', 'packs of notebooks',
            'jars of paint', 'boxes of markers', 'crates of sports gear',
        ],
    },
    'cafe': {
        'label': 'cafes and kitchens',
        'places': [
            'corner cafe', 'bakery', 'pizzeria', 'food market', 'juice bar',
            'canteen', 'ice cream parlor', 'tea room', 'delicatessen',
        ],
        'objects': [
            'espresso machine', 'pastry case', 'blender', 'menu board',
            'cash register', 'bread oven', 'dough mixer', 'refrigerator',
            'serving trolley',
        ],
        'goods': [
            'bags of coffee beans', 'trays of pastries', 'crates of oranges',
            'cartons of milk', 'loaves of bread', 'jars of jam',
            'boxes of tea',
        ],
    },
    'depot': {
        'label': 'warehouses and deliveries',
        'places': [
            'warehouse', 'loading dock', 'distribution center',
            'storage yard', 'freight office', 'packing hall',
            'container depot', 'repair garage',
        ],
        'objects': [
            'forklift', 'delivery van', 'pallet jack', 'conveyor belt',
            'hand truck', 'weighing scale', 'shipping container',
            'packing table',
        ],
        'goods': [
            'pallets of boxes', 'rolls of packing tape', 'crates of parts',
            'stacks of pallets', 'boxes of labels', 'drums of oil',
        ],
    },
    'studio': {
        'label': 'studios and media',
        'places': [
            'recording studio', 'radio station', 'theater', 'gallery',
            'print shop', 'photo studio', 'rehearsal room', 'editing suite',
        ],
        'objects': [
            'camera', 'mixing desk', 'spotlight', 'drum kit',
            'microphone stand', 'poster rack', 'amplifier', 'tripod',
        ],
        'goods': [
            'reels of cable', 'boxes of records', 'rolls of film',
            'stacks of posters', 'cases of stage bulbs', 'crates of props',
        ],
    },
    'transit': {
        'label': 'stations and travel',
        'places': [
            'bus depot', 'train station', 'airport terminal',
            'ferry terminal', 'taxi stand', 'service station',
            'parking garage', 'ticket office',
        ],
        'objects': [
            'luggage cart', 'ticket machine', 'departure board', 'fuel pump',
            'bicycle rack', 'vending machine', 'information kiosk',
            'baggage scanner',
        ],
        'goods': [
            'crates of luggage tags', 'boxes of timetables',
            'cans of de-icer', 'cartons of snacks', 'bundles of maps',
            'boxes of tickets',
        ],
    },
    'sports': {
        'label': 'sports and leisure',
        'places': [
            'sports hall', 'swimming pool', 'football ground',
            'climbing gym', 'tennis club', 'rowing club', 'skate park',
            'community center',
        ],
        'objects': [
            'rowing machine', 'scoreboard', 'trophy cabinet', 'ball cart',
            'exercise bike', 'climbing rope', 'kayak', 'table tennis table',
        ],
        'goods': [
            'bags of tennis balls', 'crates of water bottles',
            'stacks of towels', 'boxes of shuttlecocks', 'sets of jerseys',
            'bags of chalk powder',
        ],
    },
}

ORDER_DIMENSIONS = [
    ('older', 'younger', 'the oldest'),
    ('taller', 'shorter', 'the tallest'),
    ('heavier', 'lighter', 'the heaviest'),
    ('faster', 'slower', 'the fastest'),
    ('stronger', 'weaker', 'the strongest'),
]

NOT_STATED_ANSWER = 'That is not stated.'

_TEMPLATES = {
    'lives': [
        '%(person)s lives at %(place)s.',
        'The home of %(person)s is %(place)s.',
        '%(person)s has an apartment at %(place)s.',
    ],
    'works': [
        '%(person)s works at %(place)s.',
        '%(person)s spends the working day at %(place)s.',
        '%(person)s is on the payroll at %(place)s.',
    ],
    'owns': [
        'The %(object)s belongs to %(person)s.',
        '%(person)s owns the %(object)s.',
        'The %(object)s is the property of %(person)s.',
    ],
    'kept': [
        'The %(object)s is kept at %(place)s.',
        'The %(object)s stands at %(place)s.',
        'At %(place)s stands the %(object)s.',
    ],
    'older': [
        '%(first)s is older than %(second)s.',
        '%(second)s is younger than %(first)s.',
    ],
    'larger': [
        'The %(first)s is larger than the %(second)s.',
        'The %(second)s is smaller than the %(first)s.',
    ],
    'age': [
        '%(person)s is %(years)d years old.',
        '%(person)s turned %(years)d this year.',
        'At %(years)d, %(person)s is a familiar face there.',
    ],
}


def _sample_unique(random_generator, pool_a, pool_b, count):
    pairs = [(a, b) for a in pool_a for b in pool_b]
    return random_generator.sample(pairs, count)


def sample_world(random_generator, people=3, places=3, objects=4, domain=None):
    """Return a consistent small world of people, places, and objects.

    People carry distinct age ranks with consistent absolute ages in years, and
    objects distinct size ranks, so every pairwise comparison has a unique,
    consistent answer. Residence, workplace, ownership, and storage are
    functions, so every retrieval question has a unique answer. The vocabulary
    is drawn from one sampled domain so worlds range across many registers.
    """
    if domain is None:
        domain = random_generator.choice(sorted(DOMAINS))
    vocabulary = DOMAINS[domain]
    person_names = []
    while len(person_names) < people:
        if random_generator.random() < 0.5:
            name = random_generator.choice(COMMON_NAMES)
        else:
            name = seeds.invented_name(random_generator)
        if name not in person_names:
            person_names.append(name)
    place_list = [
        '%s %s' % (seeds.invented_name(random_generator), kind)
        for kind in random_generator.sample(vocabulary['places'], places)
    ]
    object_list = [
        '%s %s' % (qualifier, kind)
        for qualifier, kind in _sample_unique(
            random_generator, QUALIFIERS, vocabulary['objects'], objects
        )
    ]
    age_order = list(person_names)
    random_generator.shuffle(age_order)
    # age_rank is ordered youngest-first (see _fragment's elder test), so the
    # i-th ranked person gets the i-th smallest age and years agree with ranks.
    ages = sorted(random_generator.sample(range(18, 75), people))
    size_order = list(object_list)
    random_generator.shuffle(size_order)
    world = {
        'domain': domain,
        'people': person_names,
        'places': place_list,
        'objects': object_list,
        'lives': {p: random_generator.choice(place_list) for p in person_names},
        'works': {p: random_generator.choice(place_list) for p in person_names},
        'owner': {},
        'kept': {o: random_generator.choice(place_list) for o in object_list},
        'age_rank': {p: i for i, p in enumerate(age_order)},
        'age_years': {p: ages[i] for i, p in enumerate(age_order)},
        'size_rank': {o: i for i, o in enumerate(size_order)},
    }
    for index, item in enumerate(object_list):
        world['owner'][item] = person_names[index % len(person_names)]
    return world


def _render(fact_type, random_generator, **slots):
    template = random_generator.choice(_TEMPLATES[fact_type])
    return template % slots


def _fragment(world, random_generator):
    """Return (sentences, focus facts) for one focus person's neighborhood."""
    person = random_generator.choice(world['people'])
    owned = [o for o, owner in world['owner'].items() if owner == person]
    extra_pool = [o for o in world['objects'] if o not in owned]
    fragment_objects = owned[:2]
    while len(fragment_objects) < 2 and extra_pool:
        fragment_objects.append(extra_pool.pop(0))
    other_people = [p for p in world['people'] if p != person]
    other = random_generator.choice(other_people)

    sentences = [
        _render('lives', random_generator, person=person,
                place=world['lives'][person]),
        _render('works', random_generator, person=person,
                place=world['works'][person]),
    ]
    for item in fragment_objects:
        sentences.append(
            _render('owns', random_generator, object=item,
                    person=world['owner'][item])
        )
        sentences.append(
            _render('kept', random_generator, object=item,
                    place=world['kept'][item])
        )
    if world['age_rank'][person] < world['age_rank'][other]:
        age_pair = (other, person)
    else:
        age_pair = (person, other)
    # Half the fragments state the comparison directly; half state each
    # person's age in years, so the comparison must be derived from numbers.
    ages_stated = random_generator.random() < 0.5
    if ages_stated:
        for who in (person, other):
            sentences.append(
                _render('age', random_generator, person=who,
                        years=world['age_years'][who])
            )
    else:
        sentences.append(
            _render('older', random_generator, first=age_pair[0],
                    second=age_pair[1])
        )
    size_pair = sorted(
        fragment_objects[:2], key=lambda o: world['size_rank'][o], reverse=True
    )
    if len(size_pair) == 2:
        sentences.append(
            _render('larger', random_generator, first=size_pair[0],
                    second=size_pair[1])
        )
    random_generator.shuffle(sentences)
    facts = {
        'person': person,
        'other': other,
        'objects': fragment_objects,
        'age_pair': age_pair,
        'ages_stated': ages_stated,
        'size_pair': size_pair if len(size_pair) == 2 else None,
    }
    return sentences, facts


QUESTION_CATEGORIES = ['retrieval', 'comparison', 'multihop', 'notstated']


def _multihop_items(world, facts):
    """Objects whose owner is the focus person, so a two-fact chain is stated."""
    return [
        item for item in facts['objects']
        if world['owner'][item] == facts['person']
    ]


def _make_question(world, facts, random_generator, category=None):
    """Return (question, answer, distractor or None, category).

    Every question is answerable from the fragment alone, except category
    notstated, whose gold answer is that the fragment does not say. retrieval
    reads one stated fact back; comparison resolves a stated or number-derived
    comparison; multihop composes two stated facts (an object's owner plus
    that owner's stated residence or workplace).
    """
    person = facts['person']
    hop_items = _multihop_items(world, facts)
    if category is None:
        available = ['retrieval', 'comparison']
        if hop_items:
            available.append('multihop')
        category = random_generator.choice(available)

    if category == 'retrieval':
        choices = ['where_lives', 'where_works', 'who_owns', 'where_kept']
        if facts['ages_stated']:
            choices.append('how_old')
        kind = random_generator.choice(choices)
        if kind == 'where_lives':
            return ('Where does %s live?' % person,
                    world['lives'][person], None, category)
        if kind == 'where_works':
            return ('Where does %s work?' % person,
                    world['works'][person], None, category)
        if kind == 'who_owns':
            item = random_generator.choice(facts['objects'])
            return ('Who owns the %s?' % item,
                    world['owner'][item], None, category)
        if kind == 'where_kept':
            item = random_generator.choice(facts['objects'])
            return ('Where is the %s kept?' % item,
                    world['kept'][item], None, category)
        return ('How old is %s?' % person,
                str(world['age_years'][person]), None, category)

    if category == 'comparison':
        if facts['size_pair'] and random_generator.random() < 0.5:
            larger, smaller = facts['size_pair']
            pair = [larger, smaller]
            random_generator.shuffle(pair)
            question = 'Which is larger, the %s or the %s?' % (
                pair[0], pair[1]
            )
            return question, larger, smaller, category
        elder, junior = facts['age_pair']
        pair = [elder, junior]
        random_generator.shuffle(pair)
        question = 'Who is older, %s or %s?' % (pair[0], pair[1])
        return question, elder, junior, category

    if category == 'multihop':
        item = random_generator.choice(hop_items)
        if random_generator.random() < 0.5:
            return ('Where does the owner of the %s live?' % item,
                    world['lives'][person], None, category)
        return ('Where does the owner of the %s work?' % item,
                world['works'][person], None, category)

    # notstated: ask for a fact the fragment genuinely does not state, about
    # an entity it mentions (or an object it never mentions at all).
    choices = ['other_lives', 'unmentioned_object']
    if not facts['ages_stated']:
        choices.append('how_old')
    kind = random_generator.choice(choices)
    if kind == 'other_lives':
        question = 'Where does %s live?' % facts['other']
    elif kind == 'unmentioned_object':
        outside = [
            item for item in world['objects']
            if item not in facts['objects']
        ]
        if not outside:
            question = 'Where does %s live?' % facts['other']
        else:
            item = random_generator.choice(outside)
            question = 'Where is the %s kept?' % item
    else:
        question = 'How old is %s?' % person
    return question, NOT_STATED_ANSWER, None, category


def binding_tasks(seed, count, categories=None):
    """Return in-context binding tasks with exact answers known by construction.

    Each task is a dict with context (a paragraph of consistent facts about
    novel invented entities), question, answer, kind (the question category),
    and an optional distractor (the wrong candidate in a two-way comparison).
    Tasks cycle through the categories so each sub-score gets an even share:
    retrieval and comparison read the context back, multihop composes two
    stated facts, and notstated asks for a fact the context does not contain,
    so it measures whether the model fabricates or declines.
    """
    random_generator = random.Random(seed)
    categories = categories or QUESTION_CATEGORIES
    tasks = []
    for index in range(count):
        category = categories[index % len(categories)]
        while True:
            world = sample_world(random_generator)
            sentences, facts = _fragment(world, random_generator)
            if category == 'multihop' and not _multihop_items(world, facts):
                continue
            break
        question, answer, distractor, kind = _make_question(
            world, facts, random_generator, category
        )
        tasks.append({
            'context': ' '.join(sentences),
            'question': question,
            'answer': answer,
            'distractor': distractor,
            'kind': kind,
        })
    return tasks


def world_documents(seed, count):
    """Return consistency-bearing documents rendered from sampled worlds."""
    random_generator = random.Random(seed)
    documents = []
    for _ in range(count):
        world = sample_world(random_generator)
        sentences, _ = _fragment(world, random_generator)
        documents.append(' '.join(sentences))
    return documents


def _invented_unit(random_generator, taken):
    while True:
        unit = seeds.invented_name(random_generator).lower()
        if unit not in taken:
            taken.add(unit)
            return unit


def _transfer_puzzle(random_generator):
    giver = seeds.invented_name(random_generator)
    receiver = seeds.invented_name(random_generator)
    while receiver == giver:
        receiver = seeds.invented_name(random_generator)
    domain = random_generator.choice(sorted(DOMAINS))
    goods = random_generator.choice(DOMAINS[domain]['goods'])
    start_giver = random_generator.randint(8, 30)
    start_receiver = random_generator.randint(2, 15)
    given = random_generator.randint(2, start_giver - 2)
    facts = [
        '%s has %d %s.' % (giver, start_giver, goods),
        '%s has %d %s.' % (receiver, start_receiver, goods),
        '%s gives %d %s to %s.' % (giver, given, goods, receiver),
    ]
    random_generator.shuffle(facts)
    form = random_generator.choice(['left', 'received', 'total'])
    if form == 'left':
        question = 'How many %s does %s have now?' % (goods, giver)
        answer = start_giver - given
        derivation = ['%d - %d = %d' % (start_giver, given, answer)]
    elif form == 'received':
        question = 'How many %s does %s have now?' % (goods, receiver)
        answer = start_receiver + given
        derivation = ['%d + %d = %d' % (start_receiver, given, answer)]
    else:
        question = 'How many %s do they have between them?' % goods
        answer = start_giver + start_receiver
        derivation = [
            'the transfer does not change the total',
            '%d + %d = %d' % (start_giver, start_receiver, answer),
        ]
    return {
        'kind': 'transfer', 'facts': facts, 'question': question,
        'answer': str(answer), 'derivation': derivation,
        'task_kind': 'transfer',
    }


def _ratio_puzzle(random_generator):
    taken = set()
    small = _invented_unit(random_generator, taken)
    middle = _invented_unit(random_generator, taken)
    large = _invented_unit(random_generator, taken)
    first_ratio = random_generator.randint(2, 8)
    second_ratio = random_generator.randint(2, 6)
    quantity = random_generator.randint(2, 6)
    facts = [
        'One %s is worth %d %ss.' % (middle, first_ratio, small),
        'One %s is worth %d %ss.' % (large, second_ratio, middle),
    ]
    random_generator.shuffle(facts)
    if random_generator.random() < 0.5:
        question = 'How many %ss is one %s worth?' % (small, large)
        answer = first_ratio * second_ratio
        derivation = ['%d * %d = %d' % (second_ratio, first_ratio, answer)]
    else:
        question = 'How many %ss are %d %ss worth?' % (small, quantity, large)
        answer = first_ratio * second_ratio * quantity
        derivation = [
            'one %s is %d * %d = %d %ss' % (
                large, second_ratio, first_ratio,
                second_ratio * first_ratio, small,
            ),
            '%d * %d = %d' % (quantity, second_ratio * first_ratio, answer),
        ]
    return {
        'kind': 'ratio', 'facts': facts, 'question': question,
        'answer': str(answer), 'derivation': derivation,
        'task_kind': 'ratio',
        'units': [small, middle, large],
        'ratios': [first_ratio, second_ratio],
    }


def _order_puzzle(random_generator):
    names = []
    while len(names) < 4:
        name = seeds.invented_name(random_generator)
        if name not in names:
            names.append(name)
    comparative, inverse, superlative = random_generator.choice(
        ORDER_DIMENSIONS
    )
    order = list(names)
    random_generator.shuffle(order)
    facts = []
    for upper, lower in zip(order, order[1:]):
        if random_generator.random() < 0.5:
            facts.append('%s is %s than %s.' % (upper, comparative, lower))
        else:
            facts.append('%s is %s than %s.' % (lower, inverse, upper))
    random_generator.shuffle(facts)
    if random_generator.random() < 0.5:
        question = 'Who is %s?' % superlative
        answer, distractor = order[0], None
        derivation = [' > '.join(order)]
    else:
        # A transitive pairwise question: the two names are never adjacent in
        # the chain, so no single stated fact answers it and at least one
        # intermediate step must be composed.
        first = random_generator.randrange(0, len(order) - 2)
        second = random_generator.randrange(first + 2, len(order))
        pair = [order[first], order[second]]
        random_generator.shuffle(pair)
        question = 'Who is %s, %s or %s?' % (comparative, pair[0], pair[1])
        answer, distractor = order[first], order[second]
        derivation = [' > '.join(order[first:second + 1])]
    return {
        'kind': 'order', 'facts': facts, 'question': question,
        'answer': answer, 'derivation': derivation,
        'task_kind': 'order', 'distractor': distractor,
    }


def _fragment_grounding(random_generator, category=None):
    world = sample_world(random_generator, people=4, places=4, objects=6)
    sentences, facts = _fragment(world, random_generator)
    if category == 'multihop' and not _multihop_items(world, facts):
        category = 'retrieval'
    question, answer, distractor, question_category = _make_question(
        world, facts, random_generator, category
    )
    return {
        'kind': 'fragment', 'facts': sentences, 'question': question,
        'answer': answer, 'derivation': None, 'distractor': distractor,
        'task_kind': question_category,
        'domain': world['domain'],
        'domain_label': DOMAINS[world['domain']]['label'],
    }


_PUZZLE_KINDS = {
    'transfer': _transfer_puzzle,
    'ratio': _ratio_puzzle,
    'order': _order_puzzle,
}


def sample_grounding(random_generator, kind=None, category=None):
    """Return one program-generated grounding for the LLM writer.

    A grounding is a set of facts that are consistent by construction, plus a
    question whose answer the program derived, so the LLM can be asked to
    write text (in any register) that stays consistent with the facts and,
    where it works a problem, reaches the correct answer without having to
    solve anything itself. Kinds: fragment (a small relational world),
    transfer (countable goods arithmetic), ratio (invented units with exact
    conversion factors), order (a comparison chain with inverted surfaces and
    transitive questions). For fragments, category picks the question class
    (retrieval, comparison, multihop, notstated); left unset, the question
    ranges over the answerable classes only.
    """
    if kind is None:
        kind = random_generator.choice(
            ['fragment', 'fragment', 'transfer', 'ratio', 'order']
        )
    if kind == 'fragment':
        return _fragment_grounding(random_generator, category)
    return _PUZZLE_KINDS[kind](random_generator)


# Mix for instruction pairs: answerable fragment classes and puzzles carry
# most of the weight, multihop trains two-fact composition, and a notstated
# share teaches the model to decline when the facts do not contain the answer
# instead of fabricating one in perfect form.
PAIR_KIND_MIX = (
    [('fragment', 'retrieval')] * 3
    + [('fragment', 'comparison')] * 2
    + [('fragment', 'multihop')] * 2
    + [('fragment', 'notstated')] * 1
    + [('transfer', None)] * 2
    + [('ratio', None)] * 2
    + [('order', None)] * 2
)


def sample_pair_grounding(random_generator):
    """Return one grounding for an instruction pair, over the full task mix."""
    kind, category = random_generator.choice(PAIR_KIND_MIX)
    return sample_grounding(random_generator, kind, category)


_DECLINE_MARKERS = ['not stated', 'not given', 'does not say', 'not mentioned']


def _region_score(task, region):
    gold = task['answer'].lower()
    gold_position = region.find(gold)
    if gold_position == -1:
        return 0.0
    distractor = task.get('distractor')
    if distractor:
        distractor_position = region.find(distractor.lower())
        if distractor_position != -1 and distractor_position < gold_position:
            return 0.0
    return 1.0


def score_binding_answer(task, output):
    """Return 1.0 if the model output names the gold answer, else 0.0.

    The gold answer must lead either the head of the output (a direct answer)
    or its tail (the conclusion of a step-by-step derivation, the style half
    the training pairs use, where the answer arrives last after a restatement
    of the facts); in a two-way comparison the wrong candidate must not appear
    before the gold within the scored region. For a notstated task the gold
    behavior is declining: any recognized decline phrasing scores, and a
    specific fabricated answer does not.
    """
    text = ' '.join(output.strip().lower().split())
    if task['answer'] == NOT_STATED_ANSWER:
        return 1.0 if any(marker in text for marker in _DECLINE_MARKERS) else 0.0
    head = text[:120]
    # The conclusion region is the final sentence: a wider window would catch
    # the last derivation step, where a comparison restates the facts with the
    # wrong candidate leading.
    sentences = [part for part in text.split('.') if part.strip()]
    conclusion = sentences[-1] if sentences else ''
    return max(_region_score(task, head), _region_score(task, conclusion))


def main():
    parser = argparse.ArgumentParser(description='Preview generated worlds')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--count', type=int, default=3)
    arguments = parser.parse_args()
    for task in binding_tasks(arguments.seed, arguments.count):
        print('CONTEXT: %s' % task['context'])
        print('Q: %s' % task['question'])
        print('A: %s' % task['answer'])
        print()


if __name__ == '__main__':
    main()

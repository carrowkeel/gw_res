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

PLACE_KINDS = [
    'mill', 'forge', 'market hall', 'granary', 'boathouse', 'weaving shed',
    'brewhouse', 'stable', 'printworks', 'bakery',
]

OBJECT_KINDS = [
    'kettle', 'chest', 'ladder', 'anvil', 'loom', 'cart', 'barrel',
    'lantern', 'plough', 'bench', 'clock', 'press',
]

MATERIALS = [
    'copper', 'oak', 'iron', 'ash', 'tin', 'birch', 'leather', 'stone',
    'brass', 'elm', 'pine', 'steel',
]

_TEMPLATES = {
    'lives': [
        '%(person)s lives at %(place)s.',
        'The home of %(person)s is %(place)s.',
        '%(person)s has rooms at %(place)s.',
    ],
    'works': [
        '%(person)s works at %(place)s.',
        '%(person)s spends the working day at %(place)s.',
        'The wages of %(person)s are paid at %(place)s.',
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
}


def _sample_unique(random_generator, pool_a, pool_b, count):
    pairs = [(a, b) for a in pool_a for b in pool_b]
    return random_generator.sample(pairs, count)


def sample_world(random_generator, people=3, places=3, objects=4):
    """Return a consistent small world of people, places, and objects.

    People carry distinct age ranks and objects distinct size ranks, so every
    pairwise comparison has a unique, consistent answer. Residence, workplace,
    ownership, and storage are functions, so every retrieval question has a
    unique answer.
    """
    person_names = []
    while len(person_names) < people:
        name = seeds.invented_name(random_generator)
        if name not in person_names:
            person_names.append(name)
    place_list = [
        '%s %s' % (seeds.invented_name(random_generator), kind)
        for kind in random_generator.sample(PLACE_KINDS, places)
    ]
    object_list = [
        '%s %s' % (material, kind)
        for material, kind in _sample_unique(
            random_generator, MATERIALS, OBJECT_KINDS, objects
        )
    ]
    age_order = list(person_names)
    random_generator.shuffle(age_order)
    size_order = list(object_list)
    random_generator.shuffle(size_order)
    world = {
        'people': person_names,
        'places': place_list,
        'objects': object_list,
        'lives': {p: random_generator.choice(place_list) for p in person_names},
        'works': {p: random_generator.choice(place_list) for p in person_names},
        'owner': {},
        'kept': {o: random_generator.choice(place_list) for o in object_list},
        'age_rank': {p: i for i, p in enumerate(age_order)},
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
        'size_pair': size_pair if len(size_pair) == 2 else None,
    }
    return sentences, facts


def _make_question(world, facts, random_generator):
    """Return (question, answer, distractor or None) answerable from the fragment."""
    person = facts['person']
    choices = ['where_lives', 'who_owns', 'where_kept', 'older']
    if facts['size_pair']:
        choices.append('larger')
    kind = random_generator.choice(choices)
    if kind == 'where_lives':
        return 'Where does %s live?' % person, world['lives'][person], None
    if kind == 'who_owns':
        item = random_generator.choice(facts['objects'])
        return 'Who owns the %s?' % item, world['owner'][item], None
    if kind == 'where_kept':
        item = random_generator.choice(facts['objects'])
        return 'Where is the %s kept?' % item, world['kept'][item], None
    if kind == 'older':
        elder, junior = facts['age_pair']
        pair = [elder, junior]
        random_generator.shuffle(pair)
        question = 'Who is older, %s or %s?' % (pair[0], pair[1])
        return question, elder, junior
    larger, smaller = facts['size_pair']
    pair = [larger, smaller]
    random_generator.shuffle(pair)
    question = 'Which is larger, the %s or the %s?' % (pair[0], pair[1])
    return question, larger, smaller


def binding_tasks(seed, count):
    """Return in-context binding tasks with exact answers known by construction.

    Each task is a dict with context (a paragraph of consistent facts about
    novel invented entities), question, answer, and an optional distractor
    (the wrong candidate in a two-way comparison). A model that binds the
    context correctly can answer; nothing is answerable from world knowledge.
    """
    random_generator = random.Random(seed)
    tasks = []
    for _ in range(count):
        world = sample_world(random_generator)
        sentences, facts = _fragment(world, random_generator)
        question, answer, distractor = _make_question(
            world, facts, random_generator
        )
        tasks.append({
            'context': ' '.join(sentences),
            'question': question,
            'answer': answer,
            'distractor': distractor,
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


def score_binding_answer(task, output):
    """Return 1.0 if the model output names the gold answer, else 0.0.

    The gold answer must appear in the head of the output; in a two-way
    comparison the wrong candidate must not appear before it.
    """
    head = ' '.join(output.strip().lower().split())[:120]
    gold = task['answer'].lower()
    gold_position = head.find(gold)
    if gold_position == -1:
        return 0.0
    distractor = task.get('distractor')
    if distractor:
        distractor_position = head.find(distractor.lower())
        if distractor_position != -1 and distractor_position < gold_position:
            return 0.0
    return 1.0


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

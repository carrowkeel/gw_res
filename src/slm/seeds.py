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
    'smoke is drifting from a neighboring field toward the dry barn',
    'the tide is coming in across the sandbar with the nets still staked',
    'a wheel is wobbling on the loaded cart an hour from town',
    'the cellar is taking water and the pump handle has just cracked',
    'frost is forecast overnight and the seedlings are still uncovered',
    'the bread order was doubled and the oven can only take half',
    'a lamb is missing and the light is going fast',
    'the ferry leaves in twenty minutes and one traveler is not back',
    'the paint is drying faster than the second coat can go on',
    'a customer is disputing a bill while the queue grows behind them',
    'the ice is thinning under the fishing huts by mid-morning',
    'the well bucket has split and the troughs are still to fill',
    'a swarm of bees has settled over the schoolyard gate',
    'the account books do not balance and the auditor comes tomorrow',
    'the mare has thrown a shoe halfway up the mountain track',
    'the milk delivery is souring in the sun on a locked porch',
    'the visiting inspector has arrived a day earlier than announced',
    'the last sack of feed is open and the supplier is closed until '
    'Monday',
    'the print run carries a wrong date and half is already boxed',
    'the chimney is drawing badly with guests due at dark',
    'the rope on the flag pole has jammed with the ceremony starting',
    'the apprentice has cut the cloth short on a paid order',
    'the harvest crew is ready but the field is still wet',
    'the market stall awning is tearing in a rising wind',
    'the medicine cabinet key left with someone gone for the day',
    'the recital hall is double-booked for the same evening',
    'the firewood is running low with a cold week forecast',
    'the postman has left a parcel that clearly belongs next door',
    'a window latch has failed with rain blowing in on the ledgers',
    'the tour group is split between two platforms as the train pulls in',
    'the goat has got into the vegetable garden again',
    'the water tank reads a quarter with three days of the trek left',
    'the buyer wants the price settled tonight and the partner is '
    'unreachable',
    'the front rows of the concert have been sold twice over',
    'the churn has stopped mid-batch with the cream half turned',
    'the lantern fuel is low with the last cave chamber unmapped',
    'the delivery van keys are locked inside the van',
    'the wedding cake tier has slumped in the heat',
    'the exam papers are one short with the class already seated',
    'the anchor is dragging in the crowded moorings',
    'the kiln has fired uneven and the fair opens tomorrow',
    'a pipe is knocking behind the wall of the full guesthouse',
    'the hay loft door is banging itself loose in the gale',
    'the market fee has gone up on the morning of market day',
    'the second oar has floated out of reach of the drifting boat',
    'the ladder is too short for the gutter that is overflowing now',
    'the shepherd on the high pasture has missed the radio check-in',
    'the change has run out on the busiest morning of the year',
    'the fruit is ripening faster than pickers can be hired',
    'the schoolroom stove is smoking with the children arriving',
    'a signature is missing on papers due at the registry by noon',
    'the pack pony is favoring a leg at the base of the pass',
    'the reservoir sluice is stuck half open above the lower fields',
    'the drummer is stranded a town away an hour before the dance',
    'the dough has not proved and the stall opens at eight',
    'the survey stakes have been pulled out overnight',
    'the tar barrel has tipped across the boatyard slip',
    'the archive boxes are stacked in the path of a leaking roof',
    'the last coach is full and two passengers still hold tickets',
    'the beehives need moving before the crop spraying at dawn',
    'the scaffold planks are two short with the crew on the clock',
    'the birthday order names the wrong child and pickup is at four',
    'the tide table in use is last year\'s and the crossing is set for '
    'six',
    'the sawmill blade is dulling mid-order with no spare on the rack',
    'the seed drill has jammed with half the field sown',
    'the museum\'s only guide has lost their voice before the tour',
    'the toll box is full and the collector\'s shift runs to midnight',
    'the greenhouse vent has stuck shut on the hottest day yet',
    'the choir robes are at a cleaner\'s across town in festival traffic',
    'the fish crates have arrived without ice in high summer',
    'the map and the trail markers disagree at the fork',
    'the ink has arrived in the wrong color for a dated poster',
    'the orchard gate is open and the cows have noticed',
    'the night watchman has not signed the midnight round',
    'the spare sail is mildewed and the forecast is for heavy weather',
    'the tea urn has failed with the hall filling for the meeting',
    'the loan payment falls due the day before the invoice clears',
    'the trellis is leaning under the heaviest crop in years',
    'the courier route crosses the parade just called for noon',
    'the butter went into the pastry salted instead of plain',
    'the stage curtain rope has frayed to a few strands before the show',
    'the harbor light is out with two boats still fishing after dark',
    'the census forms are due back and a street has been missed',
    'the grain reads damp with the buyer\'s truck at the gate',
    'the footpath sign points the wrong way at the cliff fork',
]

HELDOUT_DECISION_SITUATIONS = [
    'the observatory dome has stuck open with hail beginning',
    'the lighthouse relief boat is a day late in falling weather',
    'the beekeeper\'s smoker has gone out mid-inspection',
    'the glass shipment shifted in transit and the crate rattles',
    'the town clock has been striking the wrong hour since noon',
    'the circus animals have arrived before the fencing did',
]

BRIEFING_PERIODS = ['week', 'fortnight', 'month', 'quarter', 'season']

BRIEFING_DOMAINS = [
    {'key': 'produce-stall', 'setting': 'a produce stall',
     'items': ['apples', 'pears', 'plums', 'onions', 'cherries'],
     'unit': 'crates', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'the till holds %d'},
    {'key': 'grain-store', 'setting': 'a grain store',
     'items': ['wheat', 'barley', 'oats', 'rye'],
     'unit': 'sacks', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'fish-quay', 'setting': 'a fish quay',
     'items': ['cod', 'herring', 'mackerel', 'crab'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'the float holds %d'},
    {'key': 'timber-yard', 'setting': 'a timber yard',
     'items': ['pine boards', 'oak boards', 'birch boards'],
     'unit': 'stacks', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'wool-shed', 'setting': 'a wool shed',
     'items': ['coarse wool', 'fine wool', 'dyed wool'],
     'unit': 'bales', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'the fund holds %d'},
    {'key': 'hardware-store', 'setting': 'a hardware store',
     'items': ['nails', 'hinges', 'rope', 'paint tins'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'order', 'divest': 'return',
     'resource': 'the till holds %d'},
    {'key': 'bakery', 'setting': 'a bakery',
     'items': ['loaves', 'rolls', 'buns', 'pies'],
     'unit': 'trays', 'link': 'of', 'value': 'price',
     'acquire': 'bake', 'divest': 'sell',
     'resource': 'the flour bin holds %d'},
    {'key': 'clinic-stores', 'setting': 'a clinic storeroom',
     'items': ['bandages', 'splints', 'tonic bottles'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'order', 'divest': 'release',
     'resource': 'the budget stands at %d'},
    {'key': 'expedition-depot', 'setting': 'an expedition depot',
     'items': ['ration packs', 'fuel canisters', 'rope coils'],
     'unit': 'bundles', 'link': 'of', 'value': 'weight',
     'acquire': 'take', 'divest': 'leave',
     'resource': 'the weight allowance stands at %d'},
    {'key': 'vineyard', 'setting': 'a vineyard cellar',
     'items': ['red wine', 'white wine'],
     'unit': 'casks', 'link': 'of', 'value': 'price',
     'acquire': 'make', 'divest': 'sell',
     'resource': 'the cellar fund holds %d'},
    {'key': 'dairy', 'setting': 'a dairy',
     'items': ['butter', 'cheese', 'cream'],
     'unit': 'crates', 'link': 'of', 'value': 'price',
     'acquire': 'make', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'print-shop', 'setting': 'a print shop',
     'items': ['posters', 'pamphlets', 'cards'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'print', 'divest': 'scrap',
     'resource': 'the paper stock stands at %d'},
    {'key': 'warehouse', 'setting': 'a warehouse',
     'items': ['stoves', 'lamps', 'chairs', 'kettles'],
     'unit': 'pallets', 'link': 'of', 'value': 'price',
     'acquire': 'stock', 'divest': 'clear',
     'resource': 'the budget stands at %d'},
    {'key': 'quarry', 'setting': 'a quarry',
     'items': ['flagstones', 'gravel', 'sand'],
     'unit': 'loads', 'link': 'of', 'value': 'price',
     'acquire': 'cut', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'brewery', 'setting': 'a brewery',
     'items': ['pale ale', 'stout', 'cider'],
     'unit': 'barrels', 'link': 'of', 'value': 'price',
     'acquire': 'brew', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'florist', 'setting': 'a florist\'s shop',
     'items': ['roses', 'tulips', 'lilies', 'carnations'],
     'unit': 'bunches', 'link': 'of', 'value': 'price',
     'acquire': 'order', 'divest': 'sell',
     'resource': 'the till holds %d'},
    {'key': 'bookshop', 'setting': 'a bookshop',
     'items': ['novels', 'atlases', 'almanacs'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'order', 'divest': 'return',
     'resource': 'the till holds %d'},
    {'key': 'chandlery', 'setting': 'a ship chandlery',
     'items': ['lamp oil', 'tar', 'pitch'],
     'unit': 'barrels', 'link': 'of', 'value': 'price',
     'acquire': 'stock', 'divest': 'sell',
     'resource': 'cash stands at %d'},
    {'key': 'feed-merchant', 'setting': 'a feed merchant\'s yard',
     'items': ['hay', 'straw', 'clover'],
     'unit': 'bales', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'pottery', 'setting': 'a pottery',
     'items': ['bowls', 'jugs', 'plates'],
     'unit': 'crates', 'link': 'of', 'value': 'price',
     'acquire': 'make', 'divest': 'sell',
     'resource': 'the clay store stands at %d'},
    {'key': 'orchard', 'setting': 'an orchard',
     'items': ['apples', 'cherries', 'walnuts'],
     'unit': 'baskets', 'link': 'of', 'value': 'price',
     'acquire': 'pick', 'divest': 'sell',
     'resource': 'the wage fund holds %d'},
    {'key': 'cheese-cellar', 'setting': 'a cheese cellar',
     'items': ['young cheese', 'aged cheese'],
     'unit': 'wheels', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'salt-works', 'setting': 'a salt works',
     'items': ['coarse salt', 'fine salt'],
     'unit': 'sacks', 'link': 'of', 'value': 'price',
     'acquire': 'make', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'tannery', 'setting': 'a tannery',
     'items': ['hides', 'belts', 'soles'],
     'unit': 'bundles', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'glass-works', 'setting': 'a glass works',
     'items': ['window panes', 'bottles', 'jars'],
     'unit': 'crates', 'link': 'of', 'value': 'price',
     'acquire': 'make', 'divest': 'sell',
     'resource': 'the budget stands at %d'},
    {'key': 'tea-merchant', 'setting': 'a tea merchant\'s store',
     'items': ['black tea', 'green tea', 'mint tea'],
     'unit': 'chests', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'spice-stall', 'setting': 'a spice stall',
     'items': ['pepper', 'cinnamon', 'ginger', 'cloves'],
     'unit': 'jars', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'the till holds %d'},
    {'key': 'coal-yard', 'setting': 'a coal yard',
     'items': ['house coal', 'coke', 'kindling'],
     'unit': 'loads', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'ventures', 'setting': 'a trading desk holding stakes in '
     'local ventures', 'proper': True,
     'unit': 'shares', 'link': 'of', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'grower-co-op', 'setting': 'a growers\' co-op buying from '
     'named farms', 'proper': True,
     'unit': 'crates', 'link': 'from', 'value': 'rate',
     'acquire': 'order', 'divest': 'return',
     'resource': 'the fund holds %d'},
    {'key': 'cargo-shares', 'setting': 'a harbor office holding shares '
     'in named boats', 'proper': True,
     'unit': 'shares', 'link': 'in', 'value': 'price',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
    {'key': 'contracts', 'setting': 'a dispatch office holding contracts '
     'with named firms', 'proper': True,
     'unit': 'contracts', 'link': 'with', 'value': 'rate',
     'acquire': 'sign', 'divest': 'cancel',
     'resource': 'the budget stands at %d'},
]

HELDOUT_BRIEFING_DOMAINS = [
    {'key': 'apiary', 'setting': 'an apiary',
     'items': ['honey jars', 'wax blocks'],
     'unit': 'boxes', 'link': 'of', 'value': 'price',
     'acquire': 'harvest', 'divest': 'sell',
     'resource': 'the till holds %d'},
    {'key': 'ice-house', 'setting': 'an ice house',
     'items': ['clear ice', 'packed ice'],
     'unit': 'blocks', 'link': 'of', 'value': 'price',
     'acquire': 'store', 'divest': 'sell',
     'resource': 'cash stands at %d'},
    {'key': 'cider-press', 'setting': 'a cider press',
     'items': ['sweet cider', 'dry cider'],
     'unit': 'kegs', 'link': 'of', 'value': 'price',
     'acquire': 'press', 'divest': 'sell',
     'resource': 'cash stands at %d'},
    {'key': 'fishing-stakes', 'setting': 'a quayside office holding '
     'stakes in named fishing crews', 'proper': True,
     'unit': 'stakes', 'link': 'in', 'value': 'rate',
     'acquire': 'buy', 'divest': 'sell', 'resource': 'cash stands at %d'},
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

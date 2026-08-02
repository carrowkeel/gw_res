"""Games: interactive SFTs with program-verifiable outcomes.

A game supplies tasks the program can grade, and a trainer turns graded
play into supervised updates. The design walks away from proven
imitation learning one step at a time instead of jumping to the far end:
the math SFT demonstrated real progress because every token was exactly
supervised, so each game changes exactly one thing about that recipe and
must pass before the next branch opens. The progression is a tree, not a
ladder - noise tolerance, distractor tolerance, and selection pressure
are separate branches grown from the same trunk - and the market
simulator is one game among many, re-entered only when the abilities it
demands in tandem have each been established alone and its entry
threshold is known.

Single-turn games are trained by slm.gametrain (expert iteration:
sample, verify, train on the model's own verified answers). The market
is the multi-turn game, trained by slm.simtrain.
"""

from importlib import import_module

GAMES = {
    'math': 'slm.games.math',
    'market': 'slm.games.market',
}


def load_game(name):
    if name not in GAMES:
        raise ValueError('unknown game %r; known games: %s'
                         % (name, sorted(GAMES)))
    return import_module(GAMES[name])

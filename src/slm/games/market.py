"""The market game: the multi-turn economic simulator, as one game.

The simulator predates the games framing and keeps its own machinery:
world and dynamics in slm.market, rendering in slm.render, parsing and
verification in slm.listener, the self-play trainer in slm.simtrain,
and the fixed-battery grader in slm.simeval. It is the far end of the
tree, not the trunk: eight sweep rounds showed that its outcome signal
polishes a taught policy rather than teaching one, that clean advisor
tips make it cheatable by tip-following, and that its per-turn
signal-to-noise ratio was inherited rather than chosen. It stays here
as a game to be re-entered once the single-turn games establish which
abilities are learnable from verified self-play, at what noise
threshold - and after its cheats are closed.
"""

from ..market import (
    blind_policy, oracle_policy, play_game, sample_market, start_game,
    step_game,
)

TRAINER = 'slm.simtrain'

__all__ = [
    'blind_policy', 'oracle_policy', 'play_game', 'sample_market',
    'start_game', 'step_game', 'TRAINER',
]

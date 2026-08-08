# Games: the tree from imitation to outcome-driven learning

## Why games

The math SFT made real, demonstrable progress; the market simulator did
not. The difference was not subtle: the SFT delivers dozens of exactly
supervised tokens per example with deterministic ground truth, while the
simulator delivered roughly one noisy bit per turn, in an environment
demanding many abilities at once (register production, entity naming,
report reading, cross-line composition, sizing arithmetic, state
tracking, credit assignment under noise), with teachers that had already
leaked the solution and a clean advisor channel that made the game
cheatable by tip-following. Eight sweep rounds optimized recipes around
that signal and produced model-performance progress close to zero.

The correction is to walk away from the proven method one step at a
time. A game is an interactive SFT: it supplies tasks the program can
grade, the model plays, and the graded play becomes supervised training
data. Each new game or parameter changes exactly one thing about the
recipe that worked, and must pass before the next branch opens.

## The tree

The progression is a tree, not a ladder: from a working trunk, separate
branches test separate abilities, and a branch that fails marks a
threshold rather than ending the program.

**Trunk - expert iteration on math (gametrain + games/math).** The
tasks, dialogue format, and exact answers come from the math SFT; the
only change is the data source. The model samples answers, the program
keeps the verified ones, and the model trains on its own correct play
(response-masked cross-entropy, unchanged from the SFT stages). The
trunk question: can this model improve from self-generated,
program-verified data at all? Measured as greedy solve rate on a fixed
held-out battery, every round.

**Branches, as math game parameters** (one dial each):

- distractor_lines - extra numbered statements about unrelated items.
  Solving now requires attending to the relevant quantities and
  ignoring the rest, the ability the market assumes. Where solve rates
  break as distractors grow is the attention threshold.
- score_noise_sigma - Gaussian noise on the grade the trainer sees, so
  wrong answers are sometimes kept and right ones dropped. Sweeping
  sigma measures how much selection noise verified self-improvement
  survives: the signal-to-noise threshold. Any outcome-scored game must
  be designed above it - the market's per-turn SNR was inherited, never
  chosen, and is the leading suspect for why it taught nothing.
- maximum_value and kinds - difficulty on the trunk itself.

Future branches follow the same rule (one new element per game):
sequential tasks for state, choice tasks for selection among options.
The market game is re-entered only when the abilities it needs have
each passed alone, its entry threshold is known from the branch
measurements, and its cheats (clean tips, decorative reasons the
checkers cannot see through) are closed.

## Health before progress

The simulator rounds showed that degradation arrives through the
language before it shows in scores. Every gametrain round therefore
logs, next to the solve rates: replay loss on fixed stage-1 corpus
blocks (drift), repeated-bigram rate and mean length over the round's
samples (the loop signature and padding), distinct-response rate, and
the kept fraction. A run whose solve rate climbs while its health
metrics slide is a failure, whatever the score says.

## Where things live

- src/slm/games/ - the game registry. games/math.py is the single-turn
  math game; games/market.py is the market simulator repositioned as
  one game among many (its machinery stays in market.py, simtrain.py,
  simeval.py).
- src/slm/gametrain.py - the expert-iteration trainer for single-turn
  games. Writes checkpoints/gametrain-<game>/ with history.jsonl,
  ckpt_best.pt (by held-out solve rate), and game_report.json.
- configs/game_math.yaml - the base config; configs/experiments/
  games_round1.yaml - the first sweep: the trunk, a seed replicate,
  and small doses of each branch.

Run one:

    python -m slm.gametrain --config <run>/config.yaml

Sweep (teachers are not involved; gametrain resolves its base
checkpoint from --base-run):

    python slurm/sweep.py --sweep configs/experiments/games_round1.yaml \
      --base-run runs/t1_full-<id>

# B200 experiment plan

## Background

The project trains the SGM — a from-scratch decoder whose eventual
contract is Graph→Graph: language processed inside graph nodes,
structure across edges, graph output. The route there runs through
flat-form competence first: stage-1 language pretraining, two bridging
SFT stages, and then the main effort, simulation training — the model
plays a programmatic economic market (latent factor shocks, partial
leaks as language reports, buy/sell/hold decisions in a structured
register, per-step scores) and is updated by score-weighted
cross-entropy on its own trajectories. Every input is rendered and
every output parsed by program code; every metric is program-owned
(exact-match, grounded and truthful reason rates, distinct decisions,
returns against replayed blind and oracle references). No online LLM
sits anywhere in the loop.

Five sweep rounds on single L40S GPUs (8-16 parallel 150-step runs
each) established the learning method before scaling, each round's
failures narrowing the design:

- The structured register (`move: ... | reason: ...`), taught by a
  short imitation stage before the sim, is the only configuration that
  learns returns while keeping language intact; freeform collapsed.
- A deterministic sizing teacher makes quantity arithmetic load-bearing
  (signal-proportional position sizes survive outcome training and
  outperform the random-quantity baseline); rendering numbers in the
  input instead corrupted output form and decoupled reasons — retired.
- A rotation teacher closing the sell-side demonstration gap (funding
  sells with truthful reasons) is the current best recipe: 37% of the
  blind-to-oracle headroom with truthfulness 0.82-0.88, the program
  record on both channels at once.
- Difficulty should come from richness — larger worlds (up to 20
  companies), denser reports, within-game coverage ramps all improved
  results; information scarcity instead induced register bleed and
  reason confabulation.
- The advisor tip is a copyable crutch: on clean data, tip-copying and
  news-composition are observationally equivalent, and the model learns
  the cheaper one. Corrupting tips mid-training breaks the model rather
  than teaching arbitration; the countermeasure is corrupting tips in
  the *teaching* data, where the tip-blind teacher visibly ignores
  wrong tips.

The infrastructure matches: sweeps materialize and validate per-variant
configs on the login node, submit independent single-GPU Slurm chains,
train their own tagged teachers with job dependencies, and reduce to
one comparison table. Capacity rungs (30M/60M/150M) share corpus,
tokenizer, and bridge pools with parameters as the only variable. The
graph pipeline (context-graph transform, structure-token tokenizer,
graph pretraining, matched-budget evaluation) is implemented but
dormant, gated on flat-form competence — a gate the sim results are
approaching.

Corpus and bridge-pool production (the LLM-heavy work) stays on the
current hardware; the B200s are for training-side tests only.

## 1. Simulation training at scale

**Long horizons.** 150 steps was an L40S budget, not a design choice,
and the best runs are still climbing at step 149. The first B200
experiments extend the established recipe (rotation teacher, wide-world
curriculum) to 1,000-5,000 steps to find where it saturates, and —
more important — whether the language channel survives long outcome
pressure. Truthfulness under pressure is a curve we have only seen the
first 150 steps of; Goodhart failures that take 500 steps to develop
are invisible on the pilot hardware.

**Advisor arbitration.** The one skill 150 steps could not teach:
cross-checking tips against news. Long-horizon runs from tip-skeptic
teachers, with gentle accuracy anneals (1.0 -> 0.9 -> 0.8 over
thousands of steps), test whether arbitration is learnable at all or
the crutch always wins. This is the cleanest open scientific question
in the sim program.

**Capacity under the sim.** Run the 30M/60M/150M ladder — and a new
~300M rung the B200s make practical — through identical sim configs.
Where adjacent rungs' return and truthfulness curves coincide, capacity
is not binding; where they separate is the size the method actually
uses. This decides model size before any larger commitment.

**Bigger batches, richer worlds, replication.** games_per_batch and
market_repeats scale directly with memory; more repeats sharpen the
shared-luck advantage baseline that the whole update rests on. Worlds
beyond 4x5 need only more fields in the market spec. And the sweep
unit stays 16 variants, but each variant becomes 3+ seeds — the pilot
rounds drew conclusions from single runs, acceptable for triage,
insufficient for the decisions that follow.

**Longer context.** block_size is 1024 and wide-world games already
crop early quarters away. 4k-8k contexts test whether decision quality
improves when the model can see its whole game — and set up the graph
comparison below, which is precisely about what to keep in context.

## 2. Graph native input and output

The dormant graph_*.py pipeline is the second B200 track. It folds
text into context graphs (nodes under a token limit, two growth moves,
linearized with reserved structure tokens), trains a tokenizer with
those markers, pretrains the same architecture on graph-packed
binaries, and evaluates flat versus graph context at matched token
budgets: the flat model gets the most recent transcript that fits, the
graph model gets the graph reduced to the same budget by dropping the
subtrees least related to the latest turn — recency truncation against
relevance selection at equal cost.

- **Graph pretraining at 60M and 150M** on the transformed stage-1
  corpus: does structural context change what the same parameter count
  can hold? This is a from-scratch pretraining run per rung — B200
  work by definition.
- **Matched-budget evaluation** on held-out conversations, the
  experiment the pipeline was built for, never yet run at scale.
- **Graph context for the sim** — the experiment that joins the two
  tracks and the real test of the Graph→Graph contract. Game history
  is naturally graph-shaped (per-company subtrees of reports and
  decisions rather than a chronological transcript); folding sim
  context into a graph and training the sim on top of a graph-pretrained
  checkpoint tests whether relevance-selected context beats recency
  cropping exactly where it should: long games in wide worlds, the
  regime section 1 opens up. Sim-side rendering changes are small; the
  training loop is unchanged.

## 3. Supporting tests that need the hardware

- **Batched self-play generation.** Generation is sequential per game
  within the lockstep quarter — fine at 60M on an L40S, the bottleneck
  for every long-horizon experiment above. Batching it is engineering,
  but validating throughput and identical-trajectory determinism is a
  B200 test.
- **Threshold and gate calibration at scale.** The entry gate, template
  eval, and no-signal abort were tuned on 60M pilots; each new capacity
  rung needs its thresholds re-read from the same instruments before
  the long runs consume real budget.
- **Not on B200s:** corpus generation, bridge-pool generation, and any
  other LLM-driven data production — these stay on existing hardware;
  the sim loop needs no LLM at all.

The sequencing follows from the dependencies: long-horizon sim runs and
the capacity ladder first (they validate the method the graph track
builds on and need no new code), graph pretraining and the matched
budget eval in parallel once rungs are chosen, graph-context sim last.

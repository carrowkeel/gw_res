# Conversation summary, part 2: from grounded worldgen to the SGM redirection

This document continues `docs/conversation_summary.md`, which recorded the
project up to the adoption of program-authored generation (worldgen). It
covers the events after that point, ending at the strategic redirection to
a simulator-grounded controller model (SGM). The final exchanges of the
source conversation are reproduced in detail because they define the
current plan; nothing from those exchanges has been implemented yet.

## 1. State of the repository at the time of writing

- `main` is at the merge of PR #25 (broadened grounded generation,
  kind-differentiated SFT, matched eval).
- PR #26 is open and unmerged: it replaces the archaic invented registers
  with eight modern everyday ones (office, clinic, school, cafe, depot,
  studio, transit, sports), mixes ordinary modern first names 50/50 with
  invented ones, and fixes the binding scorer to accept the gold answer in
  the final sentence (derivation conclusions), not only the first 120
  characters. Two answers reported as wrong in the second ladder were in
  fact correct and mis-scored.
- No SGM, simulator, or tier machinery exists in the code. The redirection
  below is design only.

## 2. Operational episodes between worldgen and the redirection

- **Run isolation.** Fixed output directories caused collision risk between
  concurrent runs. `--run-id` was added: each submission writes to a
  run-id-suffixed tree (`runs/world-<id>`), the resolved config is baked to
  `config.resolved.yaml`, `corpus_dir` stays literal as an input reference
  (PR #22).
- **The stalled ladder.** After a resume, nothing ran past generation. The
  diagnosis, driven by sacct output rather than guesswork: `merge-full` was
  chained `afterok:` on the 16-task generation array; several tasks failed
  at vLLM engine initialization (KV-cache allocation), and `afterok` on an
  array requires every task to exit zero, so Slurm silently dropped the
  merge and everything chained behind it. Generation itself was fine —
  workers logged "share already complete". Two clarifications came out of
  it: generation is shared across rungs (each rung's gen job tops the one
  shared pool up to a cumulative target, then freezes a snapshot), and the
  strict afterok gate was redundant because the merge has its own
  completeness check. Fix: `afterany` plus requeue of failed gen tasks
  (PR #23/#24).
- **Second ladder (world-52c749db) results.** Kind-differentiated SFT
  delivered the first genuine pretrain-to-sft capability gain: grounded
  0.28 to 0.60, binding 0.38 to 0.73 at the full rung, with reserved kinds
  doing the work (multihop 1.0, notstated 0.75). The grounded sub-scores
  were understated by the scorer bug fixed in PR #26. The fluency dip
  (6.8 to 4.9) was partly instrument change (completion seeds became half
  grounded).

## 3. The turning point: questioning the whole approach

The user rejected further incremental improvement of the grounded
generator: the leap from free LLM text (reasonable language) to fully
program-grounded text produced a corpus so narrow in text types and
vocabulary that the model answers "Hello" with "Hello is 58 years old".
The pipeline was stuck improving the generator instead of progressing.
Direction given: consider drastic changes, including scrapping the current
generation method.

The response conceded that the evaluation had been matched to the corpus,
hiding exactly this fragility, and laid out the evidence table:

| | ungrounded LLM free-writing | 100% program-grounded |
|---|---|---|
| language (grammar/coherence) | 6.8 / 7.0 | collapsed to schema |
| binding | 0.03 (chance) | 0.73-0.81 |
| what it precludes | any reasoning/graph work | any general language use |

The diagnosis: the mistake was dosage and placement, not grounding itself —
the entire language distribution was replaced with schema text instead of
embedding a binding-bearing slice inside a language distribution. One
overlooked result changed the option space: SFT alone lifted binding
0.38 to 0.73 in ~2,900 steps on reserved kinds, meaning binding is cheap to
add on top of a compatible base, undermining the assumption that the base
corpus had to be grounded. Four options were offered: (A) hybrid mixture
(free-writing bulk plus 10-30% grounded slice), (B) free base plus grounded
pairs only — the decisive cheap test, (C) scrap synthetic for the base and
use curated natural text (the TinyStories recipe), (D) keep iterating the
generator (argued against). Recommendation was B. The user did not pick any
of them; instead they reframed the goal.

## 4. The reframe: NASA-like controllers

The user's direction, closely paraphrased: not interested in prose-writing
LLMs — that would reproduce what has been done millions of times. The
target is NASA-like controllers with high-level communication and reasoning
abilities, an understanding of delegation, risk, and mission goals, learned
from the corpus rather than introduced as fine-tuning alignment layers.
Three axes to think in:

1. **Corpus composition** — how much conversation versus prose is needed
   just to teach language; fusing texts based on real-world knowledge with
   logic (for instance, templates for describing real facts as a vehicle
   for teaching logic).
2. **Training sequence** — not the vague pretraining/SFT split but distinct
   types of knowledge, logic, and response architecture, including
   multi-user architectures.
3. **Integration of logic** — more direct involvement of capable models at
   crucial points in generation (currently the capable model's influence is
   only indirect, through authored code). The cheap model (Qwen) should
   mainly provide language structures, never facts or logic. This is not
   distillation. Constraints elsewhere should be relaxed; known solutions
   kept.

The response introduced the unifying frame: worldgen's successor is a
**mission simulator, not a fact sampler**. A controller's linguistic world
is state + events + roles + decisions, all simulable (numeric budgets,
tick-based timeline, stochastic events, decision rules with computable
correct outcomes), and every rendered text — comm loop, status report,
handover, anomaly post-mortem — is about verifiable state. Key design
points: dialogue-heavy prose-light composition with the ratio as an
experiment; knowledge entering only through logic-bearing templates; a
staged curriculum (language substrate, then protocol, then judgment), each
stage with replay and reserved eval kinds; multi-user structure learned
from role-prefixed traffic (response-as-role rather than
response-as-assistant); and a division of authorship in which the capable
model authors low-volume high-leverage artifacts (blueprints: scenario
families, dynamics, anomaly distributions, decision doctrine with
rationales), the program executes them and holds all facts, Qwen renders
surface language only, and the capable model additionally acts as examiner.
One blueprint instantiates thousands of documents — structure is
transferred, not sampled behavior.

## 5. "Translating a game into language"

The user's consolidation, closely paraphrased: the ground truth underlying
everything must be a real, complex simulation that is then translated to
language. Training tiers: basic language comprehension first, then the
simulation, then real data. The test is whether simulation logic transfers
to real problems (did the agent learn arithmetic from the simulation and
can it apply it elsewhere). The agent must degrade gracefully to simple
conversation — not "poet laureate told to speak like a caveman" (how
current methods make big models terse) but "pilot asked what they ate for
lunch": the operational register is the default, plain conversation is a
retained capability. Leakage asymmetry is acceptable in one direction only:
pilot-speak in casual chat is fine; verbose polite prose in mission-control
traffic is not. The open questions: how much can be predefined, since all
possible answers of a complex simulation cannot be pre-generated; ideally
training would occur directly with the simulator, with the language model
as a translator — is that feasible, or horribly slow?

The response resolved the predefinition worry with one distinction: you do
not map the state space, you **sample trajectories** from it (a chess
engine does not enumerate chess). The simulator needs only two
capabilities: run forward from any state, and adjudicate any state on
demand (doctrine as executable rules, so the correct call for a
never-seen state is computed, not stored). Fresh trajectories forever; the
combinatorial explosion is the feature.

Feasibility rests on **two-speed rendering**: a canonical layer (the
program emits a typed structured log of states, events, decisions,
rationale codes — this layer is the graph) and a realization layer (a
surface-form bank of paraphrase variants per message type, built offline by
Qwen under capable-model-designed schemas; online rendering is canonical
log plus sampled surfaces, pure CPU). Measured numbers: training consumes
roughly 30k tokens/sec (~60 documents/sec) on one L40S at mini scale;
Qwen-per-document rendering manages ~25 docs/sec/GPU (online use would need
2-3 generation GPUs per training GPU — feasible, wasteful); bank rendering
reaches ~100-300 docs/sec per CPU core, so one core outruns the training
GPU; the bank build itself is 1-2 GPU-hours, amortized. Verdict:
simulator-fed training is cheaper than the current pipeline, because the
expensive Qwen pass happens once per bank instead of once per corpus.

An interaction ladder was laid out (each rung a known solution):
(1) fresh-epoch corpora — regenerate the corpus every epoch/rung with the
cheap renderer, training loop untouched; (2) interactive evaluation — the
sim renders a live state, the model makes the call, the sim adjudicates;
re-rolled scenarios with different states must yield different correct
calls, so surface memorization scores zero; (3) true streaming — the data
loader draws from the sim directly, only if fresh-epochs shows staleness;
(4) expert iteration (DAgger-style) — model proposes, sim adjudicates,
corrections become training data; RL-from-scratch stays off the table.
Doctrine statement: the small model is never the world model; it is the
interface between language and the sim's state and actions. The three
tiers map to mixture annealing with the conversational slice never
dropping below a ~10-15% floor, and register discipline becomes an
explicit asymmetric metric (ops outputs must pass field-completeness and
brevity budgets; chat outputs need only coherence). Two design decisions
were flagged: simulator fidelity for v1 (recommendation: a minimal
mission-day state machine — 4-6 coupled systems, budgets, stochastic
faults, ~20-40 ticks, a doctrine table — growing only when eval
saturates) and bank size versus leakage (measurable via bank-ablation
rungs).

## 6. The four refinements and the SGM notation

The user's final substantive message made four points:

1. **Bank collapse risk.** Is a stochastic state-to-language mapping enough
   variation to keep the model from learning bank phrases as whole
   commands? If it just learns the mapping, it might as well learn commands
   directly. And on the output side: how are model outputs mapped to
   commands without any LLM interface to the simulation?
2. **Interactive evaluation as a variable-length stage.** Since data is not
   pre-generated, the stage can run until a loss/metric goal, or until
   performance on real data — and the distinction between ladder rungs 2
   and 3 was unclear.
3. **Notation.** The model is the **SGM (Small Graph Model)**, as opposed
   to Qwen the LLM. Qwen is just a translator layer if it creates language
   input/output for the simulator. The SGM is not a translator: it is a
   **Graph-to-Graph function** that can process language inside graph
   nodes, process the graph itself (edges between nodes), and reason about
   multiple graphs and non-graph data to produce graph output.
4. **Fresh tier-1 corpus.** Do not reuse the old language corpus — generate
   something new to avoid absorbing old intent from previous
   implementations. Give the model the best chance at language and
   conversation by not constraining Qwen, since constrained Qwen produces
   highly similar texts.

The request that followed: address the points and produce an MVP
description plus an implementation plan as concise markdown documents.

The response's answers: (1) the fix is an **asymmetry — read many, write
canonical**. Input-side variation must be real: compositional fragment
banks (independent slots — entity form, verb form, quantity form,
hedge/urgency — giving combinatorial counts), a 20-30% fraction of
documents fully paraphrased by the LLM, bank refresh between epochs so
memorized surfaces go stale, all resting on the tier-1 natural-language
mass. Output-side: no variation at all — commands are typed,
schema-validated canonical forms parsed programmatically; no LLM anywhere
in the sim loop. The asymmetry is the controller's job description: robust
comprehension, disciplined production. (2) Rung 2 was evaluation-only, rung
3 was fresh training data; the user's frame merges them correctly into a
variable-length tier-2 stage trained on freshly sampled trajectories with
metric-based stopping criteria. (3) Notation adopted; it promotes the
dormant graph pipeline to the core interface. One concession for the MVP:
SGM v0 is the existing decoder trained over a canonical, invertible graph
serialization — a Graph-to-Graph function whose encoding happens to be a
token stream; graph-native I/O is the successor, and milestone M3 becomes
the serialized-versus-graph-native architecture decision. (4) Agreed:
fresh unconstrained conversation-heavy tier-1 corpus; old corpora retired
to baselines.

## 7. The MVP document (reproduced)

```markdown
# MVP: SGM v0 — a simulator-grounded controller model

## Notation
- **SIM** — the simulator. Sole source of facts and ground truth. Program
  code; runs forward cheaply, adjudicates any state on demand.
- **SGM** — Small Graph Model, the trained model. A Graph→Graph function:
  processes language inside graph nodes, structure across edges, and
  produces graph output (state queries, commands, communications). v0
  realizes this as the existing decoder over a canonical graph
  serialization.
- **LLM** — Qwen. Translator layer only: supplies language surfaces on the
  input side and translates at the human boundary. Never supplies facts or
  logic.
- **Architect** — the capable model (frontier-class). Authors low-volume,
  high-leverage artifacts: sim doctrine, schema design, surface-bank specs,
  hard eval cases. Never writes corpus tokens directly.

## Ground truth: the simulator
Mission-day state machine, deliberately minimal in v0:
- 4–6 systems with numeric state (power bus, comms window, thermal, fuel,
  data storage), coupled by simple dynamics per tick (~20–40 ticks/episode).
- Stochastic events (faults, weather, schedule slips) with typed severities.
- **Doctrine**: an executable decision table authored by the Architect —
  for any state, the correct action set and its rationale code are computed,
  not stored. Counterfactual property: re-rolling an episode with different
  state values changes the correct calls.
- Invariants checked by construction (budgets conserve, events causally
  precede effects, doctrine deterministic given state).

## Graph schema (canonical layer)
Every episode is a typed graph:
- **Nodes**: system states (typed, numeric), events, messages (language
  payload), decisions, rationale codes, roles (v0: two — controller,
  operator; multi-role deferred).
- **Edges**: temporal succession, causality (event→effect), reference
  (message→system), authority (decision→doctrine rule).
- **Serialization**: deterministic canonical text form (one node per line,
  typed prefixes, stable ordering). v0 SGM trains on serialized graphs;
  the serializer is invertible (parse back losslessly), which is also the
  round-trip verifier.

## Language layer and the variation asymmetry
- **Read many, write canonical.** Input-side language (state reports, event
  notices, operator messages) is varied; output-side commands are canonical
  and schema-parsed. No LLM in the sim loop.
- Input variation sources, in order of importance:
  1. Tier-1 natural-language mass keeps general language in-distribution.
  2. Compositional surface banks: per message type, independent fragment
     slots (entity form × verb form × quantity form × hedge/urgency),
     giving combinatorial surface counts. Built offline by the LLM from
     Architect-authored specs.
  3. Full-paraphrase fraction: 20–30% of rendered messages/documents pass
     through the LLM for free paraphrase (the expensive path, bounded).
  4. Bank refresh between epochs: surfaces drift, structure persists —
     whole-utterance memorization goes stale by design.
- Collapse gauge: n-gram overlap between model completions and bank
  entries, reported per run. Rising overlap with falling general-language
  scores triggers a larger paraphrase fraction, not more bank entries.

## Corpus tiers and schedule
- **T1 — language substrate** (fresh, replaces all prior corpora):
  unconstrained LLM generation, conversation-heavy mix (dialogue > prose;
  prose limited to narrative-causal accounts), diversity via seeds only.
  Purpose: general comprehension and conversational degradation ("pilot at
  lunch"). Never dropped from the mix; anneals to a 10–15% floor.
- **T2 — simulation traffic** (variable-length stage): serialized episodes
  rendered through the language layer — comm loops, status reports,
  handovers, anomaly reports — plus decision examples (state → canonical
  command + rationale). Freshly sampled per epoch (fresh-epoch corpora; no
  fixed dataset). Mixture anneals until operational register dominates:
  final distribution sets the default register. Stage runs until stopping
  criteria, not a token budget.
- **T3 — real data** (measurement first, finetune second): public
  operational corpora (e.g. Apollo loop transcripts, ATC phraseology) and
  real arithmetic word problems. Primary role: **transfer probes** — held
  out, never trained on, answering "did sim logic transfer". A small T3
  finetune is a later, separate decision.

## Interfaces
- SGM input: serialized graph context (+ optional plain-language turn).
- SGM output: canonical command/communication lines, schema-validated by
  parser; parse failures are scored errors.
- Human boundary (demo/chat): LLM translates free language ↔ canonical
  forms. Out of training loop entirely.

## Evaluation (all construction-verified unless noted)
1. State tracking: value/margin queries at arbitrary ticks.
2. Decision exact-match with **counterfactual re-rolls**: same scenario,
   perturbed state, different gold call — surface memorization scores zero.
3. Routing: which role/system a message concerns.
4. Decline: questions the episode does not answer ("not stated"
   discipline).
5. Register discipline: ops outputs pass field-completeness and brevity
   budgets (required readback fields present, no filler); chat outputs need
   only coherence. Asymmetric by design.
6. Transfer probes (T3, never trained): real arithmetic word problems; real
   ops-transcript comprehension. The falsifiable claim of the approach.
7. Language fragility gauge (judged): general conversation and the
   "Hello"-class inputs; graceful degradation required, first-class metric.

## Stopping criteria for T2 (variable-length)
- Decision exact-match ≥ 0.8 with counterfactual consistency ≥ 0.7 at the
  iteration scale (micro), sustained across two fresh corpora; or plateau
  over three consecutive fresh epochs. Transfer probes measured at every
  checkpoint but never gate T2 (they gate the *approach*).

## Explicitly out of scope for v0
- Multi-role training (later stage in the sequence), RL, graph-native SGM
  architecture (v1 successor; M3 compares serialization vs. graph-native),
  true streaming data loader (fresh-epoch regeneration approximates it),
  T3 finetuning.
```

## 8. The implementation plan (reproduced)

```markdown
# Implementation plan

## Principles
- Novelty budget: SIM, schema, doctrine, language layer, eval. Everything
  else reuses proven machinery: ladder submitter with run isolation and
  resume, packing/replay, report, early-stop (generalized to metrics).
- Every generated artifact verified by independent recomputation or
  round-trip, as with worldgen (500+ samples per type before any training).
- Architect artifacts (doctrine, schemas, bank specs, hard eval cases) are
  versioned files in the repo, reviewed like code.

## Phase 0 — redirection bookkeeping (small)
- Intent graph: new target node (NASA-like controller from corpus), SGM
  notation node, supersede/annotate nodes tied to worldgen-as-corpus
  (46, 48, 49, 52); M3 reframed as serialization vs. graph-native.
- Technical report: new target statement, this plan.
- Old corpora/runs retained on disk as baselines only.

## Phase 1 — SIM core (new: src/slm/sim/)
- `schema.py`: typed nodes/edges, command vocabulary, validation.
- `dynamics.py`: systems, tick update, event sampling.
- `doctrine.py`: executable decision rules + rationale codes
  (Architect-authored data file + interpreter, not hardcoded).
- `episode.py`: trajectory sampler → canonical graph.
- Verification: invariant suite (conservation, causal ordering, doctrine
  determinism, counterfactual divergence — perturbed state ⇒ different
  gold call in ≥N% of re-rolls), 1k episodes.
- Gate: all invariants pass; Architect doctrine review.

## Phase 2 — serialization + language layer (new: serialize.py, surface.py)
- Canonical serializer with lossless parse-back (round-trip test is the
  verifier).
- Compositional bank builder: Architect spec → LLM offline generation →
  per-slot fragment banks; refresh command.
- Renderer: canonical graph → documents (4 genres) via banks; paraphrase
  fraction through existing vLLM generation machinery.
- Collapse gauge: bank/model n-gram overlap metric in inspect/report.
- Gate: round-trip 100%; rendered-fact spot-verification (parse rendered
  docs, recover state, compare) ≥99%; measured surface entropy per message
  type above threshold.

## Phase 3 — T1 corpus (config + prompts change only)
- New config: unconstrained generation, conversation-heavy type weights,
  no grounding clauses; reuse generate/filter/dedup/workers unchanged.
- Gate: inspect diversity metrics ≥ old ungrounded corpus; judged
  grammar/coherence spot-check at pico scale.

## Phase 4 — training integration (mostly config/data changes)
- Tier mixing in data packing: per-tier weights with annealing schedule
  across stages (reuses instruction-mixing machinery, generalized to
  tiers); T1 floor enforced.
- Fresh-epoch corpora: regenerate T2 between rungs/epochs via cheap
  renderer (submitter already supports chained generation; renderer is
  CPU-fast so this is a light job).
- Pairs: decision examples in canonical output format; kind-differentiated
  SFT retained (reserved kinds mechanism already built).
- Early-stop generalized: stop on eval-metric targets (T2 variable-length),
  reusing patience machinery.
- Gate: end-to-end smoke at pico on one fresh epoch.

## Phase 5 — evaluation (rewrite evaluate around the SIM)
- Interactive evaluator: SIM adjudicates model calls on live episodes
  (no training-loop changes); counterfactual battery; routing; decline;
  register-discipline scorer (field checks + brevity budgets,
  programmatic).
- Transfer probes: held-out real word problems + public ops transcript
  comprehension set (assembled once, versioned).
- Language fragility: judged conversation + "Hello"-class battery.
- Report: tier-labeled tables; collapse gauge; transfer section explicitly
  marked never-trained.
- Gate: evaluator agrees with doctrine on 100% of adjudications in
  self-test (model replaced by doctrine oracle scores 1.0; random scores
  near floor).

## Phase 6 — first runs and decision gates
1. Pico/nano shakedown (plumbing, collapse gauge, fresh-epoch loop).
2. Micro iteration run (the established iteration rung): read decision
   exact-match, counterfactual consistency, state tracking, fragility.
3. Decisions:
   - Counterfactual ≈ chance while decision EM high → surface learning:
     raise paraphrase fraction / bank entropy before touching scale.
   - Metrics rise then plateau below target at micro → mini/small preset
     (scale path already built).
   - Transfer probes move off floor → the approach's core claim has
     evidence; plan T3 finetune and multi-role stage.
   - Fragility gauge fails ("Hello" class) → raise T1 floor.

## Sequencing and effort (rough)
- P0–P1: the design-heavy week; doctrine quality dominates outcome.
- P2–P3: parallelizable; P2 gates everything downstream.
- P4–P5: mostly reuse; evaluator is the largest new surface after SIM.
- P6: cluster time on the existing ladder; micro-first.

## Risks
| Risk | Mitigation |
|---|---|
| Bank collapse (codebook learning) | compositional banks + paraphrase fraction + refresh + collapse gauge; counterfactual eval catches it even if gauge misses |
| Sim too easy → doctrine memorized | doctrine complexity ratchet (more coupled systems/rules) gated on eval saturation |
| Tier interference (T2 erases T1) | T1 floor + replay (built) + comparable-loss forgetting signal (built) |
| Serialization too alien for decoder | it is text with stable structure — same class as code; pico shakedown detects early |
| Transfer never materializes | the honest null result; T3 probes report it cheaply and early, at micro scale |
```

## 9. Where the conversation ended

Two closing notes accompanied the documents: the worldgen puzzle
generators (transfer/ratio/order) survive naturally as the arithmetic
slice inside T2, becoming the trained counterpart of the T3 arithmetic
transfer probe; and Phase 0-1 is where user review matters most — the
doctrine file and graph schema are the two artifacts everything
downstream inherits from, so they should be put in front of the user
before any downstream code. The session ended with an offer, not yet
accepted, to start Phase 0 plus a concrete `schema.py` and doctrine
draft for review.

Open threads at handoff:
- PR #26 (register modernization + scorer fix) open and mergeable; its
  relevance is reduced but not eliminated by the redirection (the scorer
  fix and the everyday-register lesson carry forward).
- Phase 0 bookkeeping (intent graph, technical report) not started.
- No decision recorded on whether to merge/close PR #26 before starting
  the SGM work.

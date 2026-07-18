# MVP: SGM v0 — a simulator-grounded controller model

This document and the implementation plan below record the project
redirection agreed in conversation: the target is no longer a corpus
generator for a prose-writing model but a NASA-like controller trained
from its corpus (intent nodes 53–56). Old corpora and runs are retained
on disk as baselines only.

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
  1. Tier-1 natural-language mass (below) keeps general language
     in-distribution.
  2. Compositional surface banks: per message type, independent fragment
     slots (entity form × verb form × quantity form × hedge/urgency), giving
     combinatorial surface counts. Built offline by the LLM from
     Architect-authored specs.
  3. Full-paraphrase fraction: 20–30% of rendered messages/documents pass
     through the LLM for free paraphrase (the expensive path, bounded).
  4. Bank refresh between epochs: surfaces drift, structure persists —
     whole-utterance memorization goes stale by design.
- Collapse gauge: n-gram overlap between model completions and bank entries,
  reported per run. Rising overlap with falling general-language scores
  triggers a larger paraphrase fraction, not more bank entries.

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
  final distribution sets the default register.
  Stage runs until stopping criteria, not a token budget.
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
4. Decline: questions the episode does not answer ("not stated" discipline).
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

---

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
  determinism, counterfactual divergence — perturbed state ⇒ different gold
  call in ≥N% of re-rolls), 1k episodes.
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
  register-discipline scorer (field checks + brevity budgets, programmatic).
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

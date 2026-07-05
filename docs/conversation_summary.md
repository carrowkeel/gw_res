# Conversation summary: referent-free SLM pipeline

This document records the decisions made in the working session that built
and iterated on this pipeline, in chronological order, with the reasoning
behind each. It is written for an LLM continuing the work without access to
the original conversation. For the current state of the code and the
concrete next steps, see `docs/technical_report.md`. For the crystallized,
navigable form of this same content, see `.intent/project/`.

## 1. Initial scope

The request was an end-to-end pipeline to train a sub-1B-parameter language
model from scratch on synthetic English text that is fluent but carries no
identifiable real-world referents, so the model cannot state or hallucinate
real facts. Generation runs an existing instruct model (Qwen2.5-7B-Instruct)
locally via vLLM on L40S GPUs; the rest of the pipeline runs on Slurm.
Six stages were defined: generate, tokenizer, data, pretrain, finetune,
evaluate. Knowledge absence was scoped from the start as a controllable
degree, not an absolute guarantee.

## 2. Severity and text types

Severity was introduced as the dial on referent removal: S1 (default) bans
proper nouns, numbers, dates, and named real entities while allowing generic
common nouns; S2 additionally prefers category-level terms and rejects
capitalized proper-noun phrases. A third rung, S3 (a persistent, corpus-wide
invented lexicon), was scoped from the start but deferred as out of reach for
a first pass.

Five text types were discussed; four were selected for implementation as
tractable with a mid-size instruct model: prose (T1), conversational dialogue
(T2a), dictionary-style definitions using relational notation rather than real
objects (T3), and dry descriptive documents (T4, e.g. "the wood stands above
the stream"). Two reasoning-heavy types were deferred as needing stronger
generator models: argumentative or legal-style dialogue (T2b) and
notation-based logic or puzzle documents (T4-logic). A bounded real-domain
signal-injection experiment (T5, introducing a single dense real but
self-contained system, prototyped on the Apollo Lunar Module) was also
deferred, framed as testing whether a bounded real domain can be introduced
without disrupting the model, by analogy to injecting a signal without sharp
amplitude changes.

## 3. From-scratch and referent-free mechanics

Three principles were fixed early: random weight initialization (no
pretrained weights, no inherited knowledge), a fresh BPE tokenizer trained
only on the synthetic corpus (so the vocabulary itself cannot encode unseen
real tokens), and referent stripping as removing identifying detail, not
richness (a generic river is fine, a named river is not). Abstract relational
notation ("A relates to B; when B and C are active, D is active") was adopted
for definitions and technical content, giving documents something to talk
about without naming real things. Consistency of invented terms was scoped
per-document only in the first pass; a global lexicon was deferred to S3.

## 4. First smoke run and filter hardening

The first smoke run surfaced assistant self-reference ("I am a conscious
agent, part of a language model...") in generated text, which the filter had
no notion of. This was fixed by adding meta-phrase detection and by
strengthening the generation system prompt against assistant framing and
preamble, backed by few-shot exemplars per text type so the generator returns
only the target text.

## 5. Alignment discussion (non-code)

A side discussion addressed whether SFT constitutes an alignment process for
this pipeline. Conclusion: the model can still reason instrumentally about
harmful actions using in-context or referent-free abstractions (for example,
if told what fire is, it could reason about burning things) even without
real-world referents, so referent absence reduces certain risks but is not
itself a safety guarantee. A speculative, deferred experiment was scoped
("intuitive alignment measurement") to measure whether norms acquired
distributionally from the corpus (via the legal or ethical dialogue text
type) are more robust than instructed norms, without expecting a strong
result.

## 6. Cluster deployment problems (first cache-directory issue)

Running on the user's Slurm cluster, HuggingFace and related caches wrote
into the home directory despite exported shell variables, because Slurm
batch jobs do not reliably inherit the submitting shell's environment. Fixed
by baking `mkdir -p` and `export` directly into every job's `--wrap` command
via a `slurm.environment` config map.

## 7. Pilot run: diversity and quality problems

The pilot run (7B generator, ~10k texts) produced text that was fluent but
extremely repetitive and, in the user's words, "stupid" or over-idealized;
dialogues were repetitive and sometimes absurd; dictionary definitions were
poor, including one leaked full real-world optics and geometry content for
"lens" despite passing the filter. Fixes applied: added a technical and
scientific term blocklist, switched the definition prompt to invented
headwords in genus-differentia form, added rotating structural axes (tone,
point of view, length band, spatial and comparative relation vocabulary) to
diversify prompts across all four types, and added `slm.inspect` as a
diagnostic tool reporting per-type yield, length statistics, duplicate rate,
distinct-1/distinct-2 diversity metrics, and any kept text that still trips
the filter. `earth`, `sun`, and `moon` were found to be over-blocked (killing
ordinary nature prose) and were removed from the blocklist, while named
planets stayed blocked.

## 8. Evaluation scoring was unreliable

Early evaluation returned many `None` scores because the score extractor only
matched an exact "keyword: n" line. Fixed with a tolerant multi-pattern
extractor handling verbose judge replies and "X out of 10" or "X/10" forms.

## 9. Scale mismatch and catastrophic overfit

A 60M-parameter model trained on roughly 1.2M tokens overfit catastrophically
(validation loss rising from step 250 onward). Root cause: the model was
far too large for the corpus. Fixed by adding smaller presets (pico, nano,
micro, mini, in addition to poc-60m through poc-1b), an early-stop patience
on validation loss, logging of tokens per non-embedding parameter, and by
building an explicit scaling ladder (`configs/scale/`) that grows model and
corpus together at roughly 20 tokens per non-embedding parameter.

## 10. The "gibberish" measurement artifact

Even after fixing the ratio, evaluation output on the pico-scale run looked
like severe gibberish (repeated tokens, control characters). This was
diagnosed, using a new `slm.sample` tool that prints raw base-model
completions on fixed in-distribution seeds bypassing the eval harness, as a
measurement artifact with three compounding causes: (a) evaluation was
scoring the weak SFT model, not the base pretrained model; (b) it was asking
out-of-distribution real-world factual questions rather than in-distribution
completions; (c) the repetition penalty was applied unbounded over the whole
growing context, which suppresses a small byte-level model's entire
vocabulary once enough of the context has been penalized. Fixed by reworking
`evaluate.py` to score the base pretrained model on in-distribution seeds and
in-world instructions as the primary measures (demoting the real-world
knowledge probe explicitly, since it is out-of-distribution and unreliable
for a tiny model), and by windowing the repetition penalty to the most recent
64 tokens with a default of 1.0 (off).

## 11. A second cache-directory regression, and a durable fix

A later run again wrote caches into the home directory. Root cause this time
was a duplicated `environment:` key under `slurm:` in the user's config file
(the user's populated block plus the shipped placeholder `environment: {}`);
YAML silently keeps the last duplicate key, discarding the populated one.
Fixed by removing the redundant placeholder from shipped configs and adding
a durable, single-root mechanism: `slurm.cache_dir` or the `SLM_CACHE_DIR`
shell variable now derives `HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, and
`TRITON_CACHE_DIR` automatically, so a user does not need to hand-edit the
environment map at all. Precedence: derived-from-root, then already-exported
shell cache variables, then the explicit map (highest).

## 12. The SFT collapse problem and its first fix attempt

Across pilot, nano, and other runs, the separate finetune stage was observed
to catastrophically collapse the tiny model into degenerate, repetitive
output, even though the pretrained base model was fluent. Root cause
identified: the original finetune design used role control tokens
(`<user>`/`<assistant>`-style tokens) that appeared only during finetuning,
never during pretraining, so the model had to learn their meaning from very
few examples while its whole output distribution shifted, with no replay of
the pretraining data to anchor it (catastrophic forgetting in miniature). The
user's explicit decision was to redesign the SFT stage rather than patch it,
approving a co-training approach: render instruction pairs as light
`Question: ... Answer: ...` text (no role tokens) and mix them into the
pretraining corpus itself at a configured token fraction, so the base model
learns instruction-following jointly with fluency; finetune then continues on
the same pairs in the same format with response-only loss, intended only to
sharpen instruction-following without shifting the format.

## 13. A pipeline-defaults regression, and the user's correction

In implementing the co-training redesign, `finetune` was mistakenly dropped
from the default pipeline stages, and evaluation was changed to score only
the pretrain checkpoint. The user's explicit correction: keep finetune in the
default pipeline; evaluate must cover both the pretrained model and the
finetuned model, since the evaluation stage still expected an SFT checkpoint
to exist. This was fixed: `finetune` restored to `DEFAULT_STAGES` in both
`pipeline.py` and `slurm/submit.py`; `render_pair_example` (used by
finetune's `PairDataset`) rewritten to use the same light Question/Answer
format with response-only loss masking, rather than role tokens; `evaluate.py`
reworked with `run_all()` evaluating both `pretrain` and `sft` checkpoints
independently, writing `report_pretrain` and `report_sft`, skipping gracefully
if a checkpoint is absent; the Slurm evaluate command fixed from a hardcoded
`--stage sft` to `--stage both`. Separately (already validated in another
session, preserved through this work), the user added multi-worker
generation: `generate.workers` above one turns the generate stage into a
Slurm job array of single-GPU workers plus a CPU-only merge job that
deduplicates and writes the final corpus.

## 14. The SFT collapse persisted even after the redesign

After the fix in section 13, the nano scale-ladder run still showed the
finetuned checkpoint collapsing into degenerate repetition (grammar 1.7,
literal `::::::::::`-style output) while the pretrain checkpoint stayed
fluent (grammar 8.18). This is now the third confirmed instance of the
separate finetune stage harming a tiny model, and it persists despite
removing role tokens and matching the co-trained format exactly. The
recommendation (stated, not yet implemented as of this summary) is to make
`finetune` an opt-in stage rather than a pipeline default, since co-training
alone already delivers instruction-following at this scale, while leaving
the stage available for anyone testing it at larger scale. See
`docs/technical_report.md` for the precise pending change.

## 15. Theoretical discussion: what would make the corpus reasoning-capable

The remainder of the session was a theoretical discussion, prompted by two
questions: what specifically produces correct arithmetic-like competence
("what gets us 1+1=2"), and what would let a model correctly say "I don't
know what X is" for something genuinely outside its world. This led to
several linked conclusions, each now captured as an intent-graph node (see
`.intent/project/nodes/31.json` through `40.json`):

- **Layered information, not flattened content.** Text carries syntax,
  semantic types, frames or schemas, and bound facts, in that order of
  increasing specificity. Referent stripping should remove only the bound-fact
  layer; removing frames as well produces flat, information-empty text, which
  is a distinct failure from referent leakage and a candidate explanation for
  observed corpus flatness beyond simple repetition.

- **Processor, not oracle.** A referent-free model has no real facts to
  recall by construction, so scoring it on recalling or withholding real
  facts (the existing knowledge probe) tests the wrong capability and is
  known to be unreliable at small scale. The capability that matters is
  in-context binding: given a novel entity and attributes stated only in the
  prompt, can the model correctly use and combine them. This is proposed as a
  new, primary evaluation axis, not yet implemented.

- **Information is consistent novelty.** A corpus carries information only
  where it is simultaneously mutually consistent and non-redundant. Real
  corpora get consistency for free because every document describes the same
  coordinating referent, reality. A synthetic corpus generated by prompting
  an LLM independently per document has no such coordinator: if it tries to
  be specific it risks contradiction, so it is pushed toward generic,
  redundant content instead. This is offered as the mechanistic explanation
  for the corpus monoculture observed since section 7, distinct from (and
  deeper than) a simple prompting-diversity problem.

- **LLM-as-author diversity contraction.** An LLM that decides content, not
  just wording, concentrates on its highest-probability content patterns
  (genre gravity) and can reinforce that concentration across generated
  documents via shared prompt exemplars (recursive collapse). This is framed
  as a structural limit of using an LLM as author, not a fixable prompting
  bug, and explains why the diversity fixes in section 7 only partially
  worked.

- **Construction-solving asymmetry.** Many problem families have a cheap
  generation direction and an expensive solving direction (differentiation
  versus integration, multiplying versus factoring primes, building a maze
  versus solving it). A program can run the cheap direction and pose the
  expensive direction as a question with an exact, program-known ground
  truth, without needing a strong solver to generate or grade it. This
  resolves the apparent paradox that a weak generator LLM should not be able
  to construct problems exceeding a weak solver's ability: the generator
  never needs to solve the hard direction, only to run the easy one and
  record what happened.

- **Program as author, LLM as stylist.** The proposed resolution to both the
  consistency problem and the diversity-contraction problem: a non-LLM
  program samples a persistent world-state (entities, relations, an invented
  lexicon) and a task within it, decides what must appear in a document, and
  knows the correct answer by construction; the LLM receives only a local
  fragment and renders it fluently. Round-trip verification (parsing the
  LLM's rendered output back into structure and discarding mismatches with
  the intended structure) bounds rendering error without requiring the LLM to
  be reliable. This promotes S3 (previously the most extreme, and least
  developed, severity rung) from an isolated vocabulary concern to the
  central coordination mechanism for any reasoning-era or information-bearing
  corpus.

- **Parameters versus context window.** Persistent skills live in parameters
  as circuits: attention heads acting as routers or movers of information,
  MLPs acting as compute or memory, composed through the shared residual
  stream across layers. Depth (layer count) bounds how many sequential
  computation steps a single forward pass can chain; step-by-step reasoning
  (chain-of-thought) externalizes computation into the context window as
  scratch memory when a problem needs more steps than the model has layers.
  Circuits generalize, rather than memorize, only when the training
  distribution is too systematically varied for memorization to be the
  cheaper solution. This gives a mechanistic account of why co-training
  (section 12) outperforms separate finetuning: co-training lets circuits
  form around the instruction format from the start, while finetuning tries
  to redirect circuits that have already committed to a solution during
  pretraining. It also connects the corpus's early perplexity plateau to
  corpus information-narrowness rather than solely to model capacity.

- **Abstract number and counting domain.** A side proposal to reintroduce
  quantity grammar (numerals, plural agreement, quantifier scope, unit-free
  comparison) that S1 currently strips wholesale along with real-world
  numbers and dates. The proposal bans only numbers bound to real-world facts
  (a date, a measured real quantity), not abstract counting over invented or
  generic entities, restoring an important reasoning substrate that the
  current filter over-suppresses as a side effect.

None of the sixth-section conclusions (this section) have been implemented in
code; they are captured as intent-graph nodes 31 through 40, connected to the
pre-existing goal, mechanism, and stage nodes they extend or supersede, so a
future session can pick them up without re-deriving the reasoning.

## 16. This summarization request

The final request in this session was to compact the conversation before it
grew unmanageable: capture the current state and the new theoretical goals in
the `.intent/project/` graph, and write this summary plus a companion
technical report of the code's current state and next steps, so a future LLM
session without this conversation's history has everything needed to
continue.

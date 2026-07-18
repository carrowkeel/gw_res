# Technical report: current code state and next steps

Companion to `docs/conversation_summary.md`. This document describes the
pipeline as it exists in the repository right now, file by file, and the
concrete next steps that follow from the conversation, ranked by how firmly
they have been decided. It assumes no prior conversation context. The
project's crystallized intent (goals, principles, mechanisms, deferred work)
lives in `.intent/project/`; read that graph for the reasoning behind each
decision referenced here (node numbers are given in parentheses).

## Repository layout

```
.intent/project/   crystallized project intent graph
docs/               this report and the conversation summary
src/slm/
  config.py         yaml-driven configuration schema
  seeds.py          topic and structure seed vocabulary
  prompts.py        grounded per-type prompt construction
  filters.py        heuristic contamination filter
  worldgen.py       programmatic world-state: groundings, puzzles, binding tasks
  generate.py       stage 0: vLLM synthesis (multi-worker capable)
  tokenizer.py      stage 1: fresh BPE
  data.py           stage 2: pack corpus, mix instructions, datasets
  model.py          from-scratch decoder
  pretrain.py       stage 3: pretraining loop
  finetune.py       stage 4: supervised finetuning (replay + early stop)
  infer.py          load a checkpoint and sample
  evaluate.py       stage 5: judge, probes, binding
  report.py         one-document run summary
  pipeline.py       local orchestrator
  sample.py         diagnostic: raw base-model completions
  inspect.py        diagnostic: corpus yield, diversity, contamination
  sim/              SGM redirection: schema.py and doctrine.yaml drafts
slurm/
  submit.py             dependency-chained sbatch submitter
  example_stage.sbatch
configs/
  poc.yaml, smoke.yaml, pilot.yaml
  scale/world.yaml, scale/mini.yaml
```

## Current pipeline, stage by stage

### generate (`src/slm/generate.py`)

Runs the configured instruct model (default `Qwen/Qwen2.5-7B-Instruct`)
through vLLM to synthesize `generate.number_of_texts` pretraining documents
across five text types (prose, conversation, definition, description,
reasoning, weighted by `generate.text_type_weights`) and
`generate.number_of_pairs` instruction pairs. Prompts come from `prompts.py`;
**every prompt of every type is grounded in a program-generated fact set**
(node 46): `worldgen.sample_grounding` supplies a consistent world fragment
or a puzzle (transfer arithmetic, invented-unit ratios, order chains) with
its program-derived question, answer, and derivation, and the prompt
instructs the LLM to weave the facts in without contradiction and, where a
problem is worked, to reach the given correct answer. Prose uses fragment
groundings (characters and setting from the named entities), conversation
uses fragments or worked puzzles, definitions define invented units with
exact conversion relations, descriptions describe a fragment, reasoning works
a puzzle to its answer, and instruction pairs must state the facts in the
user turn so every pair is answerable from its own context. Surface variety
comes from rotating structural axes (domain, tone, form, point of view,
length band up to several paragraphs) from `seeds.py`, plus few-shot
exemplars matched to the grounded formats. Output passes through
`filters.py` (`META_PHRASES`, `check_text`/`passes`) and deduplication before
being written to `data/pretrain/shard_*.jsonl` and `data/sft/sft.jsonl`.
There is no round-trip verification of the rendered text against the
grounding yet (node 38); the renderer is trusted in this pass.

**Referent-free content restrictions are currently relaxed** (intent graph
node 41): the system prompt no longer forbids real facts, named entities,
numbers, dates, or technical vocabulary, and the filter no longer blocklists
real-world words or bans digits (the `BLOCKLIST` word list and the digit and
proper-noun-run checks described in earlier revisions of this pipeline have
been removed from `filters.py`). Only assistant-framing and meta-commentary
detection remain in the filter, since those are corpus-quality problems
independent of referent-freedom. `generate.severity` (`s1`/`s2`) still
selects concrete-noun versus category-level prompt vocabulary via
`seeds.entity_pool`, but no longer changes what content is allowed; the two
severities currently produce equivalent restrictiveness. This is a
deliberate, temporary MVP decision: the priority is a working, evaluable
prompt-response model, since other project work depends on having one;
referent-freeness is planned to return through the constructed world-state
generator (node 36), whose grounding half now feeds all generation, rather
than through re-tightening `prompts.py` and `filters.py`, since that original
mechanism is implicated in the diversity-contraction problem (node 35).

When `generate.workers` is greater than one, `slurm/submit.py` turns this
stage into a Slurm job array of single-GPU workers, each generating a
disjoint share via `--worker-count`/`--worker-index`, writing intermediate
output to `data/pretrain_workers/` and `data/sft_workers/`; a CPU-only merge
job (`--merge`) deduplicates across workers and writes the final
`data/pretrain/` and `data/sft/sft.jsonl`. This was implemented and validated
by the user in a separate session and is unchanged by the work described in
this report.

### tokenizer (`src/slm/tokenizer.py`)

Trains a fresh byte-level BPE tokenizer (`SyntheticTokenizer`) only on the
generated corpus, including instruction pairs rendered in the same light
`Question: ... Answer: ...` format used later (via `data.render_instruction`),
so the tokenizer learns the `Question:`/`Answer:` tokens as ordinary corpus
vocabulary. `TokenizerConfig.special_tokens` still lists `<|user|>` and
`<|assistant|>` role tokens; these are vestigial from the pre-redesign role-
token approach (see "Known inconsistency" below).

### data (`src/slm/data.py`)

Tokenizes pretrain shards, wraps each document with `bos`/`eos`, and mixes in
instruction pairs rendered via `render_instruction` (`'Question: %s\nAnswer:
%s'`) at `pretrain.instruction_fraction` of total tokens using
`_mix_instructions`, which upsamples by cycling if there are too few pairs or
downsamples to a subset if there are too many. Shuffles and splits into
`train.bin`/`val.bin` packed memmaps plus `meta.json` (vocabulary size, dtype,
token counts, `instruction_token_fraction`). Also provides `PackedDataset`
(random-window sampling for pretraining) and `PairDataset`/
`render_pair_example` for the finetune stage, which renders each pair with
`INSTRUCTION_PREFIX` (`'Question: %s\nAnswer:'`) and masks the prompt span
with `-100` so only the response contributes to loss.

### pretrain (`src/slm/pretrain.py`)

Trains `model.GPT` from random initialization on the packed corpus (which
already contains the mixed-in instruction pairs), with cosine schedule,
gradient accumulation, mixed precision, `early_stop_patience` on validation
loss, and logging of tokens per non-embedding parameter. This is currently
the primary mechanism by which the model learns instruction-following
(node 30, node 14 in the intent graph): because instruction pairs are
ordinary tokens in the same training stream as pretraining text, the model
does not need a second training phase to acquire the Question/Answer format.

### finetune (`src/slm/finetune.py`) -- bug fixed; consolidated to replay + early stop

Continues from the pretrain checkpoint, training on the pairs via `PairDataset`.
It repeatedly collapsed the model into repetitive, low-grammar output
(`report_sft` grammar 1.7 versus the pretrain model's 8.18). **Root cause found
and fixed:** `render_pair_example` built `labels` aligned to the same index as
the input tokens, but the model computes cross entropy of logits at position i
against target i with no internal shift (the convention the packed pretraining
data follows via `targets = data[start+1:]`). The separate finetune stage was
therefore the only path training a copy objective -- predict the current answer
token from itself -- which produces exactly the repeated-token collapse, and
explains why co-trained instruction following (correctly shifted packed path)
worked while this stage did not. Labels are now next-token targets with the
prompt span masked in `response_only` mode.

**Post-fix measurement (full rung, broadened corpus, updated eval):** the
collapse is gone; the finetuned model produces fluent, instruction-shaped
output. But plain finetuning scored slightly below the base model on every
axis (grammar 4.44 vs 5.62, coherence 3.72 vs 4.84, factual accuracy ~1.4 for
both), consistent with mild residual forgetting; the dominant remaining
failure (semantic drift, unstable referent binding) is shared with the base
model and is a pretraining-scale property, not an SFT problem.

The configured approach is therefore consolidated to one option: the base
finetune config sets `replay_fraction: 0.5` (half the training micro-batches
drawn from the packed pretraining data, countering forgetting) plus
`validation_fraction: 0.05` with `early_stop_patience: 3` (stop on held-out
pair loss), and `finetune.variants` is empty, so each rung runs a single
finetune job. The **variant sweep harness** (node 44) remains fully available:
each `{name, ...overrides}` entry under `finetune.variants` forks from the
same pretrain checkpoint, writes to `checkpoints/sft/<name>`, evaluates into
`report_sft_<name>`, and the submitter fans variants out in parallel — list
variants whenever a comparison is wanted. `loss_mode` supports
`response_only` / `full_sequence`.

### evaluate (`src/slm/evaluate.py`)

`run_all(config)` evaluates the pretrain checkpoint and every finetune variant
independently via `run(config, stage, checkpoint_dir)`, writing
`report_pretrain` and `report_sft` (or `report_sft_<variant>` per configured
variant, via `_sft_targets`) so the effect of each finetuning approach is
directly visible. `_find_checkpoint` looks for `ckpt_best.pt` then
`ckpt_last.pt`; a missing checkpoint causes that stage to be skipped, not to
error. The evaluation is matched to the grounded corpus (realigned after the
first grounded run, whose real-world question sets measured knowledge the
corpus deliberately does not contain and thereby misread a large binding win
as a regression). Each report has five parts, headline first:
`grounded_instructions` supplies facts in the user turn from the same kind
mix the pairs train on and scores exact match against program-derived
answers, per kind (retrieval, comparison, multihop, notstated, transfer,
ratio, order), with a judged coherence axis alongside so degraded English is
visible even when the gold token appears; `binding_probe` (node 33) scores
context-paragraph tasks by exact match, stratified over the four question
categories with per-kind sub-scores -- the notstated sub-score is the
fabrication gauge, measuring declining over inventing when the context does
not contain the answer; `score_completions` judges grammar and coherence over
seeds half cut from rendered world documents and half generic;
`score_instructions` and `accuracy_probe` are the demoted out-of-distribution
generalization section -- real-world instructions and facts, expected low,
with the accuracy probe read as a contamination gauge (how much real-world
fact leaked from the generator) rather than a target. Nothing in a grounded
or binding task is answerable from world knowledge, so those isolate the
processor capability (node 32), and binding remains the coherence gauge for
the graph-experiment gate (node 45). `eval.number_of_binding_tasks` (default
64) sets the task count. `_extract_score` is a tolerant multi-pattern regex
extractor handling verbose judge replies, "X/10", and "X out of 10" forms.
Judge model defaults to `eval.judge_model` or falls back to
`generate.default_model`.

### worldgen (`src/slm/worldgen.py`) -- program-as-author: worlds and puzzles

Implements the working half of the world-state mechanism (node 36):
`sample_world` builds a small world of invented people, places, and objects
whose facts are consistent by construction (residence, workplace, ownership,
and storage are functions; ages carry both a total rank order and consistent
absolute years, sizes a rank order, so every comparison has a unique answer).
Worlds draw their vocabulary from one of eight modern, real-world domain
lexicons (`DOMAINS`: office, clinic, school, cafe, depot, studio, transit,
sports), each with its own places, objects (qualified by color rather than
material), and countable goods, and person names mix ordinary modern first
names (`COMMON_NAMES`) with invented ones. The first broadening attempt used
eight pre-industrial registers, which merely varied the same invented archaic
world; since referent-freeness is a deferred objective, the registers were
replaced with everyday modern ones so the surface reads like the world the
generator LLM writes best while the program still authors all logic. `_fragment`
verbalizes a focus person's neighborhood through varied templates, including
surface forms that invert the stated relation, and states ages either
relatively or as absolute years (so the comparison must sometimes be derived
from numbers). Fragment questions span four categories (`_make_question`):
**retrieval** (residence, workplace, ownership, storage, stated age),
**comparison**, **multihop** (compose two stated facts: an object's owner
plus that owner's residence or workplace), and **notstated** (the gold answer
is `NOT_STATED_ANSWER`; the asked fact is verifiably absent from the
fragment, teaching declining over fabricating -- the failure mode the first
grounded model showed on every context-free question).
`binding_tasks(seed, count)` emits evaluation tasks stratified over the
categories, each tagged with its kind. `sample_grounding(rng, kind,
category)` is the generation feed: a fragment or one of three puzzle kinds,
each with facts, a question, the program-derived answer, and a derivation --
**transfer** (countable-goods arithmetic over domain goods), **ratio** (three
invented units with exact integer conversion factors), **order** (a four-name
comparison chain with direction-inverting surfaces, asking either the
superlative or a transitive pairwise question whose two names are never
adjacent in the chain, so no single stated fact answers it).
`sample_pair_grounding` draws the instruction-pair mix (`PAIR_KIND_MIX`,
including a notstated share). All kinds are verified by independent
recomputation over 500 samples each -- parsing the rendered fact strings and
re-deriving the answer, or checking structural absence for notstated. All
prompt builders in `prompts.py` consume these groundings (node 46), realizing
the construction-solving asymmetry (node 37). Deterministic given the seed;
`python -m slm.worldgen --seed 7` previews tasks. Remaining increments:
round-trip verification of rendered text (node 38) and a persistent world
shared across many documents.

### report (`src/slm/report.py`) -- one-document run summary

Collects a run into `eval/summary.md` plus machine-readable `summary.json`:
model and corpus sizes with tokens-per-non-embedding-parameter, pretrain and
finetune loss curves (read from `checkpoints/*/history.jsonl`, which
`pretrain.py` and `finetune.py` now write at each validation point) with
perplexities, a comparable-loss table scoring every checkpoint (pretrain and
each finetune variant) on the same held-out data -- the corpus validation
stream, where a rise after finetuning indicates forgetting, and the held-out
instruction pairs (the same split finetune used, reproduced from the seed),
where a drop indicates instruction gain -- and the judged plus exact-match
evaluation means per stage from the `report_*.json` files. Finetune
checkpoints now persist `train_loss`, `validation_loss`, and
`best_validation` alongside the weights. Written automatically at the end of
`evaluate.run_all` (failure-tolerant) and regenerable standalone:
`python -m slm.report --config <cfg>`. This document is the intended
paste-into-conversation artifact for reporting a run.

### pipeline (`src/slm/pipeline.py`) and Slurm submitter (`slurm/submit.py`)

`DEFAULT_STAGES` in both currently is
`['generate', 'tokenizer', 'data', 'pretrain', 'finetune', 'evaluate']`.
`slurm/submit.py` resolves per-stage sbatch resource requests from
`config.slurm`, chains stages with `afterok` dependencies, and (via
`effective_environment`/`_environment_prefix`) exports cache directories into
every job's `--wrap` command, since Slurm batch jobs do not reliably inherit
the submitting shell's environment. Precedence, lowest to highest:
`slurm.cache_dir` or the `SLM_CACHE_DIR` shell variable (derives `HF_HOME`,
`XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`), then any of those
variables already set in the submitting shell, then an explicit
`slurm.environment` map. Do not reintroduce a duplicate `environment:` key in
any shipped config; YAML silently keeps only the last occurrence of a
duplicate key, which previously discarded a user's populated cache map.

Each submission is isolated by a run id. `slurm/submit.py` resolves a run id
(a fresh `uuid4` hex fragment by default, or an explicit `--run-id` to target
an existing run) and passes it to `load_config(path, run_id=...)`, which
suffixes `project.out_dir` and rebases `slurm.log_dir` onto the suffixed tree
while leaving `project.corpus_dir` — an input reference to an already-frozen
corpus — untouched. The submitter materializes the fully resolved config at
`<out_dir>/config.resolved.yaml` and points every job's `--wrap` at that file,
so the suffixed paths are baked in once rather than threaded onto each command
line and read consistently by jobs that load the config independently on the
cluster. The consequence is that resubmitting a config never overwrites an
earlier run and independent runs (for example a `mini` train and a full `world`
ladder) can proceed concurrently; rerunning a subset of stages against an
existing run means passing that run's id back with `--run-id`. The scale-world
ladder inherits this automatically, since it derives `world_out` and every
rung's `out_dir`/`corpus_dir` from the already-suffixed `project.out_dir`. The
local `pipeline.py` accepts the same `--run-id` but leaves `out_dir`
unsuffixed by default, since a stable directory suits smoke tests.

A scale ladder that dies partway is finished with `--resume` against its run
id. `submit_world` classifies each rung from artifacts on disk: a rung whose
`eval/report_*.json` exists is complete and skipped; a rung whose
`corpus_<name>` snapshot is frozen (pretrain shards plus the sft file) but
which never finished training runs its stages only, off the existing snapshot;
a rung with no frozen snapshot re-runs generation, merge, and stages. It
deliberately does not re-run a finished rung's merge, because the merge writes
all accumulated worker output uncapped (the per-rung fraction is enforced by
generation timing, not by the merge), so refreezing a smaller rung after the
workers have grown toward the full target would corrupt the nested corpora.
For the same reason a bare `--run-id` into a tree that already holds frozen
rungs is refused, and `--resume` is required to continue it. Independently,
each generation array command is wrapped in a bounded retry loop: because the
generator is resumable and idempotent, a worker whose vLLM engine fails to
initialize (a transient GPU or node fault) retries in place and tops up, rather
than failing the array and stalling the rung's merge and every stage chained
behind it. This addressed a real failure where one worker of sixteen crashed at
engine init, failed the full rung's `merge` via `afterok`, and left the rung's
`slurm_logs` empty with no model built.

### Diagnostics: `slm.sample` and `slm.inspect`

`slm.sample` loads a checkpoint (pretrain by default) and prints raw
completions on fixed in-distribution seeds with a configurable repetition
penalty, bypassing the judge entirely; this was the decisive tool in proving
that "gibberish" evaluation output was a measurement artifact rather than a
model failure. `slm.inspect` reports per-type document counts and generation
yield against target, length statistics, duplicate rate, distinct-1 and
distinct-2 diversity metrics (overall and per type), any kept text that still
trips the referent-free filter, and sft pair statistics. Both should be run
before drawing conclusions from a low-scale run, since the eval judge alone
has repeatedly produced misleading signals at this scale.

### Model (`src/slm/model.py`)

`GPT`/`GPTConfig`, decoder-only with RMSNorm, RoPE, SwiGLU MLP, optional
grouped-query attention via `scaled_dot_product_attention`. `PRESETS`
(`smoke`, `pico`, `nano`, `micro`, `mini`, `poc-60m` through `poc-1b`) are
labeled by measured non-embedding parameter count via
`count_parameters(non_embedding=True)`. `generate()` applies repetition
penalty windowed to the most recent `repetition_window=64` tokens (fixed from
an earlier unbounded-penalty bug that suppressed the entire vocabulary of a
small byte-level model).

## Configuration schema (`src/slm/config.py`)

Dataclass-based, one section per stage plus `project` and `slurm`, loaded
from YAML via `load_config` with strict unknown-key validation at both the
top level and within each section (raises `ValueError` on typos or stray
keys, including duplicate-key situations that YAML has already collapsed
before validation runs). Notable fields for continuing work:
- `pretrain.instruction_fraction` (default 0.1): target token fraction for
  mixed-in instruction pairs during pretraining.
- `pretrain.early_stop_patience` (default 0, meaning disabled): number of
  non-improving validation checks before stopping.
- `eval.repetition_penalty` (default 1.0, i.e. off): must stay windowed in
  `model.py`'s `generate()` if ever changed from 1.0.
- `slurm.cache_dir` / `SLM_CACHE_DIR`: single-root cache derivation.
- `model.preset` plus `None`-default per-field overrides (`number_of_layers`
  etc.): a preset only takes effect because these fields default to `None`
  rather than a concrete number; do not give them non-`None` defaults again,
  that previously silently defeated every preset.
- `generate.workers`: parallel single-GPU generate workers under Slurm.

## Scaling ladder status

The three standalone scale configs (`s0_pico`, `s1_nano`, `s2_micro`), which
each regenerated their own corpus, have been replaced by the progressive
scale-world runner (`configs/scale/world.yaml`, node 43). It defines one full
generation target plus a `scale:` section of rungs
(`pico 0.10, nano 0.25, micro 0.50, full 1.00` by default, each with its own
preset and per-rung training overrides), and the Slurm submitter builds a
progressive DAG: cumulative generation chunks, each frozen into a snapshot
(`runs/world/corpus_<rung>/`) that a rung trains on while the next chunk keeps
generating. See the generate/config/submit sections below and the README's
scaling-ladder section for the mechanics.

The first full ladder run on the broadened corpus completed. Post-fix, the
full-rung (then micro preset, ~6M non-embedding parameters, ~120M tokens)
measurements with the updated evaluation: pretrain grammar 5.62 / coherence
4.84, sft baseline 4.44 / 3.72, factual accuracy ~1.4 for both. The model
learned grammar, register, and instruction format (numbered lists, definition
form), but shows semantic drift: it cannot hold a referent constant across a
sentence ("the ridge is generally smaller than the ridge"), blends answer
templates, and free-associates within domains. This is the expected behavior
at ~6M parameters (~20x smaller than GPT-2-small), a capability ceiling
rather than a bug, and it is shared between pretrain and sft.

In response, the ladder's top rung was changed from repeating micro on the
full corpus to the `mini` preset (~14M non-embedding parameters): with early
stopping governing effective epochs, a larger model trained for several
passes over the same corpus is nearly as good as fresh data up to roughly
four epochs and costs no new generation. `configs/scale/mini.yaml` packages
the same lever for the already-generated corpus (train-only run pointing
`project.corpus_dir` at `runs/world/corpus_full`), which is the immediate
next run:

```bash
python slurm/submit.py --config configs/scale/mini.yaml \
  --stages tokenizer,data,pretrain,finetune,evaluate
```

## Known inconsistency to clean up

`TokenizerConfig.special_tokens` still lists `<|user|>` and `<|assistant|>`
tokens left over from the original role-token SFT design, which was replaced
by the light Question/Answer text format (no special role tokens needed,
since the format is plain text). These tokens are currently unused by
`data.py` or `infer.py`. Leave them only if there is a reason to keep them for
forward compatibility; otherwise they are a candidate for removal, since they
trace back to the abandoned role-token SFT design.

## Result so far: scale buys fluency, not binding (node 47)

The cheap scale-up has run. `configs/scale/mini.yaml` trained the `mini`
preset (about fourteen million non-embedding parameters, 2.4x micro) on the
existing ungrounded `runs/world/corpus_full` with the consolidated finetune
(replay + early stop). Two things resolved and one did not:

- **The finetune pathology is cured.** Finetuning now beats pretraining on
  every axis instead of degrading it. The comparable-loss table shows corpus
  validation loss falling slightly through finetuning (2.079 to 2.070, replay
  fully suppressed forgetting) while held-out pair loss dropped (1.594 to
  1.560, real instruction gain). Capacity plus the non-destructive recipe did
  exactly what they were meant to.
- **Fluency scaled with parameters.** Grammar rose to about 7.0 and coherence
  to about 6.6, roughly a point and a half over micro.
- **Binding did not move.** In-context binding exact-match stayed at 0.03
  (one in thirty-two, chance), the accuracy probe stayed near 1.5, and the
  binding completions do not engage the question at all. Parameters bought no
  context-binding on incoherent data.

One caveat keeps this from being a pure capability verdict: the binding probe
uses worldgen-invented names in a facts-then-question format absent from this
ungrounded corpus, so the number confounds cannot-bind with
never-saw-this-distribution. The grounded corpus removes that confound, so the
grounded ladder gives the first fair reading. Net: the bottleneck has moved
from model capacity to data coherence.

## M1 result: grounding opened the gate; the ceiling is the task family (node 51)

The grounded ladder ran (world-359bf5fe). Binding across the rungs, with
grammar/coherence at pretrain:

| rung | non-emb params | tokens | binding (pre/sft) | grammar | coherence |
|---|---|---|---|---|---|
| pico | 0.43M | 19M | 0.28 / 0.25 | 2.7 | 2.2 |
| nano | 1.77M | 46M | 0.56 / 0.56 | 4.8 | 4.0 |
| micro | 5.98M | 90M | 0.78 / 0.81 | 6.2 | 6.0 |
| full (mini) | 14.16M | 180M | 0.75 / 0.78 | 6.8 | 7.0 |

Readings, in order of importance:

- **The M1 gate opened decisively.** At identical model size, grounded data
  took binding from 0.03 (chance) to 0.78, thirty times past the 0.25 gate,
  with fluency held (coherence 7.0). The binding failure was a property of
  the data, not the scale. Even 0.43M-parameter pico reaches 0.28.
- **Binding plateaus from micro up.** The micro-to-full step spent 2.4x
  parameters and 2x data for zero binding gain (0.78/0.81 to 0.75/0.78)
  while fluency kept rising. The ~0.8 ceiling is task-distribution-limited,
  not scale-limited, which resolves M2's question differently than either
  pre-committed axis: the lever is richer generation, not more of the same.
- **SFT is flat because co-training saturates it.** Pair perplexity at
  pretrain is already ~1.5 by micro; finetune early-stops within a few
  hundred steps with nothing left to teach. The consolidated recipe is
  non-destructive but redundant while SFT data repeats the co-trained mix.
- **Evaluation mismatch misread the result as regression.** The real-world
  instruction/accuracy sets measure knowledge the grounded corpus
  deliberately lacks; asked such questions, the model fabricates fact-worlds
  in perfect trained form (no pair ever showed facts being absent).
  Interactive probing confirmed: form-perfect confabulation with no
  epistemic boundary, self-generated names drifting mid-derivation, and
  zero coverage of degenerate logic.
- **The corpus narrowed to one register.** Every sample lives in the same
  artisan world; the model cannot leave it even from out-of-world seeds.
  Diversity contraction (node 35) returned through worldgen's single
  vocabulary rather than through the LLM.

## Response implemented: broaden generation, differentiate SFT, match the eval

Implemented in this order because generation and evaluation must match, and
generation leads (node 52):

1. **Broadened worldgen**: eight domain lexicons; absolute ages beside ranks;
   four fragment question categories including multihop (compose two stated
   facts) and notstated (gold answer: the facts do not say -- teaching the
   epistemic boundary the confabulation transcript showed missing);
   transitive pairwise order questions. All verified by independent
   recomputation over 500 samples per kind.
2. **SFT differentiation**: pairs carry their task kind;
   `pretrain.instruction_kinds` restricts the co-trained stream (world.yaml
   reserves multihop and notstated for finetuning), so the pretrain-to-sft
   delta becomes a capability measurement instead of a repeat.
3. **Matched evaluation**: grounded instructions as exact-match headline with
   per-kind sub-scores, stratified binding with a notstated fabrication
   gauge, half-grounded completion seeds, and the real-world sets demoted to
   a labeled out-of-distribution section with accuracy read as contamination.

**M2 (node 49, updated)**: rerun the ladder on the broadened corpus first;
the new sub-scores (multihop, notstated) say what scale is actually needed
for. **M3 (node 50)**: unchanged, gated on binding, which now stands at 0.78.

## Second ladder result (world-52c749db) and two corrections

The broadened ladder ran and delivered the mechanism's main claim: **SFT now
adds capability**. At the full rung, grounded exact-match rose 0.28 to 0.60
and binding 0.38 to 0.73 from pretrain to sft, with the reserved kinds doing
the work (multihop 1.0 and notstated 0.75 at sft in grounded instructions) --
the first meaningful pretrain-to-sft delta since co-training was adopted, and
by design a capability gain rather than a distribution repeat. Two
corrections came out of reading the reports:

- **Scorer artifact (fixed).** Exact-match read only the first 120
  characters, but half the pairs train a step-by-step style whose conclusion
  arrives last; verbally correct derivations (a ratio answer ending "one
  hauvel is worth 35 nooutors", a comparison ending "the copper chest is
  larger") scored wrong, understating grounded exact-match (ratio 0.33 and
  retrieval 0.4 at full-sft are floors, not measurements).
  `score_binding_answer` now also accepts the gold leading the final
  sentence, so direct answers and worked derivations both score; the final
  sentence, not a character window, because a wider tail catches the last
  derivation step where a comparison restates the facts with the wrong
  candidate leading.
- **Register correction (this change).** The first broadening produced eight
  flavors of the same invented pre-industrial world -- cosmetic diversity.
  Replaced with modern everyday registers and mixed ordinary names, per the
  standing priority that referent-freeness is deferred.

Judged fluency dipped (grammar 4.9 at full pretrain versus 6.8 on the narrow
corpus), but the instrument changed with it: completion seeds are now half
grounded, and a fact-list continuation of a grounded seed is in-distribution
yet judged poorly as prose. Read fluency across the next run's stages, not
against the previous ladder.

## Smaller follow-ups

- **Extend worldgen further.** Round-trip verification of rendered text
  against the grounding, discarding mismatches (node 38, the missing
  safeguard now that all text flows through the renderer); a persistent
  world shared across many documents (node 36); multi-turn pairs (the chat
  probe showed single-turn training gives cross-turn behavior no chance).
- **Cosmetic cleanup**: remove the vestigial `<|user|>`/`<|assistant|>` tokens
  from `TokenizerConfig.special_tokens`; they are remnants of the abandoned
  role-token design and nothing reads them.

## Redirection: the SGM, a simulator-grounded controller (nodes 53-56)

The project target changed after the second ladder. Both corpus extremes
failed in opposite directions — free LLM writing gave fluency with binding
at chance, and the fully grounded corpus gave binding at 0.73-0.81 while
collapsing general language to schema (the model answered "Hello" with a
fabricated fact) — and the user redirected the goal away from prose-writing
models entirely. The new target (node 53) is a NASA-like mission controller
whose communication, reasoning, delegation, and risk understanding come
from its corpus, not from alignment layers: operational register by
default, graceful degradation to plain conversation, and structured
outputs a program can parse.

The design is recorded in full in `docs/sgm_mvp.md` (the MVP specification
and the six-phase implementation plan) and crystallized in intent nodes
53-56. In brief: a mission-day simulator (SIM) is the sole source of ground
truth, running forward cheaply and adjudicating any state on demand through
Architect-authored executable doctrine; the trained model (SGM, Small Graph
Model) is a Graph-to-Graph function realized in v0 as the existing decoder
over a canonical, invertible serialization of episode graphs; the LLM
(Qwen) supplies language surfaces through offline compositional banks and
never facts or logic; input surfaces vary while output commands are
canonical and schema-parsed (read many, write canonical). The corpus has
three tiers: a fresh unconstrained conversational substrate (T1, never
below a 10-15 percent floor), freshly sampled simulation traffic (T2, a
variable-length stage with metric stopping criteria), and real operational
data (T3) held out as never-trained transfer probes.

All prior corpora and runs are retained on disk as baselines only. The
scale-ladder submitter, run isolation, packing and replay, reserved-kind
SFT, and the report machinery all carry forward unchanged; the novelty
budget is spent on the SIM, the schema, the doctrine, the language layer,
and the evaluation.

Current state of the redirection: Phase 0 (this bookkeeping) is done, and
Phase 1 has two review drafts in the tree — `src/slm/sim/schema.py` (typed
episode-graph nodes and edges, the canonical command vocabulary, and
validation) and `src/slm/sim/doctrine.yaml` (the Architect-authored
decision doctrine draft: thresholds, prioritized rules over system metrics,
and rationale codes). The doctrine file format is interpreted, not
hardcoded: each rule names a metric path, a comparison against a literal or
a named threshold, a canonical command with argument bindings, and a
rationale code, with priority resolving conflicts deterministically.
`dynamics.py`, `episode.py`, and the doctrine interpreter follow once the
schema and doctrine drafts are reviewed.

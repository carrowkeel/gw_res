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
  seeds.py          generic referent-free seed vocabulary
  prompts.py        per-type prompt construction by severity
  filters.py        heuristic contamination filter
  worldgen.py       programmatic world-state: documents and binding tasks
  generate.py       stage 0: vLLM synthesis (multi-worker capable)
  tokenizer.py      stage 1: fresh BPE
  data.py           stage 2: pack corpus, mix instructions, datasets
  model.py          from-scratch decoder
  pretrain.py       stage 3: pretraining loop
  finetune.py       stage 4: supervised finetuning (replay + early stop)
  infer.py          load a checkpoint and sample
  evaluate.py       stage 5: judge, probes, binding
  pipeline.py       local orchestrator
  sample.py         diagnostic: raw base-model completions
  inspect.py        diagnostic: corpus yield, diversity, contamination
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
across four text types (prose, conversation, definition, description,
weighted by `generate.text_type_weights`) and `generate.number_of_pairs`
instruction pairs, at the configured `generate.severity` (`s1` or `s2`).
Prompts come from `prompts.py`, using rotating structural axes (tone, point of
view, length band, spatial and comparative relation vocabulary) from
`seeds.py` to spread content, and a system prompt plus few-shot exemplars to
suppress assistant framing and meta-commentary. Output passes through
`filters.py` (`META_PHRASES`, `check_text`/`passes`) and deduplication before
being written to `data/pretrain/shard_*.jsonl` and `data/sft/sft.jsonl`.

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
severities currently produce equivalent restrictiveness. The definition text
type was changed from inventing headwords to asking for real-word
definitions, so the generator supplies genuine lexical knowledge. This is a
deliberate, temporary MVP decision: the priority is a working, evaluable
prompt-response model, since other project work depends on having one;
referent-freeness is planned to return through a constructed world-state
generator (node 36) rather than through re-tightening `prompts.py` and
`filters.py`, since that original mechanism is implicated in the
diversity-contraction problem (node 35).

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
error. Each report has four parts, the first three spanning the same subject domains
generation uses rather than a single topic (reworked alongside the referent
relaxation and content broadening, node 41, after the seed lists were found
still carrying bank and forest fragments from the old referent-free corpus):
`score_completions` (fixed seeds judged for grammar and coherence, the
`referent_free` axis dropped since it rewarded avoiding real content, which is
now backward), `score_instructions` (`TASK_INSTRUCTIONS` spanning explain,
how-to, compare, define, answer, advise, summarize, rewrite, list, and reason,
judged for coherence and whether the answer follows and is correct), and
`accuracy_probe` (a real-world factual-question probe, renamed and inverted
from `knowledge_probe`: it now scores correctness directly rather than
referent avoidance, and is promoted from demoted-and-unreliable to the most
direct available signal of what finetuning adds over the base pretrained
model, though a model this small is still expected to know very little of
it). The fourth part is `binding_probe` (node 33, implemented): tasks from
`slm.worldgen` state consistent facts about novel invented entities in a
context paragraph and ask one back (ownership, storage, residence, or a
two-way age/size comparison whose surface form may invert the stated
direction); scoring is exact match of the gold name in the head of the
completion, before the distractor for comparisons, with no judge involved.
Nothing in a binding task is answerable from world knowledge, so it isolates
the processor capability (node 32) and serves as the coherence gauge for the
graph-experiment gate (node 45). `eval.number_of_binding_tasks` (default 32)
sets the task count. `_extract_score` is a tolerant multi-pattern regex
extractor handling verbose judge replies, "X/10", and "X out of 10" forms.
Judge model defaults to `eval.judge_model` or falls back to
`generate.default_model`.

### worldgen (`src/slm/worldgen.py`) -- first increment of program-as-author

Implements the first working piece of the world-state mechanism (node 36):
`sample_world` builds a small world of invented people, places, and objects
whose facts are consistent by construction (residence, workplace, ownership,
and storage are functions; ages and sizes are total rank orders, so every
pairwise comparison has a unique answer); `_fragment` verbalizes a focus
person's neighborhood through varied templates, including surface forms that
invert the stated relation ("X is younger than Y" for an older-fact);
`binding_tasks(seed, count)` emits evaluation tasks with program-known
answers and distractors; `world_documents(seed, count)` emits
consistency-bearing documents for future mixing into the pretraining corpus.
Deterministic given the seed. `python -m slm.worldgen --seed 7` previews
tasks. Deliberately template-rendered and single-domain in this pass; the
next increments are the LLM-as-stylist rendering layer with round-trip
verification (node 38), a persistent world shared across many documents,
multi-hop questions, and a configured world-document fraction in the corpus.

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

## Next steps, ranked by decision strength

1. **Run the cheap scale-up on the existing corpus.** Submit
   `configs/scale/mini.yaml` (tokenizer through evaluate, no generation) to
   train the `mini` model on `runs/world/corpus_full` with the consolidated
   finetune (replay + early stop). Compare against the full-rung micro
   results, and watch the new in-context binding exact-match score: it is the
   coherence gauge for the graph-experiment gate (node 45).

2. **Extend worldgen toward the full program-as-author mechanism.** The v1
   module (node 36) is template-rendered, single-domain, and fresh-world-per
   -document. Next increments, in rough order: mix `world_documents` into the
   pretraining corpus at a configured fraction (directly trains the binding
   the probe measures); multi-hop questions (combining two stated facts);
   a persistent world shared across many documents; and the LLM-as-stylist
   rendering layer with round-trip verification (node 38), which is where the
   construction-solving asymmetry (node 37) starts producing reasoning data.

3. **Open the graph-experiment gate when binding clears its floor
   (node 45).** Once binding exact-match rises well clear of the guessing
   floor and holds across rungs, start the graph input/output comparisons
   using the existing graph pipeline unchanged.

4. **Abstract number and counting domain (node 40).** Largely subsumed by
   the referent relaxation (numbers are no longer stripped); the remaining
   idea worth keeping is counting and comparison tasks with program-known
   answers, which fits naturally as a worldgen question type rather than a
   prompt/filter change.

5. **Cosmetic cleanup**: remove the vestigial `<|user|>`/`<|assistant|>`
   tokens from `TokenizerConfig.special_tokens`; they are remnants of the
   abandoned role-token design and nothing reads them.

Item 1 is ready to run. Items 2 through 4 are design-bearing and should have
their scope confirmed before implementation, consistent with the project's
workflow rule that code is written only when explicitly asked for.

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
  filters.py        heuristic referent-leakage filter
  generate.py       stage 0: vLLM synthesis (multi-worker capable)
  tokenizer.py      stage 1: fresh BPE
  data.py           stage 2: pack corpus, mix instructions, datasets
  model.py          from-scratch decoder
  pretrain.py       stage 3: pretraining loop
  finetune.py       stage 4: supervised finetuning (optional, see below)
  infer.py          load a checkpoint and sample
  evaluate.py       stage 5: judge both pretrain and sft checkpoints
  pipeline.py       local orchestrator
  sample.py         diagnostic: raw base-model completions
  inspect.py        diagnostic: corpus yield, diversity, contamination
slurm/
  submit.py             dependency-chained sbatch submitter
  example_stage.sbatch
configs/
  poc.yaml, smoke.yaml, pilot.yaml
  scale/s0_pico.yaml, s1_nano.yaml, s2_micro.yaml
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

### finetune (`src/slm/finetune.py`) -- confirmed harmful at small scale, not yet made opt-in

Continues from the pretrain checkpoint, training on the same pairs and the
same light format via `PairDataset`, with response-only loss. This stage is
still a default pipeline stage in both `pipeline.py` and `slurm/submit.py`.
It has empirically collapsed the model into repetitive, low-grammar output in
three independent runs even after the redesign that removed role tokens
(node 30): an early 60M-parameter run, the nano scale-ladder run
(`report_pretrain` grammar 8.18 versus `report_sft` grammar 1.7), and a
pattern consistent with the same collapse in the micro run. **This is the
most concrete pending code change**: make `finetune` an explicit opt-in
stage (remove it from `DEFAULT_STAGES` in `src/slm/pipeline.py` and
`slurm/submit.py`, keep it runnable via `--stages ...,finetune,...`), since
co-training alone already delivers instruction-following at this scale. Not
yet implemented because it has not been explicitly approved as a code change
in this session (the session shifted to documentation before circling back
to it); it is a small, mechanical change once approved.

### evaluate (`src/slm/evaluate.py`)

`run_all(config)` evaluates the pretrain checkpoint and, if present, the sft
checkpoint independently via `run(config, stage)`, writing `report_pretrain`
and `report_sft` (`.json` and `.md`) so the effect of finetuning is directly
visible. `_checkpoint_for` looks for `ckpt_best.pt` then `ckpt_last.pt` under
`config.pretrain_dir` or `config.sft_dir`; a missing checkpoint causes that
stage to be skipped, not to error. Each report has three parts:
`score_completions` (in-distribution seed completions judged for grammar,
coherence, referent-freedom), `score_instructions` (in-world instructions
judged for coherence and instruction-following), and `knowledge_probe` (a
real-world factual-question probe, explicitly labeled unreliable and demoted
in the written report, since a tiny model answers out-of-distribution
questions poorly regardless of how referent-free it is). `_extract_score` is
a tolerant multi-pattern regex extractor handling verbose judge replies,
"X/10", and "X out of 10" forms. Judge model defaults to
`eval.judge_model` or falls back to `generate.default_model`.

Pending, not yet implemented: an in-context binding evaluation axis
(node 32, node 33) -- present a novel invented entity with attributes stated
only in the prompt, then ask comprehension or inference questions requiring
those attributes to be combined, scored for correctness rather than judged
subjectively. This is argued to be a more meaningful primary metric than the
existing knowledge probe, since it tests the presence of a genuine capability
(binding) rather than only the absence of one (real-world recall). It has no
existing code to build on and needs either hand-written task templates or the
world-state generator described below.

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

`configs/scale/s0_pico.yaml` (0.43M non-embedding params, ~75k texts, ~9M
tokens), `s1_nano.yaml` (1.8M params, ~300k texts, ~35M tokens), and
`s2_micro.yaml` (6.0M params, 1M texts, ~120M tokens) target roughly 20
tokens per non-embedding parameter. Results so far: pico and nano both ran
end-to-end; the base pretrained model is fluent and correctly avoids stating
real facts at both scales, and the separate sft checkpoint collapsed at
nano (see above). Micro's interactive session showed coherent, referent-free
completions from the pretrain model; no separate sft report was surfaced by
the user for micro, so its sft behavior is unconfirmed but should be assumed
consistent with pico and nano until shown otherwise.

## Known inconsistency to clean up

`TokenizerConfig.special_tokens` still lists `<|user|>` and `<|assistant|>`
tokens left over from the original role-token SFT design, which was replaced
by the light Question/Answer text format (no special role tokens needed,
since the format is plain text). These tokens are currently unused by
`data.py` or `infer.py`. Leave them only if there is a reason to keep them for
forward compatibility; otherwise they are a candidate for removal alongside
the finetune-default change, since both trace back to the same historical
design.

## Next steps, ranked by decision strength

1. **Make `finetune` opt-in, not default.** Concretely decided in direction
   (three confirmed collapse instances, explicit recommendation stated), not
   yet implemented in code. Change: remove `'finetune'` from `DEFAULT_STAGES`
   in `src/slm/pipeline.py` and `slurm/submit.py`; keep the stage's code and
   Slurm command construction unchanged so it remains runnable via an
   explicit `--stages` list. Update `README.md`'s stage table and pipeline
   description accordingly.

2. **In-context binding evaluation axis.** Theoretically well-motivated
   (node 32, node 33) and the clearest next validation improvement, but needs
   design work: either hand-author a small set of template tasks (novel
   invented entity, stated attributes, a question requiring combining them,
   with a known correct answer for automatic scoring instead of judge
   scoring), or build it on top of item 4 below. Not yet started.

3. **Abstract number and counting domain (node 40).** A scoped, bounded
   change to `filters.py` and `prompts.py`: distinguish real-world bound
   quantities (dates, measured real facts, still banned) from abstract
   counting and comparison over invented or generic entities (currently
   over-suppressed by S1's blanket number ban). Would restore quantity
   grammar as a reasoning substrate. Not yet started; needs a concrete
   design for what counts as "abstract" versus "bound" before implementation.

4. **Persistent world-state generator (node 36), program-as-author
   architecture (node 38), and construction-solving asymmetry (node 37).**
   The largest and least concrete pending item: a new, non-LLM `worldgen`
   stage that samples a persistent world-state (entities, relations, an
   invented lexicon) and tasks within it, uses the construction-solving
   asymmetry to generate problems harder than the generator LLM could solve
   itself, hands the LLM only local fragments to render fluently, and
   round-trip-verifies rendered output against the intended structure,
   discarding mismatches. This would supersede per-document local consistency
   (node 10) for reasoning-era text, promote S3 (node 25) from an isolated
   severity rung to the corpus's coordination mechanism, resolve the
   information-is-consistent-novelty problem (node 34) and the LLM-as-author
   diversity contraction problem (node 35), and supply ground-truth-bearing
   material for item 2. This is a substantial redesign of the generate stage
   and has not been scoped into concrete modules or file changes yet; it
   should be designed before being implemented, likely starting with a single
   narrow world-state domain (for example, spatial or relational facts among
   a handful of invented entities) rather than a general system.

5. **Cosmetic cleanup**: remove the vestigial `<|user|>`/`<|assistant|>`
   tokens from `TokenizerConfig.special_tokens` once finetune's status is
   settled (item 1), since both are remnants of the pre-redesign role-token
   approach and neither is read anywhere in the current codebase.

Items 1 and 5 are small, mechanical, and low-risk once approved. Items 2
through 4 are open-ended design work motivated by the theoretical discussion
in `docs/conversation_summary.md` section 15 and the intent-graph nodes it
produced; none should be started without confirming scope first, consistent
with the project's stated workflow rule that code is written only when
explicitly asked for.

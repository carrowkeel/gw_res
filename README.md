# Synthetic-data SLM (referent-free)

An end-to-end, parameterized pipeline for training sub-billion-parameter
language models from scratch on synthetic, referent-free English text. The aim
is a model that is fluent in English but carries no identifiable real-world
referents, so it cannot state or hallucinate real facts. Knowledge absence is
treated as a controllable degree, not an absolute.

Generation runs existing models locally on L40S GPUs; jobs are submitted via
Slurm.

The crystallized intent for the project (goals, principles, parameters, text
types, and the deferred experiments) lives in the intent graph at
`.intent/project/`. Read it before making structural changes.

> Status: authored for execution on GPU and Slurm infrastructure. The two
> GPU-only stages (generate, evaluate) have not been run here. The four
> non-GPU stages run end-to-end on CPU via the smoke config. Referent
> stripping is currently relaxed in generation for an MVP push toward a
> functional prompt-response model; see "Referent-free design" below and the
> intent graph.

## Stages

| Stage | Module | GPU | Output |
|-------|--------|-----|--------|
| generate | `slm.generate` | yes (vLLM) | `data/pretrain/*.jsonl`, `data/sft/sft.jsonl` |
| tokenizer | `slm.tokenizer` | no | `tokenizer/tokenizer.json` |
| data | `slm.data` | no | `data/packed/{train,val}.bin`, `meta.json` |
| pretrain | `slm.pretrain` | yes | `checkpoints/pretrain/ckpt_{best,last}.pt` |
| finetune | `slm.finetune` | yes | `checkpoints/sft/ckpt_last.pt` |
| evaluate | `slm.evaluate` | yes (vLLM) | `eval/report_{pretrain,sft}.{json,md}` |

All artifacts land under `project.out_dir`, for example `runs/poc/`.

Instruction following is learned twice over. First it is co-trained during
pretraining: the generated prompt and response pairs are rendered in a light
format (`Question: ... Answer: ...`) and mixed into the pretraining corpus at
`pretrain.instruction_fraction` of the tokens, so the base model already follows
instructions. The finetune stage then continues training on those same pairs in
the same format, computing loss on next-token targets with the prompt span
masked (`loss_mode: response_only`), sharpening instruction following.
Evaluation covers the base model and each finetune variant, writing
`report_pretrain` and one `report_sft` (or `report_sft_<variant>` per variant),
so the effect of finetuning is visible.

The finetune stage supports a sweep of variants, each forked from the same
pretrain checkpoint, so several approaches can be compared cheaply against one
pretraining run. A variant is a `{name, ...overrides}` entry under
`finetune.variants`, and the knobs are `loss_mode` (`response_only` or
`full_sequence`), `replay_fraction` (mix that fraction of pretraining batches in
to counter forgetting), `validation_fraction` with `early_stop_patience` and
`evaluation_interval` (stop on held-out pair loss), and any optimization field
(`learning_rate`, `epochs`, ...). With no variants a config runs a single
finetune to `checkpoints/sft/`, as before; with variants each writes to
`checkpoints/sft/<name>/` and the submitter fans them out in parallel.

## Referent-free design

> Currently relaxed: generation's referent-stripping rules (no proper nouns,
> numbers, dates, real facts, or technical vocabulary) are switched off in
> `prompts.py` and `filters.py` so the pipeline produces an ordinary,
> knowledgeable prompt-response model for an MVP push, since other work
> depends on having a working small language model. The mechanism below is
> the design to reinstate, planned to happen through a constructed world-state
> generator (see the intent graph) rather than through prompt and filter
> restriction. `severity` still selects concrete-noun (`s1`) versus
> category-level (`s2`) prompt vocabulary, but neither currently restricts
> content.

- Train from scratch (random initialization): no pretrained weights, no
  inherited knowledge.
- Fresh BPE tokenizer trained only on the synthetic corpus: the vocabulary
  cannot encode unseen real-world tokens.
- Referent-free generation (relaxed, see above): a strict system prompt at the
  chosen severity that forbids assistant framing and meta-commentary, a
  few-shot exemplar per text type so the generator returns only the text with
  no preamble, and a best-effort filter that drops texts with digits, urls,
  blocklisted real or technology words, or assistant and meta phrases (for
  example self-reference to being a model, or to the act of writing).

Severity is the dial on the degree of referent removal:

- `s1` (default): no proper nouns, numbers, dates, or named real entities;
  generic common nouns such as forest, lake, animal are allowed.
- `s2`: additionally prefer category-level terms (a creature, a substance, a
  structure) over specific kinds, and reject capitalized proper-noun phrases.

A global invented lexicon (the most extreme rung) is described in the intent
graph and not implemented in this pass.

## Text types

Implemented: prose, conversation, definition, description, and reasoning. Every
prompt is anchored in a sampled subject domain (everyday life, work, science,
history, the arts, relationships, health, technology, food, travel, law,
sport, money, ideas, craft, and the land), so the corpus ranges over real
subjects instead of a single kind of scene. Each type also carries a
structural demand:

- **prose**: a story with a plot (a character who wants something, an obstacle,
  a turning point, an outcome), not static description.
- **conversation**: a multi-turn exchange in which each turn does work and the
  exchange reaches a definite outcome.
- **definition**: real terms defined accurately in genus-and-differentia form.
- **description**: a factual account of a real thing or process, made clear
  through stated relations.
- **reasoning**: strictly ordered explanation or argument (cause before effect,
  steps in order, reasons by weight), so the logic is real rather than poetic.

Instruction pairs span many task kinds (explain, how-to, compare, define,
answer, advise, summarize, rewrite, list, reason), each set in a sampled
domain, so the model learns to perform tasks and not only to describe.

The reasoning-heavy document types that need a stronger generator (legal and
argumentative dialogue, notation-logic and puzzle documents) and the bounded
real-domain signal-injection experiment remain described in the intent graph
as deferred work.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torch
pip install "vllm>=0.5"
```

## Run via Slurm

Submits each stage as a dependent sbatch job; the chain stops if any stage
fails. Resource requests come from the slurm section of the config.

```bash
python slurm/submit.py --config configs/poc.yaml --dry-run
python slurm/submit.py --config configs/poc.yaml
python slurm/submit.py --config configs/poc.yaml --stages pretrain,finetune,evaluate
```

Each GPU job is emitted in the requested form, for example:

```bash
sbatch --job-name slm-generate --mem 64G --cpus-per-task 8 --gres gpu:l40s:1 \
  --time 24:00:00 --parsable --output runs/poc/slurm_logs/%x-%j.out \
  --wrap "cd <repo> && PYTHONPATH=src python3 -m slm.generate --config configs/poc.yaml"
```

`slurm/example_stage.sbatch` runs a single stage by hand.

### Parallel generation across GPUs

Set `generate.workers` above one and the submitter turns the generate stage
into a Slurm job array of that many single-GPU workers, each generating a
disjoint share of every text type and of the pairs with a worker-specific
prompt seed. A CPU-only `slm-generate-merge` job runs after the array,
deduplicates across workers, and writes the final `data/pretrain/` shards and
`data/sft/sft.jsonl`; later stages depend on the merge job. Workers write
intermediate output to `data/pretrain_workers/` and `data/sft_workers/`.

Each worker also compiles into its own Triton, Inductor, and vLLM cache
subdirectory (derived from the configured cache locations), so workers sharing
a node do not race on the same compiled kernel files; the HuggingFace download
cache stays shared so model weights are fetched once. A worker can also be run
by hand:

```bash
PYTHONPATH=src python3 -m slm.generate --config configs/poc.yaml \
  --worker-count 8 --worker-index 3
PYTHONPATH=src python3 -m slm.generate --config configs/poc.yaml --merge
```

Generation is resumable and fails loudly on a shortfall. Each worker counts the
output it already wrote and generates only the missing amount, so a worker that
fails partway (out of memory, preemption) is topped up rather than restarted. A
worker that cannot reach its target within the per-run attempt cap exits
non-zero, and the merge refuses to run until every worker has produced its full
share, naming any worker that is short. Because the merge depends on the array
via `afterok`, a failed worker leaves the merge and later stages pending; to
recover, resubmit the same command once the cause is addressed:

```bash
python slurm/submit.py --config configs/scale/world.yaml
```

Completed workers detect their finished output and exit immediately without
reloading the generator, so only the failed worker regenerates. (Cancel the
original pending merge job first, since its dependency can no longer be met.)

### Redirecting caches off the home directory

Batch jobs do not reliably inherit your login-shell environment, so exporting
cache variables before submitting is not enough. The submitter exports the
cache variables (and creates their directories) inside each job, independent of
cluster propagation policy. Set the cache location once, in any of three ways
(later ones override earlier ones):

1. Set a single root once and forget it, with no per-config edits:

   ```bash
   export SLM_CACHE_DIR=/your/lab/slm_cache   # put this in your shell profile
   ```

   The submitter derives `HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, and
   `TRITON_CACHE_DIR` under that root for every run. The same works per config
   via `slurm.cache_dir: /your/lab/slm_cache`.

2. Any of the known cache variables already exported in your submitting shell
   (`HF_HOME`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, and the
   other HuggingFace and torch cache variables) are forwarded into the jobs.

3. Full control per config via an explicit map:

   ```yaml
   slurm:
     environment:
       HF_HOME: /your/lab/hf_cache
       VLLM_CACHE_ROOT: /your/lab/cache/vllm
   ```

Note that vLLM uses `VLLM_CACHE_ROOT` and does not read `XDG_CACHE_HOME`, so it
must be set explicitly (the single-root option above does this for you). Confirm
what a job will export with `python slurm/submit.py --config <cfg> --dry-run`.

## Run locally

```bash
PYTHONPATH=src python -m slm.pipeline --config configs/poc.yaml
PYTHONPATH=src python -m slm.pipeline --config configs/smoke.yaml \
  --stages tokenizer,data,pretrain,finetune
```

## Scaling up: the pilot tier

`configs/pilot.yaml` sits between smoke and poc: the real 7B generator, about
ten thousand texts, and a 60M model. It is meant to validate 7B generation
quality and yield, and to get a first real signal that a small model learns
fluent English, before committing to a poc-scale run. Run generation first,
inspect the corpus, then train:

```bash
python slurm/submit.py --config configs/pilot.yaml --stages generate
PYTHONPATH=src python -m slm.inspect --config configs/pilot.yaml
python slurm/submit.py --config configs/pilot.yaml \
  --stages tokenizer,data,pretrain,finetune,evaluate
```

`slm.inspect` reports per-type counts, generation yield against the target,
length statistics, and any kept text that still trips the referent-free filter,
so the "look at the data" step is one command.

## Scaling ladder: the scale-world runner

`configs/scale/world.yaml` defines one growing corpus and a ladder of models
trained on nested fractions of it. Because every fraction of the corpus is the
same quality, the smaller corpora are prefixes of the larger one, so no data is
ever regenerated per rung: a rung just trains on a snapshot of the corpus so
far. The config sets the full (largest) generation target and model, plus a
`scale:` section listing rungs, each a `{name, fraction, + per-rung overrides}`:

| Rung | fraction | preset | approx texts | approx tokens |
|------|----------|--------|--------------|---------------|
| pico | 0.10 | pico | 100k | ~12M |
| nano | 0.25 | nano | 250k | ~30M |
| micro | 0.50 | micro | 500k | ~60M |
| full | 1.00 | micro | 1M | ~120M |

The submitter detects the `scale` section and runs a **progressive** ladder.
Generation proceeds in cumulative chunks (the resumable generator tops up to
each rung's target); as each chunk is merged into a frozen snapshot
(`runs/world/corpus_<rung>/`), that rung trains a model on the snapshot while
the next chunk keeps generating. Models are therefore built while data is still
being generated, and because data generation is the most expensive step, you
can stop the whole process early if a rung looks wrong:

```bash
python slurm/submit.py --config configs/scale/world.yaml --dry-run
python slurm/submit.py --config configs/scale/world.yaml
# inspect a rung's frozen corpus snapshot at any point
python -m slm.inspect --config runs/world/pico/config.yaml
# stop early if a rung reveals a problem
scancel --name slm-gen-nano --name slm-merge-nano   # and later rung jobs
```

Each rung writes its own tokenizer, packed data, checkpoints, and evaluation
reports under `runs/world/<rung>/`; the submitter materializes each rung's
resolved config there. `world.yaml` also defines a `finetune.variants` sweep
(baseline, replay, early-stopping, full-sequence loss, low learning rate), so
every rung runs all variants against its one pretrain checkpoint and evaluates
each, yielding a `report_sft_<variant>` at every scale. That is the cheap,
repeated signal used to iterate on the finetuning stage across both model size
and approach in a single run. The pretrain log prints
`tokens per non-embedding parameter` per rung so the size ratio stays visible.

## Graph-context pipeline (experimental)

A second pipeline tests whether a graph-structured context carries
conversation state more token-efficiently than flat text. It reuses the
generated corpus and the flat pretrained model of a run as its baseline and
adds five stages that never modify the base artifacts:

| Stage | Module | GPU | Output |
|-------|--------|-----|--------|
| graph_transform | `slm.graph_transform` | no | `data/graphs/{graphs,holdout}.jsonl` |
| graph_tokenizer | `slm.graph_tokenizer` | no | `tokenizer/graph_tokenizer.json` |
| graph_data | `slm.graph_data` | no | `data/graph_packed/{train,val}.bin` |
| graph_pretrain | `slm.graph_pretrain` | yes | `checkpoints/graph_pretrain/ckpt_{best,last}.pt` |
| graph_evaluate | `slm.graph_evaluate` | yes (vLLM judge) | `eval/report_graph.{json,md}` |

The transform stage segments each generated text (speaker turns for
conversations, grouped sentences otherwise) and folds the segments into a
per-text context graph with two moves: extend the most lexically related
node, or add a new node rooted under node zero when nothing related exists.
An extension that pushes a node past `graph.node_token_limit` splits the
overflow into a child node; there is no rebalancing or merging, so growth is
append-mostly. The graphs are trees by construction and use the write2
intent-graph storage layout for interchange (a few examples are exported
under `data/graphs/intent_examples/`).

Training examples are the depth-first linearization of the graph built from
a prefix of the segments (structural markers are reserved tokenizer tokens),
followed by a next marker and the raw following segment. The graph model is
pretrained from scratch on these with the same loop and preset as the flat
model. Evaluation compares the two on held-out conversations at matched
context-token budgets: the flat model gets the most recent transcript tokens,
the graph model gets the folded graph reduced to the budget by dropping the
leaf subtrees least related to the latest turn, and a judge scores each
continuation for coherence and consistency.

Run it after (or alongside) the base pipeline of the same config:

```bash
python slurm/submit.py --config configs/poc.yaml \
  --stages graph_transform,graph_tokenizer,graph_data,graph_pretrain,graph_evaluate
```

Key parameters live in the `graph` section: `segment_tokens`,
`node_token_limit`, `relatedness_threshold`, `examples_per_text`,
`context_dropout`, `holdout_fraction`, `context_budgets`,
`number_of_eval_conversations`, and `judge_enabled`. Held-out conversations
are excluded from graph training but remain in the flat corpus, which biases
the comparison against the graph model, not for it.

## Multi-GPU pretraining

Set `slurm.pretrain_gres: gpu:l40s:4` (the submitter switches that stage to
`torchrun --nproc_per_node=4`), or launch directly:

```bash
torchrun --standalone --nproc_per_node=4 -m slm.pretrain --config configs/poc.yaml
```

## Key parameters

- `generate.number_of_texts`, `generate.number_of_pairs`: corpus size.
- `generate.workers`: parallel single-GPU generate jobs under Slurm.
- `generate.default_model`, `generate.type_models`: generator routing per type.
- `generate.severity`: referent-removal degree (`s1`, `s2`).
- `generate.text_type_weights`: relative amount of each text type.
- `tokenizer.vocabulary_size`: fresh BPE vocabulary size.
- `model.preset`: `poc-60m` through `poc-1b`, default `poc-150m`.
- `pretrain.*`, `finetune.*`: optimization and schedule.
- `eval.judge_model`: model that scores and interrogates the student.
- `slurm.*`: `gres`, `memory`, `cpus_per_task`, `time_limit`, `pretrain_gres`.

## Evaluation

Evaluation runs on the pretrained model and every finetune variant, writing a
separate report for each (`report_pretrain` and `report_sft` or
`report_sft_<variant>` per variant); a stage with no checkpoint is skipped. An
existing model judges three ways, all over seeds and instructions that span
the same subject domains generation uses (everyday life, work, science,
history, arts, relationships, health, technology, and more), not a single
topic: completions from fixed seeds, scored for grammar and coherence;
task instructions (explain, how-to, compare, define, answer, advise,
summarize, rewrite, list, reason), scored for coherence and whether the
answer follows the request, correctly where it has a factual or practical
answer; and a factual accuracy probe on fixed real-world questions, scored for
correctness. Since generation now targets a knowledgeable model, the accuracy
probe is a direct signal of what finetuning adds over the base pretrained
model, though a model this small should be expected to know very little of it.

Score parsing tolerates verbose judge replies, the completion rubric forces low
grades for text that is not well-formed English, and the report names the judge
model. Scores are only meaningful with a capable judge; the smoke config uses a
small judge for plumbing and its scores should not be trusted. The repetition
penalty defaults off and is windowed to recent tokens when used, so it cannot
suppress the whole vocabulary of a small byte-level model.

To read raw completions directly, use `slm.sample`, which completes
in-distribution seeds with the base pretrained model:

```bash
python -m slm.sample --config runs/world/pico/config.yaml
```

To probe a built model interactively by hand, use `slm.chat`, a prompt loop
over a checkpoint. The pretrain stage continues your text in completion style;
`--stage sft` answers in the Question and Answer framing. Sampling settings are
adjustable at runtime (`/temp`, `/topp`, `/penalty`, `/tokens`), and `/stage`
switches between the two models in place. These are tiny models, so this runs
fine on CPU; grab an interactive allocation rather than the login node:

```bash
srun --pty python -m slm.chat --config runs/world/pico/config.yaml
```

## Layout

```
.intent/project/   crystallized project intent graph
src/slm/
  config.py        yaml-driven configuration schema
  seeds.py         generic referent-free seed vocabulary
  prompts.py       per-type prompt construction by severity
  filters.py       heuristic referent-leakage filter
  generate.py      stage 0: vLLM synthesis
  tokenizer.py     stage 1: fresh BPE
  data.py          stage 2: pack corpus, datasets
  model.py         from-scratch decoder
  graph.py             context graph structure, fold moves, linearization
  graph_transform.py   graph stage 1: texts to context graphs
  graph_tokenizer.py   graph stage 2: BPE with structural marker tokens
  graph_data.py        graph stage 3: pack linearized graph examples
  graph_pretrain.py    graph stage 4: pretrain the graph-context model
  graph_evaluate.py    graph stage 5: flat versus graph at matched budgets
  pretrain.py      stage 3: pretraining loop
  finetune.py      stage 4: supervised finetuning
  infer.py         load a checkpoint and sample
  chat.py          interactive prompt loop over a checkpoint
  evaluate.py      stage 5: judge, probe, interrogation
  pipeline.py      local orchestrator
slurm/
  submit.py            dependency-chained sbatch submitter
  example_stage.sbatch
configs/
  poc.yaml, smoke.yaml
```

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

All artifacts land under `project.out_dir`, for example `runs/poc/`. The Slurm
submitter suffixes this with a per-run id (see Run isolation and reruns) so
runs never overwrite each other, for example `runs/poc-a1b2c3d4/`.

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

The configured finetuning approach combines pretraining-data replay
(`replay_fraction: 0.5`, drawing half the training micro-batches from the
packed pretraining data to counter forgetting) with validation early stopping
(`validation_fraction`, `early_stop_patience`, `evaluation_interval`), since
plain finetuning measured slightly below the base model on every axis. The
stage also supports a sweep of variants when a comparison is wanted: a variant
is a `{name, ...overrides}` entry under `finetune.variants`, overriding any
finetune field including `loss_mode` (`response_only` or `full_sequence`).
With no variants a config runs a single finetune to `checkpoints/sft/`; with
variants each writes to `checkpoints/sft/<name>/`, the submitter fans them out
in parallel, and each is evaluated into `report_sft_<name>`.

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

Implemented: prose, conversation, definition, description, and reasoning.
**Every prompt of every type is grounded in a program-generated fact set from
`slm.worldgen`**: the program authors the logic (a consistent world fragment,
or a puzzle whose answer it derived by construction) and the LLM only writes
it up in the requested register. The same grounding can surface as any type -
a transfer puzzle as a dialogue or a reasoning piece, a world fragment as a
story or a description. The rationale: a small model can only learn patterns
that are actually there, and text whose internal logic is loose or
contradictory gives it nothing stable to learn, so all generated text is
anchored to facts that cannot contradict each other, with correct answers
supplied to the writer rather than left for it to compute.

Grounding kinds: **fragment** (a small relational world of invented people,
places, and objects), **transfer** (countable-goods arithmetic), **ratio**
(invented units with exact conversion factors), **order** (a comparison chain
with direction-inverting surface forms and transitive pairwise questions).
Worlds are sampled from eight modern, real-world domain lexicons (offices,
clinics, schools, cafes, warehouses, studios, transit, sports), each with its
own places, objects, and goods, and people mix ordinary first names with
invented ones - the absence of real-world referents is a deferred objective,
so surfaces read like everyday life rather than an invented archaic world,
and the corpus does not collapse into one register; fragments state
ages either relatively or in absolute years (so comparisons must sometimes be
derived from numbers), and fragment questions span four categories:
**retrieval** (read one stated fact back), **comparison**, **multihop**
(compose two stated facts, e.g. where the owner of an object lives), and
**notstated** (the correct answer is that the facts do not say - the pairs
teach declining over fabricating). Per type:

- **prose**: a story with a plot whose characters and setting are the people
  and places named in the grounding facts.
- **conversation**: either two people from a fragment handling its facts, or
  two speakers working a puzzle through to its (given) correct answer.
- **definition**: dictionary entries for invented units of measure with exact
  conversion relations, in genus-and-differentia form.
- **description**: a dry factual account of a fragment's people, places, and
  things, made clear through stated relations.
- **reasoning**: state a puzzle's facts, pose its question, and work in strict
  order to the given correct answer.

Instruction pairs are grounded the same way: the user message must state the
facts and ask the question, so the pair is answerable from its own context,
and the response is fixed to the program-derived answer, so no pair can teach
a wrong conclusion. Each pair carries its task kind, and
`pretrain.instruction_kinds` can restrict the co-trained instruction stream
to a subset of kinds: the excluded kinds (in `world.yaml`, multihop and
notstated) are then seen only by the finetune stage, which always trains on
all pairs, so the pretrain-to-sft delta measures a capability finetuning
actually added rather than a distribution co-training already taught.
Surface content stays varied through sampled domains, tones, forms, and
lengths (documents now run up to several paragraphs).

The LLM-as-stylist layer does not yet round-trip-verify the rendered text
against the grounding (see the intent graph); the reasoning-heavy document
types needing a stronger generator and the bounded real-domain
signal-injection experiment also remain deferred there.

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

### Run isolation and reruns

Every submission writes into its own tree. The submitter suffixes
`project.out_dir` with a run id — a fresh random id by default — so
resubmitting a config never overwrites an earlier run and two runs (say a
`mini` train and a full `world` ladder) can proceed at once without colliding.
The chosen id and the resolved output tree are printed at the top of the
submission, and the fully resolved config is materialized at
`<out_dir>/config.resolved.yaml`; every job reads that file, so the suffixed
paths are baked in.

To rerun stages against an existing run instead of starting a new one, pass its
id back with `--run-id`:

```bash
python slurm/submit.py --config configs/scale/mini.yaml            # -> runs/world/mini-a1b2c3d4
python slurm/submit.py --config configs/scale/mini.yaml \
  --run-id a1b2c3d4 --stages evaluate                              # reruns eval on that same tree
```

Only outputs are suffixed. `project.corpus_dir` is an input reference to an
already-frozen corpus and is left untouched, so a run can read an existing
corpus (for example `runs/world/corpus_full`) while writing to its own tree.
The local runner (`python -m slm.pipeline`) takes the same `--run-id` flag but
leaves `out_dir` unsuffixed by default, since a stable directory is convenient
for smoke tests.

### Resuming a partial scale ladder

If a scale-world run dies partway — most often a transient worker crash that
fails a rung's generation and leaves later rungs blocked — resubmit the same
config against its run id with `--resume`:

```bash
python slurm/submit.py --config configs/scale/world.yaml --run-id 359bf5fe --resume
```

Resume walks the rungs and, per rung, skips work that is already done: a rung
whose evaluate report exists is skipped entirely; a rung whose corpus snapshot
is frozen but which never finished training runs its stages only, off the
existing snapshot; a rung with no frozen snapshot re-runs generation (which
tops up from durable worker output), merge, and stages. Crucially it does **not**
re-run the finished rungs' merges, which would refreeze the smaller rungs with
the now-larger accumulated worker output and corrupt the nested ladder — for
that reason a bare `--run-id` into a tree that already holds frozen rungs is
refused, and `--resume` is required to continue it.

Generation is also self-healing per worker: each gen array command retries a
few times in its allocation before failing, so a single worker whose vLLM
engine fails to initialize (a transient GPU or node fault) tops itself up
instead of failing the array's `afterok` and stalling the whole rung.

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
recover, resubmit against the **same run** once the cause is addressed — pass
its id with `--run-id` so the topped-up output lands in the original tree
rather than a new one:

```bash
python slurm/submit.py --config configs/scale/world.yaml --run-id <id>
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
| full | 1.00 | mini | 1M | ~120M |

The full rung steps the model up to `mini` rather than repeating `micro` on
more data: with early stopping governing the effective number of epochs,
training a larger model for several passes over the same corpus is nearly as
good as fresh data (returns hold up to roughly four epochs), and it costs no
new generation. The same lever works on an already-generated corpus:
`configs/scale/mini.yaml` points `project.corpus_dir` at an existing snapshot
(`runs/world/corpus_full`) and trains the `mini` model on it with no
generation stage at all:

```bash
python slurm/submit.py --config configs/scale/mini.yaml \
  --stages tokenizer,data,pretrain,finetune,evaluate
```

The submitter detects the `scale` section and runs a **progressive** ladder.
Generation proceeds in cumulative chunks (the resumable generator tops up to
each rung's target); as each chunk is merged into a frozen snapshot
(`runs/world/corpus_<rung>/`), that rung trains a model on the snapshot while
the next chunk keeps generating. Models are therefore built while data is still
being generated, and because data generation is the most expensive step, you
can stop the whole process early if a rung looks wrong:

```bash
python slurm/submit.py --config configs/scale/world.yaml --dry-run
python slurm/submit.py --config configs/scale/world.yaml   # prints the run id, e.g. world-a1b2c3d4
# inspect a rung's frozen corpus snapshot at any point (fill in the printed run id)
python -m slm.inspect --config runs/world-<id>/pico/config.yaml
# stop early if a rung reveals a problem
scancel --name slm-gen-nano --name slm-merge-nano   # and later rung jobs
```

Each rung writes its own tokenizer, packed data, checkpoints, and evaluation
reports under `runs/world-<id>/<rung>/`; the submitter materializes each rung's
resolved config there. Finetuning is consolidated to one configured approach
(pretraining-data replay at half the micro-batches, plus validation early
stopping), so each rung runs a single finetune job; the variant-sweep harness
remains available by listing entries under `finetune.variants` when a
comparison is wanted. The pretrain log prints
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
`report_sft_<variant>` per variant); a stage with no checkpoint is skipped.
The evaluation is matched to the grounded corpus: what the corpus trains is
processing facts given in context, so that is what the headline sections
measure, by exact match against program-derived answers, with no judge noise.

- **Grounded instructions** (headline): tasks drawn from the same kind mix
  the pairs train on - the user message states the facts and asks the
  question, and the answer is exact-match scored per kind (retrieval,
  comparison, multihop, notstated, transfer, ratio, order). The notstated
  sub-score measures declining over fabricating when the facts do not contain
  the answer. A judge additionally rates answer coherence so degraded English
  is visible even when the gold token appears.
- **In-context binding** (judge-free): `slm.worldgen` samples small worlds of
  invented people, places, and objects whose facts are consistent by
  construction, verbalizes a fragment into a context paragraph, and asks a
  question whose exact answer the program knows. Tasks are stratified over
  the four question categories and reported with per-kind sub-scores. This
  is the coherence gauge that gates the later experiments (notably the graph
  input/output approach). Preview tasks with `python -m slm.worldgen --seed 7`.
- **Completions**: judged grammar and coherence over seeds half grounded
  (openings cut from rendered world documents) and half generic.
- **Out-of-distribution generalization** (demoted, explicitly labeled): the
  earlier real-world instruction set and factual accuracy probe. The grounded
  corpus deliberately contains none of this knowledge, so low scores are
  expected; the accuracy probe reads as a contamination gauge (how much
  real-world fact leaked from the generator), not a target.

### One-document run summary

After evaluation, `eval/summary.md` (with `summary.json` beside it) collects
the whole run into one printable document: model and corpus sizes with the
tokens-per-parameter ratio, pretrain and finetune loss curves (persisted to
`checkpoints/*/history.jsonl` during training) with perplexities, a
comparable-loss table that scores every checkpoint on the same held-out data
(corpus validation stream, where a rise after finetuning indicates
forgetting, and held-out instruction pairs, where a drop indicates
instruction gain), and the evaluation scores per stage with the exact-match
grounded and binding columns first. It is written automatically at the end
of the evaluate stage and can be regenerated any time:

```bash
python -m slm.report --config runs/world/mini-<id>/config.resolved.yaml
```

Score parsing tolerates verbose judge replies, the completion rubric forces low
grades for text that is not well-formed English, and the report names the judge
model. Scores are only meaningful with a capable judge; the smoke config uses a
small judge for plumbing and its scores should not be trusted. The repetition
penalty defaults off and is windowed to recent tokens when used, so it cannot
suppress the whole vocabulary of a small byte-level model.

To read raw completions directly, use `slm.sample`, which completes
in-distribution seeds with the base pretrained model:

```bash
python -m slm.sample --config runs/world-<id>/pico/config.yaml
```

To probe a built model interactively by hand, use `slm.chat`, a prompt loop
over a checkpoint. The pretrain stage continues your text in completion style;
`--stage sft` answers in the Question and Answer framing. Sampling settings are
adjustable at runtime (`/temp`, `/topp`, `/penalty`, `/tokens`), and `/stage`
switches between the two models in place. These are tiny models, so this runs
fine on CPU; grab an interactive allocation rather than the login node:

```bash
srun --pty python -m slm.chat --config runs/world-<id>/pico/config.yaml
```

## Layout

```
.intent/project/   crystallized project intent graph
src/slm/
  config.py        yaml-driven configuration schema
  seeds.py         topic and structure seed vocabulary for prompts
  prompts.py       per-type prompt construction
  filters.py       heuristic contamination filter
  worldgen.py      programmatic world-state: documents and binding tasks
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
  evaluate.py      stage 5: judge, probes, binding
  report.py        one-document run summary (losses, curves, scores)
  pipeline.py      local orchestrator
slurm/
  submit.py            dependency-chained sbatch submitter
  example_stage.sbatch
configs/
  poc.yaml, smoke.yaml, pilot.yaml
  scale/world.yaml, scale/mini.yaml
```

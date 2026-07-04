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
> non-GPU stages run end-to-end on CPU via the smoke config.

## Stages

| Stage | Module | GPU | Output |
|-------|--------|-----|--------|
| generate | `slm.generate` | yes (vLLM) | `data/pretrain/*.jsonl`, `data/sft/sft.jsonl` |
| tokenizer | `slm.tokenizer` | no | `tokenizer/tokenizer.json` |
| data | `slm.data` | no | `data/packed/{train,val}.bin`, `meta.json` |
| pretrain | `slm.pretrain` | yes | `checkpoints/pretrain/ckpt_{best,last}.pt` |
| evaluate | `slm.evaluate` | yes (vLLM) | `eval/report.{json,md}` |

All artifacts land under `project.out_dir`, for example `runs/poc/`.

Instruction following is learned during pretraining rather than in a separate
stage. The generated prompt and response pairs are rendered in a light,
pretraining-adjacent format (`Question: ... Answer: ...`) and mixed into the
pretraining corpus at `pretrain.instruction_fraction` of the tokens. A separate
`slm.finetune` stage with role control tokens still exists for later, more
traditional supervised finetuning, but it is not in the default pipeline;
request it explicitly with `--stages ...,finetune,...`.

## Referent-free design

- Train from scratch (random initialization): no pretrained weights, no
  inherited knowledge.
- Fresh BPE tokenizer trained only on the synthetic corpus: the vocabulary
  cannot encode unseen real-world tokens.
- Referent-free generation: a strict system prompt at the chosen severity that
  forbids assistant framing and meta-commentary, a few-shot exemplar per text
  type so the generator returns only the text with no preamble, and a
  best-effort filter that drops texts with digits, urls, blocklisted real or
  technology words, or assistant and meta phrases (for example self-reference
  to being a model, or to the act of writing).

Severity is the dial on the degree of referent removal:

- `s1` (default): no proper nouns, numbers, dates, or named real entities;
  generic common nouns such as forest, lake, animal are allowed.
- `s2`: additionally prefer category-level terms (a creature, a substance, a
  structure) over specific kinds, and reject capitalized proper-noun phrases.

A global invented lexicon (the most extreme rung) is described in the intent
graph and not implemented in this pass.

## Text types

Implemented in this pass: prose, conversation, definition, description. Each is
serious and referent-free. The reasoning-oriented types (legal and
argumentative dialogue, notation-logic and puzzle documents) and the bounded
real-domain signal-injection experiment are described in the intent graph as
deferred work.

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

## Scaling ladder

`configs/scale/` holds a ladder of self-contained runs that grow the model and
the corpus together at a roughly twenty-tokens-per-parameter ratio, so the model
is neither starved nor oversized at any rung. Run them in order and compare the
pretrain validation loss and the evaluation report at each step.

| Config | preset | non-embed params | texts | approx tokens | ratio |
|--------|--------|------------------|-------|---------------|-------|
| `s0_pico.yaml` | pico | 0.43M | 75k | ~9M | ~20:1 |
| `s1_nano.yaml` | nano | 1.8M | 300k | ~35M | ~20:1 |
| `s2_micro.yaml` | micro | 6.0M | 1M | ~120M | ~20:1 |

Each rung generates its own corpus, so generation wall time grows with the
rung (the rung configs raise `generate.workers` accordingly, up to sixteen
parallel generate jobs for `micro`; use `slurm.pretrain_gres` for multi-GPU
pretraining at the higher rungs). The pretrain log prints
`tokens per non-embedding parameter` so the ratio is visible per run.

```bash
python slurm/submit.py --config configs/scale/s0_pico.yaml --stages generate
python -m slm.inspect --config configs/scale/s0_pico.yaml
python slurm/submit.py --config configs/scale/s0_pico.yaml \
  --stages tokenizer,data,pretrain,finetune,evaluate
```

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

Evaluation runs on the base pretrained model, which is the product at this
scale. An existing model judges it two primary ways: completions from
in-distribution seeds, scored for grammar, coherence, and how free they are of
real-world referents; and in-world instructions, scored for coherence and for
whether the answer follows the request. A small real-world knowledge probe is
kept but demoted, because a tiny model answers such out-of-distribution
questions poorly and those scores are unreliable. Results are written as
`report.json` and `report.md`.

Score parsing tolerates verbose judge replies, the completion rubric forces low
grades for text that is not well-formed English, and the report names the judge
model. Scores are only meaningful with a capable judge; the smoke config uses a
small judge for plumbing and its scores should not be trusted. The repetition
penalty defaults off and is windowed to recent tokens when used, so it cannot
suppress the whole vocabulary of a small byte-level model.

To read raw completions directly, use `slm.sample`, which completes
in-distribution seeds with the base pretrained model:

```bash
python -m slm.sample --config configs/scale/s1_nano.yaml
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
  pretrain.py      stage 3: pretraining loop
  finetune.py      stage 4: supervised finetuning
  infer.py         load a checkpoint and sample
  evaluate.py      stage 5: judge, probe, interrogation
  pipeline.py      local orchestrator
slurm/
  submit.py            dependency-chained sbatch submitter
  example_stage.sbatch
configs/
  poc.yaml, smoke.yaml
```

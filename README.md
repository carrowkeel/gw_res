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
| finetune | `slm.finetune` | yes | `checkpoints/sft/ckpt_last.pt` |
| evaluate | `slm.evaluate` | yes (vLLM) | `eval/report.{json,md}` |

All artifacts land under `project.out_dir`, for example `runs/poc/`.

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

## Run locally

```bash
PYTHONPATH=src python -m slm.pipeline --config configs/poc.yaml
PYTHONPATH=src python -m slm.pipeline --config configs/smoke.yaml \
  --stages tokenizer,data,pretrain,finetune
```

## Multi-GPU pretraining

Set `slurm.pretrain_gres: gpu:l40s:4` (the submitter switches that stage to
`torchrun --nproc_per_node=4`), or launch directly:

```bash
torchrun --standalone --nproc_per_node=4 -m slm.pretrain --config configs/poc.yaml
```

## Key parameters

- `generate.number_of_texts`, `generate.number_of_pairs`: corpus size.
- `generate.default_model`, `generate.type_models`: generator routing per type.
- `generate.severity`: referent-removal degree (`s1`, `s2`).
- `generate.text_type_weights`: relative amount of each text type.
- `tokenizer.vocabulary_size`: fresh BPE vocabulary size.
- `model.preset`: `poc-60m` through `poc-1b`, default `poc-150m`.
- `pretrain.*`, `finetune.*`: optimization and schedule.
- `eval.judge_model`: model that scores and interrogates the student.
- `slurm.*`: `gres`, `memory`, `cpus_per_task`, `time_limit`, `pretrain_gres`.

## Evaluation

An existing model judges the student three ways: quality scoring (grammar,
coherence, creativity), a referent-free probe (factual questions the model
should not answer, scoring the degree of referent absence), and a
model-queries-model interrogation. Results are written as `report.json` and
`report.md`.

Score parsing tolerates verbose judge replies, the quality rubric forces low
grades for text that is not well-formed English, and the report names the judge
model. Scores are only meaningful with a capable judge; the smoke config uses a
small judge for plumbing and its scores should not be trusted.

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

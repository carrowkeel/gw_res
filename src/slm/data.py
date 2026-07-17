"""Stage two: pack the corpus and provide datasets for training.

Pretraining documents are tokenized, wrapped with sequence-boundary tokens, and
packed into flat binary token arrays for fast random-window sampling. Instruction
and response pairs are rendered in a light, pretraining-adjacent format and mixed
into the same corpus at a target token fraction, so the model learns to follow
instructions during pretraining rather than in a separate, collapse-prone stage.

    python -m slm.data --config configs/poc.yaml
"""

import argparse
import json

import numpy

from .config import load_config
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger

logger = get_logger('data')

INSTRUCTION_TEMPLATE = 'Question: %s\nAnswer: %s'
INSTRUCTION_PREFIX = 'Question: %s\nAnswer:'


def render_instruction(prompt, response):
    """Render one instruction pair as light pretraining-adjacent text."""
    return INSTRUCTION_TEMPLATE % (prompt.strip(), response.strip())


def iterate_pairs(config):
    """Yield (prompt, response, kind) tuples from the generated pairs file.

    kind is the program-assigned task kind of the pair's grounding, or None
    for pairs from corpora generated before kinds were recorded.
    """
    path = config.corpus_sft_path
    if not path.exists():
        return
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                record = json.loads(stripped)
                yield record['prompt'], record['response'], record.get('kind')


def _dtype_for_vocabulary(vocabulary_size):
    return numpy.uint16 if vocabulary_size < 2**16 else numpy.uint32


def _wrap(tokenizer, text, dtype):
    token_ids = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
    return numpy.array(token_ids, dtype=dtype)


def _mix_instructions(pretrain_documents, instruction_documents, fraction,
                      random_generator):
    """Sample instruction documents to a target fraction of total tokens.

    Upsamples by cycling when there are too few pairs, and downsamples by using
    a subset when there are too many, so the instruction share of the packed
    corpus matches the requested fraction from either side.
    """
    if not instruction_documents or fraction <= 0.0:
        return pretrain_documents, 0
    pretrain_tokens = sum(len(document) for document in pretrain_documents)
    desired = int(fraction / (1.0 - fraction) * pretrain_tokens)
    if desired <= 0:
        return pretrain_documents, 0
    order = random_generator.permutation(len(instruction_documents))
    mixed = []
    accumulated = 0
    position = 0
    while accumulated < desired:
        document = instruction_documents[order[position % len(order)]]
        mixed.append(document)
        accumulated += len(document)
        position += 1
    return pretrain_documents + mixed, accumulated


def prepare_pretrain(config):
    """Tokenize the corpus, mix in instructions, and write packed binaries."""
    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    dtype = _dtype_for_vocabulary(tokenizer.vocabulary_size)
    pretrain_directory = config.corpus_pretrain_dir
    output_directory = ensure_directory(config.data_dir / 'packed')

    shards = sorted(pretrain_directory.glob('shard_*.jsonl'))
    if not shards:
        raise FileNotFoundError('no pretrain shards in %s' % pretrain_directory)

    documents = []
    for shard in shards:
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                text = json.loads(stripped)['text']
                documents.append(_wrap(tokenizer, text, dtype))
    logger.info('tokenized %d pretraining documents', len(documents))

    # The co-trained instruction stream can be restricted to configured kinds,
    # reserving the excluded kinds for the finetune stage (which always trains
    # on all pairs), so finetuning teaches capabilities pretraining did not
    # co-train rather than repeating the same distribution.
    allowed_kinds = set(config.pretrain.instruction_kinds)
    instruction_documents = []
    excluded = 0
    for prompt, response, kind in iterate_pairs(config):
        if allowed_kinds and kind is not None and kind not in allowed_kinds:
            excluded += 1
            continue
        instruction_documents.append(
            _wrap(tokenizer, render_instruction(prompt, response), dtype)
        )
    if excluded:
        logger.info(
            'reserved %d pairs of excluded kinds for finetuning', excluded
        )
    random_generator = numpy.random.default_rng(config.project.seed)
    documents, instruction_tokens = _mix_instructions(
        documents, instruction_documents,
        config.pretrain.instruction_fraction, random_generator,
    )
    if instruction_documents:
        logger.info(
            'mixed instructions: %d pairs upsampled to %d tokens',
            len(instruction_documents), instruction_tokens,
        )

    random_generator.shuffle(documents)
    validation_size = max(
        1, int(len(documents) * config.pretrain.validation_fraction)
    )
    validation_documents = documents[:validation_size]
    train_documents = documents[validation_size:]

    def write_split(name, split_documents):
        total_tokens = int(sum(len(document) for document in split_documents))
        path = output_directory / ('%s.bin' % name)
        array = numpy.memmap(path, dtype=dtype, mode='w+', shape=(total_tokens,))
        cursor = 0
        for document in split_documents:
            array[cursor:cursor + len(document)] = document
            cursor += len(document)
        array.flush()
        logger.info('wrote %s (%d tokens)', path.name, total_tokens)
        return total_tokens

    train_tokens = write_split('train', train_documents)
    validation_tokens = write_split('val', validation_documents)

    total = train_tokens + validation_tokens
    meta = {
        'vocabulary_size': tokenizer.vocabulary_size,
        'dtype': numpy.dtype(dtype).name,
        'train_tokens': train_tokens,
        'validation_tokens': validation_tokens,
        'number_of_documents': len(documents),
        'instruction_token_fraction': (
            round(instruction_tokens / total, 4) if total else 0.0
        ),
    }
    with open(output_directory / 'meta.json', 'w') as handle:
        json.dump(meta, handle, indent=2)
    return meta


class PackedDataset:
    """Random-offset sampler over a packed token memmap."""

    def __init__(self, binary_path, dtype, block_size):
        self.data = numpy.memmap(binary_path, dtype=numpy.dtype(dtype), mode='r')
        self.block_size = block_size

    def length(self):
        return max(0, len(self.data) - self.block_size - 1)

    def get_batch(self, batch_size, device, random_generator):
        import torch

        offsets = random_generator.integers(0, self.length(), size=batch_size)
        inputs = numpy.stack(
            [self.data[start:start + self.block_size] for start in offsets]
        )
        targets = numpy.stack(
            [
                self.data[start + 1:start + 1 + self.block_size]
                for start in offsets
            ]
        )
        inputs = torch.from_numpy(inputs.astype(numpy.int64))
        targets = torch.from_numpy(targets.astype(numpy.int64))
        if device.startswith('cuda'):
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        return inputs, targets


def render_pair_example(tokenizer, instruction, response, maximum_length,
                        loss_mode='response_only'):
    """Return input ids and next-token labels for one instruction pair.

    Uses the same light Question and Answer format as the co-trained
    instructions. Labels are the next token at each position, matching the
    convention the model expects (loss is cross entropy of the logits at
    position i against the token at position i+1, with no internal shift, the
    same convention the packed pretraining data follows). In the default
    response_only mode every position that would predict a token inside the
    prompt is masked with -100, so loss falls only on predicting the answer,
    starting from predicting its first token given the whole prompt. In
    full_sequence mode no position is masked, so the model is trained to
    predict the prompt tokens as well, the same objective as pretraining.
    """
    prefix_ids = (
        [tokenizer.bos_id]
        + tokenizer.encode(INSTRUCTION_PREFIX % instruction.strip())
    )
    answer_ids = tokenizer.encode(' ' + response.strip()) + [tokenizer.eos_id]
    tokens = prefix_ids + answer_ids
    input_ids = tokens[:-1]
    if loss_mode == 'full_sequence':
        labels = tokens[1:]
    else:
        labels = [-100] * (len(prefix_ids) - 1) + answer_ids
    return input_ids[:maximum_length], labels[:maximum_length]


class PairDataset:
    """Supervised finetuning examples with next-token labels.

    loss_mode selects response_only masking (loss on the answer only) or
    full_sequence (loss on every token). maximum_length and the loss mode
    default to the base finetune config but can be overridden per variant.
    """

    def __init__(self, config, tokenizer, loss_mode=None, maximum_length=None):
        self.examples = []
        self.pad_id = tokenizer.pad_id
        path = config.corpus_sft_path
        if maximum_length is None:
            maximum_length = config.finetune.maximum_sequence_length
        if loss_mode is None:
            loss_mode = config.finetune.loss_mode
        with open(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                self.examples.append(
                    render_pair_example(
                        tokenizer,
                        record['prompt'],
                        record['response'],
                        maximum_length,
                        loss_mode,
                    )
                )
        logger.info('loaded %d finetuning examples', len(self.examples))

    def length(self):
        return len(self.examples)

    def collate(self, indices, device):
        import torch

        items = [self.examples[index] for index in indices]
        longest = max(len(input_ids) for input_ids, _ in items)
        input_batch = []
        label_batch = []
        attention_batch = []
        for input_ids, labels in items:
            padding = longest - len(input_ids)
            input_batch.append(input_ids + [self.pad_id] * padding)
            label_batch.append(labels + [-100] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
        as_tensor = lambda rows: torch.tensor(rows, dtype=torch.long, device=device)
        return as_tensor(input_batch), as_tensor(label_batch), as_tensor(attention_batch)


def main():
    parser = argparse.ArgumentParser(description='Pack pretraining data')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    meta = prepare_pretrain(load_config(arguments.config))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

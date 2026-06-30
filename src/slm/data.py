"""Stage two: pack the corpus and provide datasets for training.

Pretraining documents are tokenized, wrapped with sequence-boundary tokens, and
packed into flat binary token arrays for fast random-window sampling. Finetuning
pairs are rendered with role control tokens and a response-only loss mask.

    python -m slm.data --config configs/poc.yaml
"""

import argparse
import json

import numpy

from .config import load_config
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger

logger = get_logger('data')


def _dtype_for_vocabulary(vocabulary_size):
    return numpy.uint16 if vocabulary_size < 2**16 else numpy.uint32


def prepare_pretrain(config):
    """Tokenize the corpus and write packed train and validation binaries."""
    tokenizer = SyntheticTokenizer(config.tokenizer_path)
    dtype = _dtype_for_vocabulary(tokenizer.vocabulary_size)
    pretrain_directory = config.data_dir / 'pretrain'
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
                token_ids = [tokenizer.bos_id] + tokenizer.encode(text) + [
                    tokenizer.eos_id
                ]
                documents.append(numpy.array(token_ids, dtype=dtype))
    logger.info('tokenized %d documents', len(documents))

    random_generator = numpy.random.default_rng(config.project.seed)
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

    meta = {
        'vocabulary_size': tokenizer.vocabulary_size,
        'dtype': numpy.dtype(dtype).name,
        'train_tokens': train_tokens,
        'validation_tokens': validation_tokens,
        'number_of_documents': len(documents),
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


def render_pair_example(tokenizer, instruction, response, maximum_length):
    """Return input ids and labels with the instruction span masked out."""
    instruction_ids = (
        [tokenizer.bos_id, tokenizer.user_id]
        + tokenizer.encode(instruction)
        + [tokenizer.assistant_id]
    )
    response_ids = tokenizer.encode(response) + [tokenizer.eos_id]
    input_ids = instruction_ids + response_ids
    labels = [-100] * len(instruction_ids) + response_ids
    return input_ids[:maximum_length], labels[:maximum_length]


class PairDataset:
    """Supervised finetuning examples with response-only loss masks."""

    def __init__(self, config, tokenizer):
        self.examples = []
        self.pad_id = tokenizer.pad_id
        path = config.data_dir / 'sft' / 'sft.jsonl'
        maximum_length = config.finetune.maximum_sequence_length
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

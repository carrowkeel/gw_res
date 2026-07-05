"""Graph stage three: build and pack graph-context training examples.

Each graph record is replayed segment by segment. For selected prefixes, one
example is emitted: the linearized graph of everything folded so far, a next
marker, and the following raw segment as the continuation target. Examples are
tokenized and packed into the same flat binary format the base pipeline uses,
so the unmodified pretraining loop trains the graph model with a standard
full-sequence next-token loss. A masked continuation-only loss remains an
interchangeable alternative: it would follow the PairDataset pattern in
finetune, swapping the dataset rather than the objective code.

A small context dropout optionally removes random leaf subtrees from training
contexts so the model is robust to the reduced subtrees it will see when
evaluation applies a context budget.

    python -m slm.graph_data --config configs/poc.yaml
"""

import argparse
import json

import numpy

from .config import load_config
from .graph import ContextGraph
from .graph_tokenizer import marker_ids
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger

logger = get_logger('graph_data')


def _dtype_for_vocabulary(vocabulary_size):
    return numpy.uint16 if vocabulary_size < 2**16 else numpy.uint32


def select_prefixes(segment_count, examples_per_text):
    """Choose which fold prefixes become examples, evenly when capped."""
    candidates = list(range(1, segment_count))
    if len(candidates) <= examples_per_text:
        return set(candidates)
    positions = numpy.linspace(
        0, len(candidates) - 1, examples_per_text
    ).round().astype(int)
    return {candidates[position] for position in positions}


def _dropout_include(graph, last_node, dropout, random_generator):
    """Randomly drop unprotected leaf subtrees with the dropout probability."""
    if dropout <= 0.0 or graph.node_count() <= 2:
        return None
    include = set(range(graph.node_count()))
    for leaf in graph.leaves():
        if leaf == 0 or leaf == last_node:
            continue
        if random_generator.random() < dropout:
            include.discard(leaf)
    return include if len(include) < graph.node_count() else None


def build_examples(record, graph_config, tokenizer, markers, dtype,
                   random_generator):
    """Replay one record's fold and emit packed example token arrays."""
    segments = record['segments']
    prefixes = select_prefixes(len(segments), graph_config.examples_per_text)
    graph = ContextGraph()
    examples = []
    last_node = None
    for position, segment in enumerate(segments):
        if position in prefixes:
            include = _dropout_include(
                graph, last_node, graph_config.context_dropout,
                random_generator,
            )
            token_ids = (
                [tokenizer.bos_id]
                + graph.linearize_ids(tokenizer.encode, markers, include)
                + [markers['next']]
                + tokenizer.encode(segment)
                + [tokenizer.eos_id]
            )
            examples.append(numpy.array(token_ids, dtype=dtype))
        _, last_node, _ = graph.fold(
            segment,
            graph_config.relatedness_threshold,
            graph_config.node_token_limit,
        )
    return examples


def run(config):
    """Build graph-context examples and write packed binaries."""
    tokenizer = SyntheticTokenizer(config.graph_tokenizer_path)
    markers = marker_ids(tokenizer)
    dtype = _dtype_for_vocabulary(tokenizer.vocabulary_size)
    graphs_path = config.graphs_dir / 'graphs.jsonl'
    if not graphs_path.exists():
        raise FileNotFoundError('no graph shard at %s' % graphs_path)
    output_directory = ensure_directory(config.graph_packed_dir)
    random_generator = numpy.random.default_rng(config.project.seed)

    documents = []
    with open(graphs_path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            documents.extend(build_examples(
                record, config.graph, tokenizer, markers, dtype,
                random_generator,
            ))
    if not documents:
        raise ValueError('no graph examples were built from %s' % graphs_path)
    logger.info('built %d graph-context examples', len(documents))

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
        'mean_example_tokens': round(
            (train_tokens + validation_tokens) / len(documents), 1
        ),
    }
    with open(output_directory / 'meta.json', 'w') as handle:
        json.dump(meta, handle, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description='Pack graph-context data')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    meta = run(load_config(arguments.config))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

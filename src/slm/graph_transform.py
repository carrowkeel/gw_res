"""Graph stage one: transform generated texts into context graphs.

Each generated text is segmented (conversation turns for dialogue, grouped
sentences otherwise) and folded segment by segment into a per-text
ContextGraph using the two growth moves: extend the most related node, or add
a new node rooted under node zero when nothing related exists. A held-out
fraction of the conversations is reserved for the graph evaluation stage and
excluded from graph training data.

    python -m slm.graph_transform --config configs/poc.yaml
"""

import argparse
import json
import re

import numpy

from .config import load_config
from .graph import ContextGraph, estimate_tokens, split_sentences
from .utils import ensure_directory, get_logger

logger = get_logger('graph_transform')

_TURN_PATTERN = re.compile(r'^\s*([A-Z][A-Za-z]*):\s+')


def conversation_turns(text):
    """Split dialogue into speaker turns, or return None when not dialogue."""
    turns = []
    current = None
    for line in text.split('\n'):
        if _TURN_PATTERN.match(line):
            if current is not None:
                turns.append(current.strip())
            current = line.strip()
        elif current is not None and line.strip():
            current = current + ' ' + line.strip()
    if current is not None:
        turns.append(current.strip())
    return turns if len(turns) >= 2 else None


def prose_segments(text, segment_tokens):
    """Group sentences into segments of roughly segment_tokens tokens."""
    segments = []
    current = []
    current_tokens = 0
    for sentence in split_sentences(text):
        current.append(sentence)
        current_tokens += estimate_tokens(sentence)
        if current_tokens >= segment_tokens:
            segments.append(' '.join(current))
            current = []
            current_tokens = 0
    if current:
        segments.append(' '.join(current))
    return segments


def segment_text(text, text_type, segment_tokens):
    """Return (segments, is_conversation) for one generated text."""
    if text_type == 'conversation':
        turns = conversation_turns(text)
        if turns is not None:
            return turns, True
    return prose_segments(text, segment_tokens), False


def fold_segments(segments, graph_config):
    """Fold all segments into a fresh graph, counting the moves taken."""
    graph = ContextGraph()
    counts = {'root': 0, 'extend': 0, 'new': 0, 'split': 0}
    for segment in segments:
        move, _, split = graph.fold(
            segment,
            graph_config.relatedness_threshold,
            graph_config.node_token_limit,
        )
        counts[move] += 1
        if split:
            counts['split'] += 1
    return graph, counts


def iterate_texts(config):
    pretrain_directory = config.data_dir / 'pretrain'
    shards = sorted(pretrain_directory.glob('shard_*.jsonl'))
    if not shards:
        raise FileNotFoundError('no pretrain shards in %s' % pretrain_directory)
    for shard in shards:
        with open(shard) as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    record = json.loads(stripped)
                    yield record['text'], record.get('type', 'prose')


def run(config):
    """Segment and fold the corpus, writing graph and holdout shards."""
    graph_config = config.graph
    output_directory = ensure_directory(config.graphs_dir)
    random_generator = numpy.random.default_rng(config.project.seed)

    totals = {
        'texts': 0, 'skipped': 0, 'graphs': 0, 'holdout': 0,
        'root': 0, 'extend': 0, 'new': 0, 'split': 0, 'nodes': 0,
    }
    exported = 0

    graphs_path = output_directory / 'graphs.jsonl'
    holdout_path = output_directory / 'holdout.jsonl'
    with open(graphs_path, 'w') as graphs_handle, \
            open(holdout_path, 'w') as holdout_handle:
        for text, text_type in iterate_texts(config):
            totals['texts'] += 1
            segments, is_conversation = segment_text(
                text, text_type, graph_config.segment_tokens
            )
            if len(segments) < 2:
                totals['skipped'] += 1
                continue
            if (is_conversation
                    and random_generator.random()
                    < graph_config.holdout_fraction):
                holdout_handle.write(json.dumps(
                    {'type': text_type, 'segments': segments},
                    ensure_ascii=False,
                ) + '\n')
                totals['holdout'] += 1
                continue
            graph, counts = fold_segments(segments, graph_config)
            for move in ('root', 'extend', 'new', 'split'):
                totals[move] += counts[move]
            totals['graphs'] += 1
            totals['nodes'] += graph.node_count()
            graphs_handle.write(json.dumps(
                {
                    'type': text_type,
                    'segments': segments,
                    'graph': graph.to_record(),
                },
                ensure_ascii=False,
            ) + '\n')
            if exported < graph_config.export_intent_examples:
                graph.export_intent_files(
                    output_directory / 'intent_examples' / str(exported),
                    'example_%d' % exported,
                )
                exported += 1

    meta = dict(totals)
    meta['mean_nodes_per_graph'] = (
        round(totals['nodes'] / totals['graphs'], 2) if totals['graphs'] else 0
    )
    with open(output_directory / 'meta.json', 'w') as handle:
        json.dump(meta, handle, indent=2)
    logger.info(
        'transformed %d texts into %d graphs (%d held out, %d skipped)',
        totals['texts'], totals['graphs'], totals['holdout'], totals['skipped'],
    )
    logger.info(
        'moves: %d extend, %d new, %d splits, %.2f nodes per graph',
        totals['extend'], totals['new'], totals['split'],
        meta['mean_nodes_per_graph'],
    )
    return meta


def main():
    parser = argparse.ArgumentParser(description='Transform texts into graphs')
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    meta = run(load_config(arguments.config))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

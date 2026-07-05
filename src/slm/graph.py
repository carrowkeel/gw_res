"""Context graph structures for the graph pipeline.

A ContextGraph holds text content as nodes of a tree that grows by two moves:
a new node rooted under node zero (used when nothing related exists yet), or an
extension of the most related existing node. An extension that pushes a node
past a fixed token limit splits the overflow into a new child node, so
restructuring is local and append-mostly and cached prefixes of the linearized
form survive growth. There is no rebalancing and no merging.

Storage is compatible with the write2 intent graph layout (meta.json,
edges.json, nodes/{index}.json) through export_intent_files, and with a compact
single-object record for jsonl shards through to_record and from_record.
"""

import json
import math
import re

from .utils import ensure_directory

GRAPH_OPEN = '<|g|>'
GRAPH_CLOSE = '<|/g|>'
NODE_OPEN = '<|n|>'
NODE_CLOSE = '<|/n|>'
NEXT_MARKER = '<|next|>'

STRUCTURE_TOKENS = [GRAPH_OPEN, GRAPH_CLOSE, NODE_OPEN, NODE_CLOSE, NEXT_MARKER]

_STOPWORDS = {
    'the', 'and', 'that', 'this', 'with', 'from', 'for', 'was', 'were', 'are',
    'is', 'be', 'been', 'has', 'had', 'have', 'not', 'but', 'its', 'it',
    'of', 'to', 'in', 'on', 'at', 'as', 'by', 'an', 'or', 'if', 'so',
    'a', 'no', 'nor', 'do', 'did', 'does', 'can', 'could', 'will', 'would',
    'there', 'their', 'they', 'them', 'then', 'than', 'when', 'where',
    'which', 'while', 'into', 'over', 'under', 'after', 'before', 'between',
    'one', 'two', 'any', 'all', 'each', 'more', 'most', 'some', 'such',
    'only', 'own', 'same', 'you', 'your', 'she', 'her', 'he', 'his', 'him',
    'we', 'our', 'us', 'what', 'who', 'how', 'why', 'also', 'because',
}


def estimate_tokens(text):
    """Cheap token estimate used for node limits, matching write2's rule."""
    return max(1, math.ceil(len(text) / 4))


def content_words(text):
    """Lowercased content words used for lexical relatedness."""
    words = re.findall(r"[a-z']+", text.lower())
    return {word for word in words if len(word) >= 3 and word not in _STOPWORDS}


def relatedness(words_a, words_b):
    """Cosine similarity between two binary bags of content words."""
    if not words_a or not words_b:
        return 0.0
    shared = len(words_a & words_b)
    if shared == 0:
        return 0.0
    return shared / math.sqrt(len(words_a) * len(words_b))


def split_sentences(text):
    """Split text into sentence-like pieces, keeping their punctuation."""
    pieces = re.findall(r'[^.!?\n]+[.!?]*\s*', text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _split_content(content):
    """Split node content near its midpoint at a sentence boundary.

    Returns (head, tail); tail is empty when no usable boundary exists.
    """
    sentences = split_sentences(content)
    if len(sentences) < 2:
        words = content.split()
        if len(words) < 8:
            return content, ''
        midpoint = len(words) // 2
        return ' '.join(words[:midpoint]), ' '.join(words[midpoint:])
    total = sum(estimate_tokens(sentence) for sentence in sentences)
    accumulated = 0
    boundary = 1
    for position, sentence in enumerate(sentences[:-1]):
        accumulated += estimate_tokens(sentence)
        if accumulated >= total / 2:
            boundary = position + 1
            break
    head = ' '.join(sentences[:boundary])
    tail = ' '.join(sentences[boundary:])
    return head, tail


class ContextGraph:
    """Tree-shaped context graph grown by rooted insertion and extension."""

    def __init__(self):
        self.contents = []
        self.parents = []
        self._word_sets = []

    def node_count(self):
        return len(self.contents)

    def add_node(self, content, parent):
        index = len(self.contents)
        self.contents.append(content)
        self.parents.append(parent)
        self._word_sets.append(content_words(content))
        return index

    def children(self, index):
        return [
            child for child, parent in enumerate(self.parents)
            if parent == index
        ]

    def leaves(self):
        parent_set = set(
            parent for parent in self.parents if parent is not None
        )
        return [
            index for index in range(len(self.contents))
            if index not in parent_set
        ]

    def best_match(self, segment):
        """Return (index, score) of the node most related to the segment."""
        segment_words = content_words(segment)
        best_index = None
        best_score = 0.0
        for index, node_words in enumerate(self._word_sets):
            score = relatedness(segment_words, node_words)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index, best_score

    def _extend(self, index, segment, node_token_limit):
        self.contents[index] = self.contents[index] + '\n' + segment
        self._word_sets[index] = content_words(self.contents[index])
        if estimate_tokens(self.contents[index]) <= node_token_limit:
            return index, False
        head, tail = _split_content(self.contents[index])
        if not tail:
            return index, False
        self.contents[index] = head
        self._word_sets[index] = content_words(head)
        child = self.add_node(tail, index)
        return child, True

    def fold(self, segment, relatedness_threshold, node_token_limit):
        """Fold one text segment into the graph.

        Returns (move, node_index, split) where move is 'root', 'extend', or
        'new', node_index is the node now holding the segment, and split says
        whether the extension overflowed into a new child.
        """
        if not self.contents:
            return 'root', self.add_node(segment, None), False
        match_index, score = self.best_match(segment)
        if match_index is not None and score >= relatedness_threshold:
            node_index, split = self._extend(
                match_index, segment, node_token_limit
            )
            return 'extend', node_index, split
        return 'new', self.add_node(segment, 0), False

    def dfs_indices(self, include=None):
        """Depth-first node order with creation-ordered children.

        Yields (index, event) pairs where event is 'open' or 'close'. When an
        include set is given, excluded nodes and their subtrees are skipped.
        """
        if not self.contents:
            return
        stack = [(0, 'open')]
        while stack:
            index, event = stack.pop()
            if include is not None and index not in include:
                continue
            yield index, event
            if event == 'open':
                stack.append((index, 'close'))
                for child in reversed(self.children(index)):
                    stack.append((child, 'open'))

    def linearize_ids(self, encode, markers, include=None):
        """Serialize the graph into token ids using structural marker ids.

        markers maps the names graph_open, graph_close, node_open, and
        node_close to token ids. encode maps text to a list of token ids.
        """
        token_ids = [markers['graph_open']]
        for index, event in self.dfs_indices(include):
            if event == 'open':
                token_ids.append(markers['node_open'])
                token_ids.extend(encode(self.contents[index]))
            else:
                token_ids.append(markers['node_close'])
        token_ids.append(markers['graph_close'])
        return token_ids

    def linearize_text(self, include=None):
        """Human-readable linearization used for inspection and debugging."""
        parts = [GRAPH_OPEN]
        for index, event in self.dfs_indices(include):
            if event == 'open':
                parts.append(NODE_OPEN)
                parts.append(self.contents[index])
            else:
                parts.append(NODE_CLOSE)
        parts.append(GRAPH_CLOSE)
        return ' '.join(parts)

    def reduce_to_budget(self, query_text, node_costs, budget, protected=None):
        """Select a root-connected subtree that fits a token budget.

        Drops the leaf subtree least related to the query until the summed
        node costs fit the budget or nothing else can be dropped. Node zero
        and protected nodes are never dropped. Returns the include set.
        """
        include = set(range(len(self.contents)))
        protected_set = {0} | set(protected or [])
        query_words = content_words(query_text)

        def total_cost():
            return sum(node_costs[index] for index in include)

        while total_cost() > budget:
            included_parents = {
                self.parents[index] for index in include
                if self.parents[index] is not None
                and self.parents[index] in include
            }
            droppable = [
                index for index in include
                if index not in protected_set and index not in included_parents
            ]
            if not droppable:
                break
            worst = min(
                droppable,
                key=lambda index: relatedness(
                    query_words, self._word_sets[index]
                ),
            )
            include.discard(worst)
        return include

    def to_record(self):
        return {'contents': self.contents, 'parents': self.parents}

    @classmethod
    def from_record(cls, record):
        graph = cls()
        for content, parent in zip(record['contents'], record['parents']):
            graph.contents.append(content)
            graph.parents.append(parent)
            graph._word_sets.append(content_words(content))
        return graph

    def export_intent_files(self, directory, graph_id):
        """Write the graph in the write2 intent layout for interchange."""
        base = ensure_directory(directory)
        nodes_directory = ensure_directory(base / 'nodes')
        meta = {
            'graph_id': graph_id,
            'node_count': len(self.contents),
            'resolution_levels': [{'name': 'full', 'target_tokens': None}],
        }
        edges = [
            {'from': parent, 'to': index, 'weight': 1.0}
            for index, parent in enumerate(self.parents)
            if parent is not None
        ]
        with open(base / 'meta.json', 'w') as handle:
            json.dump(meta, handle, indent=2)
        with open(base / 'edges.json', 'w') as handle:
            json.dump(edges, handle, indent=2)
        for index, content in enumerate(self.contents):
            with open(nodes_directory / ('%d.json' % index), 'w') as handle:
                json.dump([content], handle, indent=2)

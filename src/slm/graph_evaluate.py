"""Graph stage five: compare flat and graph context at matched token budgets.

Held-out conversations from the transform stage drive the comparison. For each
conversation the models must continue the dialogue given every turn but the
last. The flat model receives the most recent transcript tokens that fit the
budget; the graph model receives the folded conversation graph reduced to the
budget by dropping the leaf subtrees least related to the latest turn, so
recency truncation competes against relevance selection at equal token cost.
A judge model scores each continuation for coherence with the full dialogue
and consistency with what was established earlier.

    python -m slm.graph_evaluate --config configs/poc.yaml
"""

import argparse
import json

from .config import load_config
from .graph import ContextGraph
from .graph_tokenizer import marker_ids
from .model import GPT, build_config
from .tokenizer import SyntheticTokenizer
from .utils import ensure_directory, get_logger, set_seed

logger = get_logger('graph_evaluate')

JUDGE_SYSTEM_PROMPT = (
    'You judge continuations of a dialogue. You are shown the dialogue so far '
    'and a candidate next turn produced by a small model. Rate the candidate '
    'on two axes from 1 to 10. Coherence: is it well-formed English that '
    'plausibly continues this dialogue? Consistency: does it respect what was '
    'established earlier in the dialogue (speakers, decisions, stated facts)? '
    'Poorly formed or off-topic text must score low on both. Reply with '
    'exactly two lines: "coherence: N" and "consistency: N".'
)


class _Student:
    """Checkpoint loader and id-level generator shared by both models."""

    def __init__(self, config, checkpoint_directory, tokenizer_path):
        import torch

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_path = checkpoint_directory / 'ckpt_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_directory / 'ckpt_last.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError('no checkpoint in %s' % checkpoint_directory)
        saved = torch.load(checkpoint_path, map_location=self.device)
        self.tokenizer = SyntheticTokenizer(tokenizer_path)
        gpt_config = build_config(config.model, saved['vocabulary_size'])
        self.model = GPT(gpt_config).to(self.device).eval()
        self.model.load_state_dict(saved['model'])
        self.block_size = gpt_config.block_size

    def generate_from_ids(self, prompt_ids, eval_config, max_new_tokens):
        import torch

        prompt_ids = prompt_ids[-self.block_size:]
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                input_ids, max_new_tokens,
                temperature=eval_config.temperature,
                top_p=eval_config.top_p,
                eos_id=self.tokenizer.eos_id,
                repetition_penalty=eval_config.repetition_penalty,
            )
        generated = output[0, len(prompt_ids):].tolist()
        return self.tokenizer.decode(generated).strip()


def flat_prompt_ids(student, context_turns, budget):
    """Most recent transcript tokens that fit the budget."""
    transcript_ids = student.tokenizer.encode('\n'.join(context_turns))
    return [student.tokenizer.bos_id] + transcript_ids[-budget:]


def graph_prompt_ids(student, markers, graph_config, context_turns, budget):
    """Folded conversation graph reduced to the budget, plus a next marker.

    Whole leaf subtrees least related to the latest turn are dropped first.
    When the remaining nodes still exceed the budget, node contents are
    trimmed from the head (extensions append, so tails are most recent),
    least related nodes first, making the budget a hard guarantee.
    """
    from .graph import content_words, relatedness

    graph = ContextGraph()
    last_node = None
    for turn in context_turns:
        _, last_node, _ = graph.fold(
            turn,
            graph_config.relatedness_threshold,
            graph_config.node_token_limit,
        )
    encoded = {
        index: student.tokenizer.encode(content)
        for index, content in enumerate(graph.contents)
    }
    node_costs = {index: len(ids) + 2 for index, ids in encoded.items()}
    content_budget = budget - 3
    include = graph.reduce_to_budget(
        context_turns[-1], node_costs, content_budget, protected=[last_node]
    )

    query_words = content_words(context_turns[-1])
    overflow = sum(node_costs[index] for index in include) - content_budget
    if overflow > 0:
        trim_order = sorted(
            include,
            key=lambda index: (
                index in (0, last_node),
                relatedness(query_words, content_words(graph.contents[index])),
            ),
        )
        for index in trim_order:
            if overflow <= 0:
                break
            removable = min(overflow, len(encoded[index]))
            encoded[index] = encoded[index][removable:]
            overflow -= removable

    token_ids = [student.tokenizer.bos_id, markers['graph_open']]
    for index, event in graph.dfs_indices(include):
        if event == 'open':
            token_ids.append(markers['node_open'])
            token_ids.extend(encoded[index])
        else:
            token_ids.append(markers['node_close'])
    token_ids.append(markers['graph_close'])
    token_ids.append(markers['next'])
    return token_ids


def load_holdout(config):
    holdout_path = config.graphs_dir / 'holdout.jsonl'
    if not holdout_path.exists():
        raise FileNotFoundError('no holdout shard at %s' % holdout_path)
    conversations = []
    with open(holdout_path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if len(record['segments']) >= 3:
                conversations.append(record['segments'])
    return conversations


def _judge_continuation(engine, sampling, context_turns, continuation):
    from .evaluate import _extract_score, _judge

    user_prompt = (
        'Dialogue so far:\n%s\n\nCandidate next turn:\n%s'
        % ('\n'.join(context_turns), continuation)
    )
    reply = _judge(engine, sampling, JUDGE_SYSTEM_PROMPT, user_prompt)
    return (
        _extract_score(reply, 'coherence'),
        _extract_score(reply, 'consistency'),
    )


def _mean(values):
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 3) if present else None


def run(config):
    """Run the flat-versus-graph comparison and write the report."""
    import numpy

    set_seed(config.project.seed)
    graph_config = config.graph
    conversations = load_holdout(config)
    if not conversations:
        raise ValueError('holdout shard contains no usable conversations')
    random_generator = numpy.random.default_rng(config.project.seed)
    random_generator.shuffle(conversations)
    conversations = conversations[:graph_config.number_of_eval_conversations]
    logger.info('evaluating on %d held-out conversations', len(conversations))

    graph_student = _Student(
        config, config.graph_pretrain_dir, config.graph_tokenizer_path
    )
    markers = marker_ids(graph_student.tokenizer)
    flat_student = None
    if (config.pretrain_dir / 'ckpt_best.pt').exists() \
            or (config.pretrain_dir / 'ckpt_last.pt').exists():
        flat_student = _Student(
            config, config.pretrain_dir, config.tokenizer_path
        )
    else:
        logger.warning(
            'no flat checkpoint under %s, evaluating graph model only',
            config.pretrain_dir,
        )

    judge = None
    if graph_config.judge_enabled:
        from .evaluate import _load_judge

        engine, sampling, judge_name = _load_judge(config)
        judge = (engine, sampling)
    else:
        judge_name = None
        logger.warning('judge disabled, report will carry samples only')

    rows = []
    samples = []
    for conversation_index, turns in enumerate(conversations):
        context_turns = turns[:-1]
        for budget in graph_config.context_budgets:
            candidates = {}
            if flat_student is not None:
                candidates['flat'] = flat_prompt_ids(
                    flat_student, context_turns, budget
                )
            candidates['graph'] = graph_prompt_ids(
                graph_student, markers, graph_config, context_turns, budget
            )
            for model_name, prompt_ids in candidates.items():
                student = (
                    flat_student if model_name == 'flat' else graph_student
                )
                continuation = student.generate_from_ids(
                    prompt_ids, config.eval, graph_config.max_new_tokens
                )
                coherence = consistency = None
                if judge is not None and continuation:
                    coherence, consistency = _judge_continuation(
                        judge[0], judge[1], context_turns, continuation
                    )
                rows.append({
                    'model': model_name,
                    'budget': budget,
                    'prompt_tokens': len(prompt_ids),
                    'coherence': coherence,
                    'consistency': consistency,
                })
                if conversation_index < 3:
                    samples.append({
                        'model': model_name,
                        'budget': budget,
                        'context_turns': context_turns,
                        'reference_turn': turns[-1],
                        'continuation': continuation,
                    })

    summary = {}
    for model_name in ('flat', 'graph'):
        model_rows = [row for row in rows if row['model'] == model_name]
        if not model_rows:
            continue
        summary[model_name] = {}
        for budget in graph_config.context_budgets:
            budget_rows = [
                row for row in model_rows if row['budget'] == budget
            ]
            summary[model_name][str(budget)] = {
                'coherence': _mean([row['coherence'] for row in budget_rows]),
                'consistency': _mean(
                    [row['consistency'] for row in budget_rows]
                ),
                'mean_prompt_tokens': _mean(
                    [row['prompt_tokens'] for row in budget_rows]
                ),
                'count': len(budget_rows),
            }

    report = {
        'judge_model': judge_name,
        'number_of_conversations': len(conversations),
        'context_budgets': list(graph_config.context_budgets),
        'summary': summary,
        'samples': samples,
    }
    output_directory = ensure_directory(config.eval_dir)
    with open(output_directory / 'report_graph.json', 'w') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = ['# Graph versus flat context', '']
    lines.append('Judge: %s' % (judge_name or 'disabled'))
    lines.append('Conversations: %d' % len(conversations))
    lines.append('')
    lines.append('| model | budget | coherence | consistency | prompt tokens |')
    lines.append('|-------|--------|-----------|-------------|---------------|')
    for model_name, budgets in summary.items():
        for budget, values in budgets.items():
            lines.append('| %s | %s | %s | %s | %s |' % (
                model_name, budget, values['coherence'],
                values['consistency'], values['mean_prompt_tokens'],
            ))
    with open(output_directory / 'report_graph.md', 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    logger.info(
        'wrote %s', output_directory / 'report_graph.json'
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Compare flat and graph context models'
    )
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    run(load_config(arguments.config))


if __name__ == '__main__':
    main()

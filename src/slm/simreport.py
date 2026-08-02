"""Compare simulation-training runs side by side.

Reads each run's history.jsonl and reduces it to the numbers the
scrap-or-keep decisions have actually turned on: returns against the
blind and oracle references, the program-owned health gauges (distinct
decision rate, eligible and grounded rates), and register drift measured
by the replay loss. Flags mark the known failure signatures - a distinct
rate through the collapse tripwire, a run stuck in no-signal steps -
so a sweep of eight or sixteen variants can be triaged at a glance.
Everything here is program-owned arithmetic over the histories; no LLM
judges anything.

Runs graded by the simeval battery additionally show their battery
mean, standard error, and headroom, plus a paired delta against the
first battery-graded row: the battery's games are seeded identically
across runs, so per-game differences cancel world luck and the delta
column (starred when it clears two standard errors) is the number that
separates real improvements from run-to-run noise.

Below the table, each run contributes sampled transcript moves from a
few timepoints (early, middle, and late by default) - the raw context
and decision pairs are the ground truth behind every rate in the table,
and every failure mode so far surfaced there first. --transcript-steps
picks the timepoints and --transcript-samples the moves per timepoint
(0 turns transcripts off).

    python -m slm.simreport --sweep runs/sweeps/sim-options-<run_id>
    python -m slm.simreport --runs runs/sim-a runs/sim-b
"""

import argparse
import json
import statistics
from pathlib import Path

COLLAPSE_DISTINCT = 0.5
TAIL_ROWS = 10


def load_history(run_dir):
    path = Path(run_dir) / 'checkpoints' / 'simtrain' / 'history.jsonl'
    if not path.exists():
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval(run_dir):
    path = Path(run_dir) / 'checkpoints' / 'simtrain' / 'eval.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def attach_eval(summaries, evals):
    """Attach battery results and paired deltas against the first row.

    The battery's games are seeded identically for every run, so
    per-game return differences are paired and world luck cancels; the
    delta column carries the mean difference against the first run that
    has battery results, starred when it clears two standard errors.
    Runs graded on a different battery are compared unpaired-free: they
    keep their own mean but no delta.
    """
    baseline = None
    for summary, evaluation in zip(summaries, evals):
        if evaluation is None:
            continue
        summary['eval_mean'] = evaluation.get('mean_return')
        summary['eval_se'] = evaluation.get('standard_error')
        summary['eval_headroom'] = evaluation.get('headroom')
        summary['eval_truthful'] = evaluation.get('truthful_reason_rate')
        if baseline is None:
            baseline = evaluation
            summary['eval_delta'] = 'base'
            continue
        same_battery = all(
            evaluation.get(key) == baseline.get(key)
            for key in ('battery_seed', 'games', 'quarters', 'field_count',
                        'companies_per_field', 'report_coverage',
                        'advisor_coverage', 'advisor_accuracy',
                        'market_noise_sigma')
        )
        returns = evaluation.get('returns') or []
        base_returns = baseline.get('returns') or []
        if not same_battery or len(returns) != len(base_returns):
            continue
        diffs = [a - b for a, b in zip(returns, base_returns)]
        delta = statistics.mean(diffs)
        if len(diffs) > 1:
            error = statistics.stdev(diffs) / (len(diffs) ** 0.5)
        else:
            error = None
        summary['eval_delta_mean'] = round(delta, 2)
        summary['eval_delta_se'] = round(error, 2) if error else None
        significant = error is not None and abs(delta) > 2 * error
        summary['eval_delta'] = '%+.1f%s' % (delta, '*' if significant
                                             else '')


def load_transcripts(run_dir):
    path = Path(run_dir) / 'checkpoints' / 'simtrain' / 'transcripts.jsonl'
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _timepoints(available, steps_spec):
    """Resolve a timepoint request to recorded transcript steps.

    'auto' picks the earliest, middle, and latest recorded steps;
    otherwise each requested step maps to the nearest recorded one, so
    asking for step 100 works whether the trainer logged 99 or 100.
    """
    if not available:
        return []
    if steps_spec == 'auto':
        targets = [available[0], available[len(available) // 2],
                   available[-1]]
    else:
        targets = [int(part) for part in steps_spec.split(',')
                   if part.strip()]
    resolved = []
    for target in targets:
        nearest = min(available, key=lambda step: abs(step - target))
        if nearest not in resolved:
            resolved.append(nearest)
    return resolved


def select_transcripts(records, steps_spec, samples_per_step):
    """Pick sampled moves at the requested timepoints, in step order."""
    available = sorted({record['step'] for record in records
                        if 'step' in record})
    selected = []
    for step in _timepoints(available, steps_spec):
        at_step = [record for record in records
                   if record.get('step') == step]
        selected.extend(at_step[:samples_per_step])
    return selected


def print_transcripts(name, samples):
    for record in samples:
        if record.get('executed'):
            state = 'executed'
        elif record.get('match') == 'none':
            state = 'no move'
        else:
            state = 'not executed'
        earnings = record.get('earnings')
        print('\n--- %s @ step %s | game %s quarter %s | match %s | '
              'earnings %s | %s'
              % (name, record.get('step'), record.get('game'),
                 record.get('quarter'), record.get('match'),
                 '%+.2f' % earnings if earnings is not None else '-',
                 state))
        context = (record.get('context') or '').rstrip()
        if context:
            print(context)
        print('-> %s' % record.get('decision'))


def _tail_mean(rows, key, count=TAIL_ROWS):
    values = [row[key] for row in rows[-count:] if key in row]
    return round(statistics.mean(values), 3) if values else None


def _series(rows, key):
    return [row[key] for row in rows if key in row]


def summarize(name, rows):
    """Reduce one run's history to the comparison row.

    The references come from the run's own final rows, so a sweep whose
    variants end at different difficulties compares on the headroom
    fraction (rolling minus blind, over oracle minus blind) rather than
    on raw returns. The collapse flag reads the tail of the distinct
    rate; a dip that recovered is reported as dip@step instead, since
    rung transitions routinely cause transient dips that say nothing
    about the final model. Replay drift is measured between the mean of
    the first and last tail windows, so it is stable against one noisy
    opening batch - but replay batches are sampled per seed, so the
    number only compares across runs sharing a seed.
    """
    if not rows:
        return {'name': name, 'status': 'no history'}
    last = rows[-1]
    distinct = _series(rows, 'distinct_decision_rate')
    replay = _series(rows, 'replay_loss')
    rolling = _series(rows, 'rolling_return')
    summary = {
        'name': name,
        'status': 'ok',
        'steps': last['step'] + 1,
        'final_rolling': last.get('rolling_return'),
        'best_rolling': round(max(rolling), 2) if rolling else None,
        'blind_reference': last.get('blind_reference'),
        'oracle_reference': last.get('oracle_reference'),
        'distinct_last': _tail_mean(rows, 'distinct_decision_rate'),
        'distinct_min': round(min(distinct), 3) if distinct else None,
        'acted_last': _tail_mean(rows, 'acted_rate'),
        'eligible_last': _tail_mean(rows, 'eligible_rate'),
        'grounded_last': _tail_mean(rows, 'grounded_rate'),
        'truthful_last': _tail_mean(rows, 'truthful_reason_rate'),
        'match_exact_last': _tail_mean(rows, 'match_exact_rate'),
        'match_fuzzy_last': _tail_mean(rows, 'match_fuzzy_rate'),
        'replay_first': (
            round(statistics.mean(replay[:TAIL_ROWS]), 3) if replay else None
        ),
        'replay_last': _tail_mean(rows, 'replay_loss'),
        'rehearsal_last': _tail_mean(rows, 'rehearsal_loss'),
        'anchor_last': _tail_mean(rows, 'anchor_loss'),
        'form_capped_total': sum(_series(rows, 'form_capped')) or None,
        'seconds_per_step': _tail_mean(rows, 'update_seconds'),
    }
    blind = summary['blind_reference']
    oracle = summary['oracle_reference']
    if (summary['final_rolling'] is not None and blind is not None
            and oracle is not None and oracle > blind):
        summary['headroom'] = round(
            (summary['final_rolling'] - blind) / (oracle - blind), 3
        )
    if (summary['replay_first'] is not None
            and summary['replay_last'] is not None):
        summary['replay_drift'] = round(
            summary['replay_last'] - summary['replay_first'], 3
        )
    flags = []
    if distinct and min(distinct) < COLLAPSE_DISTINCT:
        dip_index = min(range(len(distinct)), key=distinct.__getitem__)
        dip_rows = [row for row in rows if 'distinct_decision_rate' in row]
        dip_step = dip_rows[dip_index].get('step')
        if (summary['distinct_last'] is not None
                and summary['distinct_last'] < COLLAPSE_DISTINCT):
            flags.append('collapse')
        else:
            flags.append('dip@%s' % dip_step)
    if last.get('no_signal'):
        flags.append('no-signal')
    if any(row.get('language_training') for row in rows):
        flags.append('language-training')
    summary['flags'] = flags
    return summary


def sweep_runs(sweep_root):
    """Resolve (name, run_dir) pairs for a sweep directory.

    The sweep.json manifest is authoritative when present; otherwise any
    subdirectory carrying a simtrain history counts, so hand-built
    sweeps compare the same way.
    """
    sweep_root = Path(sweep_root)
    manifest_path = sweep_root / 'sweep.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return [(variant['name'], variant['out_dir'])
                for variant in manifest['variants']]
    pairs = []
    for child in sorted(sweep_root.iterdir()):
        if (child / 'checkpoints' / 'simtrain' / 'history.jsonl').exists():
            pairs.append((child.name, str(child)))
    return pairs


_COLUMNS = [
    ('name', 'variant'),
    ('steps', 'steps'),
    ('final_rolling', 'roll'),
    ('best_rolling', 'best'),
    ('blind_reference', 'blind'),
    ('oracle_reference', 'oracle'),
    ('headroom', 'headroom'),
    ('eval_mean', 'eval'),
    ('eval_se', 'ese'),
    ('eval_headroom', 'ehead'),
    ('eval_delta', 'dbase'),
    ('distinct_last', 'dist'),
    ('distinct_min', 'dmin'),
    ('eligible_last', 'elig'),
    ('grounded_last', 'grnd'),
    ('truthful_last', 'truth'),
    ('match_fuzzy_last', 'fuzzy'),
    ('replay_drift', 'rdrift'),
    ('flags', 'flags'),
]


def _format_cell(value):
    if value is None:
        return '-'
    if isinstance(value, list):
        return ','.join(value) if value else '-'
    if isinstance(value, float):
        return '%.2f' % value
    return str(value)


def print_table(summaries):
    table = [[label for _, label in _COLUMNS]]
    for summary in summaries:
        table.append(
            [_format_cell(summary.get(key)) for key, _ in _COLUMNS]
        )
    widths = [
        max(len(row[column]) for row in table)
        for column in range(len(_COLUMNS))
    ]
    for row in table:
        print('  '.join(
            cell.ljust(width) for cell, width in zip(row, widths)
        ).rstrip())


def run(pairs, report_path=None, transcript_steps='auto',
        transcript_samples=1):
    summaries = [summarize(name, load_history(run_dir))
                 for name, run_dir in pairs]
    attach_eval(summaries, [load_eval(run_dir) for _, run_dir in pairs])
    print_table(summaries)
    if transcript_samples > 0:
        for (name, run_dir), summary in zip(pairs, summaries):
            samples = select_transcripts(
                load_transcripts(run_dir), transcript_steps,
                transcript_samples,
            )
            if samples:
                summary['transcripts'] = samples
                print_transcripts(name, samples)
    if report_path is not None:
        with open(report_path, 'w') as handle:
            json.dump({'runs': summaries}, handle, indent=2)
        print('\nreport written to %s' % report_path)
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description='Compare simulation-training runs side by side'
    )
    parser.add_argument('--sweep', help='sweep directory written by '
                                        'slurm/sweep.py')
    parser.add_argument('--runs', nargs='*', help='explicit run trees to '
                                                  'compare')
    parser.add_argument(
        '--transcript-steps', default='auto',
        help='comma-separated steps to sample transcripts at, mapped to '
             'the nearest recorded steps; auto picks early, middle, late',
    )
    parser.add_argument(
        '--transcript-samples', type=int, default=1,
        help='sampled moves per timepoint per run; 0 disables transcripts',
    )
    arguments = parser.parse_args()
    if not arguments.sweep and not arguments.runs:
        parser.error('pass --sweep or --runs')
    pairs = []
    if arguments.sweep:
        pairs.extend(sweep_runs(arguments.sweep))
    for run_dir in arguments.runs or []:
        pairs.append((Path(run_dir).name, run_dir))
    if not pairs:
        raise SystemExit('no runs with a simtrain history found')
    report_path = (
        Path(arguments.sweep) / 'sweep_report.json'
        if arguments.sweep else None
    )
    run(pairs, report_path, arguments.transcript_steps,
        arguments.transcript_samples)


if __name__ == '__main__':
    main()

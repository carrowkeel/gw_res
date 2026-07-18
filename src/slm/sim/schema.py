"""Typed schema for the mission simulator's canonical layer.

This module is the contract every other sim module builds on: dynamics
produces state and event nodes shaped like these specifications, doctrine
produces decisions whose commands come from this vocabulary, and the
serializer renders and parses exactly these structures. Ground truth lives
in the simulator, so anything failing validation here is a bug upstream,
never data to be tolerated.

An episode is a typed graph: nodes carry the mission (system states,
events, messages, decisions, rationale codes, roles) and edges carry its
structure (temporal succession, causality, reference, authority). The SGM
reads varied language but writes only the canonical command vocabulary
defined here, parsed programmatically with no LLM in the loop; a command
that fails validation is a scored error, not a judgment call.

Validation functions return lists of problem strings, empty when valid, so
callers can gate on completeness rather than stop at the first fault.

    python -m slm.sim.schema
"""

import re

ROLES = ('controller', 'operator')

SEVERITIES = ('info', 'caution', 'warning', 'critical')

SYSTEMS = {
    'power': (
        'generation', 'load', 'margin', 'battery_charge',
    ),
    'thermal': (
        'temperature', 'temperature_limit', 'headroom',
    ),
    'comms': (
        'window_open', 'bandwidth', 'ticks_to_next_window',
    ),
    'data': (
        'storage_used', 'storage_capacity', 'storage_fraction',
        'downlink_active',
    ),
    'propellant': (
        'reserve', 'reserve_floor', 'next_burn_cost', 'margin',
    ),
}

EVENT_KINDS = (
    'panel_degradation', 'load_spike', 'heater_fault', 'storm_front',
    'ground_station_outage', 'recorder_fault', 'schedule_slip',
    'sensor_dropout',
)

MESSAGE_KINDS = (
    'status_report', 'event_notice', 'directive', 'query', 'readback',
    'chat',
)

COMMANDS = {
    'shed_load': {'load': 'identifier'},
    'restore_load': {'load': 'identifier'},
    'start_downlink': {'volume': 'number'},
    'stop_downlink': {},
    'suspend_collection': {'instrument': 'identifier'},
    'resume_collection': {'instrument': 'identifier'},
    'hold_burn': {'burn': 'identifier'},
    'release_burn': {'burn': 'identifier'},
    'enter_safe_mode': {},
    'monitor': {},
    'acknowledge': {'reference': 'identifier'},
    'report_status': {'system': 'system'},
    'escalate': {'reference': 'identifier', 'severity': 'severity'},
    'decline': {'reference': 'identifier'},
}

NODE_FIELDS = {
    'role': {'identifier': 'identifier', 'name': 'role'},
    'state': {
        'identifier': 'identifier', 'tick': 'integer', 'system': 'system',
        'metrics': 'metrics',
    },
    'event': {
        'identifier': 'identifier', 'tick': 'integer', 'kind': 'event_kind',
        'system': 'system', 'severity': 'severity', 'magnitude': 'number',
    },
    'message': {
        'identifier': 'identifier', 'tick': 'integer',
        'kind': 'message_kind', 'sender': 'identifier',
        'recipient': 'identifier', 'text': 'text',
    },
    'decision': {
        'identifier': 'identifier', 'tick': 'integer', 'actor': 'identifier',
        'command': 'command',
    },
    'rationale': {
        'identifier': 'identifier', 'code': 'rationale_code', 'text': 'text',
    },
}

EDGE_ENDPOINTS = {
    'succession': {
        'from': ('state', 'event', 'message', 'decision'),
        'to': ('state', 'event', 'message', 'decision'),
    },
    'causality': {
        'from': ('event', 'decision'),
        'to': ('state', 'event'),
    },
    'reference': {
        'from': ('message',),
        'to': ('state', 'event', 'decision'),
    },
    'authority': {
        'from': ('decision',),
        'to': ('rationale',),
    },
}

IDENTIFIER_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
RATIONALE_CODE_PATTERN = re.compile(r'^R_[A-Z][A-Z0-9_]*$')


def _check_value(value, field_kind):
    if field_kind == 'identifier':
        if not isinstance(value, str) or not IDENTIFIER_PATTERN.match(value):
            return 'not a lowercase identifier'
    elif field_kind == 'integer':
        if not isinstance(value, int) or isinstance(value, bool):
            return 'not an integer'
    elif field_kind == 'number':
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 'not a number'
    elif field_kind == 'text':
        if not isinstance(value, str) or not value.strip():
            return 'not non-empty text'
    elif field_kind == 'role':
        if value not in ROLES:
            return 'not a known role'
    elif field_kind == 'system':
        if value not in SYSTEMS:
            return 'not a known system'
    elif field_kind == 'severity':
        if value not in SEVERITIES:
            return 'not a known severity'
    elif field_kind == 'event_kind':
        if value not in EVENT_KINDS:
            return 'not a known event kind'
    elif field_kind == 'message_kind':
        if value not in MESSAGE_KINDS:
            return 'not a known message kind'
    elif field_kind == 'rationale_code':
        if not isinstance(value, str) or not RATIONALE_CODE_PATTERN.match(value):
            return 'not a rationale code'
    return None


def validate_command(command):
    """Validate a canonical command structure against the vocabulary.

    A command is a mapping with a name from COMMANDS and an arguments
    mapping matching that command's argument specification exactly. This is
    the sole gate between SGM output and the simulator, so it accepts
    nothing the vocabulary does not declare.
    """
    problems = []
    if not isinstance(command, dict):
        return ['command is not a mapping']
    name = command.get('name')
    if name not in COMMANDS:
        return ['unknown command name: %r' % (name,)]
    specification = COMMANDS[name]
    arguments = command.get('arguments')
    if not isinstance(arguments, dict):
        return ['command %s arguments are not a mapping' % name]
    for argument_name in specification:
        if argument_name not in arguments:
            problems.append(
                'command %s missing argument %s' % (name, argument_name))
    for argument_name, value in arguments.items():
        if argument_name not in specification:
            problems.append(
                'command %s has unknown argument %s' % (name, argument_name))
            continue
        fault = _check_value(value, specification[argument_name])
        if fault:
            problems.append(
                'command %s argument %s: %s' % (name, argument_name, fault))
    return problems


def validate_node(node):
    """Validate one episode-graph node against its type's field spec."""
    problems = []
    if not isinstance(node, dict):
        return ['node is not a mapping']
    node_type = node.get('type')
    if node_type not in NODE_FIELDS:
        return ['unknown node type: %r' % (node_type,)]
    fields = NODE_FIELDS[node_type]
    label = '%s %s' % (node_type, node.get('identifier', '?'))
    for field_name in fields:
        if field_name not in node:
            problems.append('%s missing field %s' % (label, field_name))
    for field_name, value in node.items():
        if field_name == 'type':
            continue
        if field_name not in fields:
            problems.append('%s has unknown field %s' % (label, field_name))
            continue
        field_kind = fields[field_name]
        if field_kind == 'metrics':
            problems.extend(
                '%s %s' % (label, fault)
                for fault in _check_metrics(value, node.get('system')))
        elif field_kind == 'command':
            problems.extend(
                '%s %s' % (label, fault) for fault in validate_command(value))
        else:
            fault = _check_value(value, field_kind)
            if fault:
                problems.append(
                    '%s field %s: %s' % (label, field_name, fault))
    return problems


def _check_metrics(metrics, system):
    if not isinstance(metrics, dict):
        return ['metrics are not a mapping']
    problems = []
    declared = SYSTEMS.get(system, ())
    for metric_name in declared:
        if metric_name not in metrics:
            problems.append('missing metric %s' % metric_name)
    for metric_name, value in metrics.items():
        if metric_name not in declared:
            problems.append('unknown metric %s' % metric_name)
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append('metric %s is not a number' % metric_name)
    return problems


def validate_edge(edge, node_types_by_identifier):
    """Validate one edge's shape, endpoint existence, and endpoint types."""
    problems = []
    if not isinstance(edge, dict):
        return ['edge is not a mapping']
    edge_type = edge.get('type')
    if edge_type not in EDGE_ENDPOINTS:
        return ['unknown edge type: %r' % (edge_type,)]
    allowed = EDGE_ENDPOINTS[edge_type]
    for endpoint, allowed_types in (('from', allowed['from']),
                                    ('to', allowed['to'])):
        identifier = edge.get(endpoint)
        if identifier not in node_types_by_identifier:
            problems.append(
                '%s edge %s endpoint %r is not a node'
                % (edge_type, endpoint, identifier))
        elif node_types_by_identifier[identifier] not in allowed_types:
            problems.append(
                '%s edge %s endpoint %s has type %s, allowed %s'
                % (edge_type, endpoint, identifier,
                   node_types_by_identifier[identifier],
                   '/'.join(allowed_types)))
    return problems


def validate_episode(episode):
    """Validate a whole episode graph.

    Checks every node and edge, identifier uniqueness, that message and
    decision role references point at role nodes, and that succession
    edges never run backward in time.
    """
    problems = []
    if not isinstance(episode, dict):
        return ['episode is not a mapping']
    nodes = episode.get('nodes')
    edges = episode.get('edges')
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return ['episode must hold node and edge lists']
    node_types = {}
    ticks = {}
    for node in nodes:
        problems.extend(validate_node(node))
        identifier = node.get('identifier') if isinstance(node, dict) else None
        if identifier is None:
            continue
        if identifier in node_types:
            problems.append('duplicate identifier %s' % identifier)
        node_types[identifier] = node.get('type')
        if 'tick' in node:
            ticks[identifier] = node['tick']
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for role_field in ('sender', 'recipient', 'actor'):
            if role_field in node:
                referenced = node.get(role_field)
                if node_types.get(referenced) != 'role':
                    problems.append(
                        '%s %s field %s does not reference a role node'
                        % (node.get('type'), node.get('identifier'),
                           role_field))
    for edge in edges:
        problems.extend(validate_edge(edge, node_types))
        if (isinstance(edge, dict) and edge.get('type') == 'succession'
                and edge.get('from') in ticks and edge.get('to') in ticks
                and ticks[edge['to']] < ticks[edge['from']]):
            problems.append(
                'succession edge %s -> %s runs backward in tick'
                % (edge['from'], edge['to']))
    return problems


def example_episode():
    """A minimal hand-built episode used by the self-check and as a
    reference for what dynamics and doctrine must produce."""
    return {
        'nodes': [
            {'type': 'role', 'identifier': 'flight', 'name': 'controller'},
            {'type': 'role', 'identifier': 'ground', 'name': 'operator'},
            {'type': 'state', 'identifier': 'power_t3', 'tick': 3,
             'system': 'power',
             'metrics': {'generation': 90.0, 'load': 82.0, 'margin': 8.0,
                         'battery_charge': 61.0}},
            {'type': 'event', 'identifier': 'spike_t3', 'tick': 3,
             'kind': 'load_spike', 'system': 'power', 'severity': 'warning',
             'magnitude': 14.0},
            {'type': 'message', 'identifier': 'notice_t3', 'tick': 3,
             'kind': 'event_notice', 'sender': 'ground',
             'recipient': 'flight',
             'text': 'Load stepped up fourteen units on the main bus.'},
            {'type': 'decision', 'identifier': 'shed_t3', 'tick': 3,
             'actor': 'flight',
             'command': {'name': 'shed_load',
                         'arguments': {'load': 'science_instrument'}}},
            {'type': 'rationale', 'identifier': 'why_shed_t3',
             'code': 'R_POWER_MARGIN',
             'text': 'power margin below the low threshold'},
        ],
        'edges': [
            {'type': 'succession', 'from': 'power_t3', 'to': 'spike_t3'},
            {'type': 'causality', 'from': 'spike_t3', 'to': 'power_t3'},
            {'type': 'reference', 'from': 'notice_t3', 'to': 'spike_t3'},
            {'type': 'succession', 'from': 'notice_t3', 'to': 'shed_t3'},
            {'type': 'authority', 'from': 'shed_t3', 'to': 'why_shed_t3'},
        ],
    }


def main():
    episode = example_episode()
    problems = validate_episode(episode)
    print('example episode: %d problems' % len(problems))
    for problem in problems:
        print('  ' + problem)
    broken_cases = [
        ('unknown command',
         {'name': 'launch_missiles', 'arguments': {}}),
        ('missing argument',
         {'name': 'shed_load', 'arguments': {}}),
        ('wrong argument type',
         {'name': 'escalate',
          'arguments': {'reference': 'spike_t3', 'severity': 'extreme'}}),
    ]
    for label, command in broken_cases:
        faults = validate_command(command)
        state = 'caught' if faults else 'MISSED'
        print('%s: %s' % (label, state))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())

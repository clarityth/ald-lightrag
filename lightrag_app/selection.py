"""Select searched measurements and evidence-linked concept neighbors."""
from collections import defaultdict
import re


def select_measurements(records, graph, query_concepts, sdk_data, limit=32, retrieved_ids=()):
    returned = {entity.get('entity_name') for entity in sdk_data.get('entities', [])}
    for edge in sdk_data.get('relationships', []):
        returned.update((edge.get('src_id'), edge.get('tgt_id')))
    source_ids = set(retrieved_ids)
    for chunk in sdk_data.get('chunks', []):
        match = re.match(r'^\[([^]]+)\]', chunk.get('content') or '')
        if match:
            source_ids.add(match.group(1))

    query = set(query_concepts)
    seeds = {name for name in returned if isinstance(name, str) and name.startswith('concept:')}
    experiments = {experiment['id']: experiment for experiment in records.get('experiments', [])}
    rows = []
    for measurement in records['measurements']:
        concepts = (set(graph['experiment_concepts'].get(measurement['experiment_id'], [])) |
                    set(graph['measurement_concepts'].get(measurement['id'], [])))
        experiment = experiments.get(measurement['experiment_id'], {})
        direct = (measurement['id'] in returned or
                  bool(source_ids.intersection(measurement['source_ids'])))
        indirect = (measurement['experiment_id'] in returned or
                    bool(source_ids.intersection(experiment.get('source_ids', []))) or
                    any(item['paragraph_id'] in source_ids for item in experiment.get('purpose_evidence', [])))
        tier = 0 if direct else 1 if indirect else 2
        if tier < 2:
            seeds.update(concepts)
        rows.append((measurement, concepts, tier))

    candidates = []
    for measurement, concepts, tier in rows:
        matched = concepts & seeds
        if tier < 2 or matched:
            rank = (-len(concepts & query), tier, -len(matched), measurement['id'])
            candidates.append((rank, measurement, matched))

    selected, paths = [], []
    labels = {entity['entity_name']: entity['description'] for entity in graph['entities']}
    groups = defaultdict(list)
    for _, measurement, matched in sorted(candidates, key=lambda row: row[0]):
        groups[measurement['doc_id']].append((measurement, matched))
    for position in range(max(map(len, groups.values()), default=0)):
        for paper_rows in groups.values():
            if position >= len(paper_rows) or len(selected) >= limit:
                continue
            measurement, matched = paper_rows[position]
            selected.append(measurement)
            for concept in sorted(matched):
                paths.append({'concept_id': concept, 'concept_label': labels.get(concept, concept),
                              'doc_id': measurement['doc_id'], 'experiment_id': measurement['experiment_id'],
                              'measurement_id': measurement['id'], 'paragraph_ids': measurement['source_ids']})
    return selected, paths

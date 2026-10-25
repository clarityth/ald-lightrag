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
        linked_ids = (set(measurement['source_ids']) | set(experiment.get('source_ids', [])) |
                      {item['paragraph_id'] for item in experiment.get('purpose_evidence', [])})
        direct = (measurement['id'] in returned or measurement['experiment_id'] in returned or
                  bool(source_ids.intersection(linked_ids)))
        if direct:
            seeds.update(concepts)
        rows.append((measurement, concepts, direct))

    candidates = []
    for measurement, concepts, direct in rows:
        matched = concepts & seeds
        if direct or matched:
            rank = (not direct, -len(concepts & query), -len(matched), measurement['id'])
            candidates.append((rank, measurement, matched))

    selected, paths = [], []
    labels = {entity['entity_name']: entity['description'] for entity in graph['entities']}
    # Keep direct evidence ahead of expansions, with round-robin papers within each tier.
    for direct_tier in (False, True):
        groups = defaultdict(list)
        for rank, measurement, matched in sorted(candidates, key=lambda row: row[0]):
            if rank[0] == direct_tier:
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

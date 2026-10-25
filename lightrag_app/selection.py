"""Question-independent traversal over normalized experiment and metric links."""
from collections import defaultdict


def select_measurements(records, graph, query_concepts, sdk_data, limit=32):
    returned = {e.get('entity_name') for e in sdk_data.get('entities', [])}
    for edge in sdk_data.get('relationships', []):
        returned.update((edge.get('src_id'), edge.get('tgt_id')))
    seeds = set(query_concepts) & returned
    if not seeds:
        return [], []
    by_category = defaultdict(set)
    for concept in query_concepts:
        by_category[concept.split(':')[1]].add(concept)
    candidates = []
    for measurement in records['measurements']:
        concepts = set(graph['experiment_concepts'].get(measurement['experiment_id'], [])) | set(graph['measurement_concepts'].get(measurement['id'], []))
        if not concepts & seeds:
            continue
        if any(not concepts & values for values in by_category.values()):
            continue
        rank = (measurement['id'] not in returned, -len(concepts & seeds), measurement['id'])
        candidates.append((rank, measurement, concepts & seeds))
    # Round-robin papers prevents one prolific paper from filling every slot.
    groups = defaultdict(list)
    for _, measurement, matched in sorted(candidates, key=lambda row: row[0]):
        groups[measurement['doc_id']].append((measurement, matched))
    selected, paths = [], []
    labels = {e['entity_name']:e['description'] for e in graph['entities']}
    for position in range(max(map(len, groups.values()), default=0)):
        for rows in groups.values():
            if position >= len(rows) or len(selected) >= limit:
                continue
            measurement, matched = rows[position]
            selected.append(measurement)
            for concept in sorted(matched):
                paths.append({'concept_id':concept,'concept_label':labels.get(concept,concept), 'doc_id':measurement['doc_id'],'experiment_id':measurement['experiment_id'],'measurement_id':measurement['id'],'paragraph_ids':measurement['source_ids']})
    return selected, paths

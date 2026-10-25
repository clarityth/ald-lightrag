"""Deterministic, question-independent concepts for validated experiment records."""

import json
import re
from pathlib import Path


_DICTIONARY = json.loads(
    (Path(__file__).resolve().parent.parent / "config/concepts.json").read_text(encoding="utf-8")
)
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_CATEGORIES = ("material", "substrate", "device", "process", "metric")
_FIELDS = {"sample": _CATEGORIES[:3], "varied_parameter": ("process",), "metric": ("metric",)}


def _normalize(text):
    return text.translate(_SUBSCRIPTS).casefold()


def _contains(text, alias):
    pattern = rf"(?<![a-z0-9]){re.escape(_normalize(alias))}(?![a-z0-9])"
    for match in re.finditer(pattern, text):
        if _normalize(alias) == "silicon" and re.match(r"\s+dioxide\b", text[match.end():]):
            continue
        return True
    return False


def _matching(text, categories):
    normalized = _normalize(text)
    return [
        (entry["category"], f"concept:{entry['category']}:{entry['slug']}", entry)
        for entry in _DICTIONARY
        if entry["category"] in categories
        and any(_contains(normalized, alias) for alias in entry["aliases"])
    ]


def match_query(text):
    """Return canonical concept IDs from the public static dictionary."""
    return [concept_id for _, concept_id, _ in _matching(text, _CATEGORIES)]


def _source(record):
    ids = record.get("source_ids") or [item["paragraph_id"] for item in record.get("evidence", [])]
    doc_id = record["doc_id"]
    if not ids or not ids[0].startswith(f"{doc_id}:"):
        raise ValueError(f"개념 간선의 문헌 근거가 없습니다: {record['id']}")
    return ids[0]


def build_concepts(records):
    """Build SDK-ready concept entities and evidence-backed record→concept edges."""
    entities = {}
    relationships = []
    experiment_concepts = {}
    measurement_concepts = {}
    papers = {}
    unmapped = {category: 0 for category in _CATEGORIES}

    for kind, rows, fields, destination in (
        ("experiment", records["experiments"], ("sample", "varied_parameter"), experiment_concepts),
        ("measurement", records["measurements"], ("metric",), measurement_concepts),
    ):
        for record in rows:
            matches = []
            for field in fields:
                found = _matching(record.get(field, ""), _FIELDS[field])
                matches.extend(found)
                for category in _FIELDS[field]:
                    if not any(item[0] == category for item in found):
                        unmapped[category] += 1
            destination[record["id"]] = []
            if not matches:
                continue
            source_id = _source(record)
            for category, concept_id, entry in matches:
                if concept_id in destination[record["id"]]:
                    continue
                destination[record["id"]].append(concept_id)
                papers.setdefault(concept_id, set()).add(record["doc_id"])
                entities.setdefault(concept_id, {
                    "entity_name": concept_id,
                    "entity_type": "concept",
                    "description": f"{entry['name']} ({category})",
                    "source_id": source_id,
                    "file_path": record["doc_id"],
                })
                relationships.append({
                    "src_id": record["id"], "tgt_id": concept_id,
                    "description": f"{kind} {record['id']} has {category} {entry['name']}",
                    "keywords": f"{entry['name']}, {category}",
                    "source_id": source_id, "file_path": record["doc_id"],
                })

    experiment_count = len(records["experiments"])
    measurement_count = len(records["measurements"])
    experiments_mapped = sum(bool(ids) for ids in experiment_concepts.values())
    measurements_mapped = sum(bool(ids) for ids in measurement_concepts.values())
    return {
        "entities": list(entities.values()),
        "relationships": relationships,
        "experiment_concepts": experiment_concepts,
        "measurement_concepts": measurement_concepts,
        "stats": {
            "concept_count": len(entities),
            "multi_paper_concepts": sum(len(doc_ids) > 1 for doc_ids in papers.values()),
            "experiment_count": experiment_count,
            "measurement_count": measurement_count,
            "experiments_mapped": experiments_mapped,
            "measurements_mapped": measurements_mapped,
            "experiment_mapping_rate": experiments_mapped / experiment_count if experiment_count else 0.0,
            "measurement_mapping_rate": measurements_mapped / measurement_count if measurement_count else 0.0,
            "unmapped": unmapped,
        },
    }

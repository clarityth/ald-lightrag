"""Offline, retrospective context-budget comparison on the frozen H01-H12 run."""

import hashlib
import json
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "mechanism_poc/sources/dataset.json"
QUESTIONS = ROOT / "evaluation/questions-007.json"
COMPARISON = ROOT / "reports/evidence-preservation/comparison.json"
BACKEND = ROOT / "lightrag_app/backend.py"
OUTPUT = ROOT / "reports/comparison-workspace/budget.json"
BUDGETS = (4000, 8000, 12000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context(ids, paragraphs):
    # Match backend._context/_chunk_context for raw source paragraphs exactly.
    return "\n\n".join(
        f"[{number}] {paragraphs[pid][0]['title']} ({pid}, {paragraphs[pid][1]['section']})\n{paragraphs[pid][1]['text']}"
        for number, pid in enumerate(ids, 1)
    )


def fit(seed, candidates, budget, paragraphs, encoding):
    selected = []
    skipped = []
    for pid in seed + candidates:
        if pid in selected:
            continue
        if len(encoding.encode(context(selected + [pid], paragraphs))) <= budget:
            selected.append(pid)
        else:
            skipped.append(pid)
    return selected, skipped, len(encoding.encode(context(selected, paragraphs)))


def main():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    graph_rows = [row for row in json.loads(COMPARISON.read_text(encoding="utf-8")) if row["mode"] == "graph"]
    assert len(questions) == len(graph_rows) == 12
    rows = {row["question_id"]: row for row in graph_rows}
    assert len(rows) == 12
    documents = {doc["id"]: doc for doc in dataset["documents"]}
    paragraphs = {p["id"]: (doc, p) for doc in dataset["documents"] for p in doc["paragraphs"]}
    assert len(paragraphs) == sum(len(doc["paragraphs"]) for doc in dataset["documents"])
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    results = []
    for question in questions:
        row = rows[question["id"]]
        assert row["question"] == question["question"]
        seed = list(dict.fromkeys(row["retrieved_paragraph_ids"]))
        graph = [citation["paragraph_id"] for citation in row["citations"]]
        assert len(graph) == len(set(graph)) and graph[:len(seed)] == seed
        assert all(pid in paragraphs for pid in graph)
        assert len(encoding.encode(context(graph, paragraphs))) == row["context_tokens"]
        docs = list(dict.fromkeys(paragraphs[pid][0]["id"] for pid in seed))
        by_doc = [[p["id"] for p in documents[doc_id]["paragraphs"] if p["id"] not in seed] for doc_id in docs]
        expanded = [pid for rank in range(max(map(len, by_doc), default=0)) for paper in by_doc if rank < len(paper) for pid in [paper[rank]]]
        required = question["required_measurements"]
        for budget in BUDGETS:
            for mode, candidates in (("graph", graph[len(seed):]), ("document_expansion", expanded)):
                selected, skipped, tokens = fit(seed, candidates, budget, paragraphs, encoding)
                assert tokens <= budget and len(selected) == len(set(selected))
                selected_set = set(selected)
                recovered = [item["measurement_id"] for item in required if set(item["paragraph_ids"]) <= selected_set]
                results.append({
                    "question_id": question["id"], "budget": budget, "mode": mode,
                    "seed_paragraph_ids": seed, "source_doc_ids": docs,
                    "selected_paragraph_ids": selected, "skipped_overflow_paragraph_ids": skipped,
                    "context_tokens": tokens, "required_count": len(required),
                    "recovered_measurement_ids": recovered, "recovered_count": len(recovered),
                })
    for graph, expansion in zip(results[::2], results[1::2]):
        assert (graph["question_id"], graph["budget"]) == (expansion["question_id"], expansion["budget"])
        assert [pid for pid in graph["seed_paragraph_ids"] if pid in graph["selected_paragraph_ids"]] == [
            pid for pid in expansion["seed_paragraph_ids"] if pid in expansion["selected_paragraph_ids"]
        ]
    summary = []
    for budget in BUDGETS:
        for mode in ("graph", "document_expansion"):
            group = [row for row in results if row["budget"] == budget and row["mode"] == mode]
            summary.append({"budget": budget, "mode": mode, "questions": len(group),
                            "recovered_count": sum(row["recovered_count"] for row in group),
                            "required_count": sum(row["required_count"] for row in group),
                            "context_tokens_total": sum(row["context_tokens"] for row in group),
                            "context_tokens_mean": round(sum(row["context_tokens"] for row in group) / len(group), 1)})
    output = {
        "scope": "Retrospective context reconstruction from the existing 12 questions; no retrieval, answer generation, model calls, or timing measurement.",
        "inputs_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (DATASET, QUESTIONS, COMPARISON, BACKEND)},
        "tokenizer": "tiktoken.encoding_for_model('gpt-4o-mini')",
        "context_format": "backend._context raw source paragraph format, including citation numbers and separators",
        "budgets": list(BUDGETS), "summary": summary, "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()

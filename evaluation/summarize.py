"""Summarize the frozen comparison and source citation audits without API calls."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODES = ("all", "naive", "graph")
AUDITS = ("01-05", "06-10", "11-15")


def summarize():
    rows = json.loads((ROOT / "reports/comparison.json").read_text(encoding="utf-8"))
    audits = [
        item
        for span in AUDITS
        for item in json.loads((ROOT / f"reports/citation-audit-{span}.json").read_text(encoding="utf-8"))
    ]
    by_key = {(row["question_id"], row["mode"]): row for row in rows}
    audit_by_key = {(row["question_id"], row["mode"]): row for row in audits}
    question_ids = {row["question_id"] for row in rows}
    assert len(rows) == len(audits) == len(by_key) == len(audit_by_key) == 45
    assert set(by_key) == set(audit_by_key) == {(qid, mode) for qid in question_ids for mode in MODES}
    assert len(question_ids) == 15
    assert all(len({by_key[qid, mode]["required_count"] for mode in MODES}) == 1 for qid in question_ids)
    assert all(len({by_key[qid, mode]["kind"] for mode in MODES}) == 1 for qid in question_ids)

    result = {}
    for mode in MODES:
        selected = [by_key[qid, mode] for qid in sorted(question_ids)]
        reviews = [audit_by_key[qid, mode] for qid in sorted(question_ids)]
        required = sum(row["required_count"] for row in selected)
        recovered = sum(len(set(row["recovered_measurements"])) for row in selected)
        assert required == 55
        assert all(len(row["recovered_measurements"]) == len(set(row["recovered_measurements"])) <= row["required_count"] for row in selected)
        statuses = Counter(claim["status"] for review in reviews for claim in review["claims"])
        assert set(statuses) <= {"supported", "unsupported", "unclear"}
        claim_count = sum(statuses.values())
        by_kind = {}
        for kind in ("single_paper", "multiple_papers"):
            kind_rows = [row for row in selected if row["kind"] == kind]
            kind_required = sum(row["required_count"] for row in kind_rows)
            kind_recovered = sum(len(row["recovered_measurements"]) for row in kind_rows)
            by_kind[kind] = {"recovered": kind_recovered, "required": kind_required,
                             "rate": kind_recovered / kind_required}
        result[mode] = {
            "evidence_recall": {"recovered": recovered, "required": required, "rate": recovered / required},
            "numeric_citation": {"supported": statuses["supported"], "total": claim_count,
                                 "rate": statuses["supported"] / claim_count if claim_count else None,
                                 "unsupported": statuses["unsupported"], "unclear": statuses["unclear"]},
            "context_tokens": {"mean": sum(row["context_tokens"] for row in selected) / 15,
                               "total": sum(row["context_tokens"] for row in selected)},
            "seconds": {"mean": sum(row["seconds"] for row in selected) / 15,
                        "total": sum(row["seconds"] for row in selected)},
            "estimated_cost_usd": {"mean": sum(row["estimated_cost_usd"] for row in selected) / 15,
                                   "total": sum(row["estimated_cost_usd"] for row in selected)},
            "answers_without_numeric_claims": sum(not review["claims"] for review in reviews),
            "evidence_recall_by_kind": by_kind,
        }
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))

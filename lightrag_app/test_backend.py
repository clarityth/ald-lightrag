import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import tiktoken

from lightrag_app import backend
from lightrag_app.experiments import validate


class BackendTest(unittest.TestCase):
    def test_custom_kg_keeps_all_paragraphs_and_scoped_observations(self):
        dataset = backend._dataset()
        payload = backend._payload(dataset)
        source_ids = {chunk["source_id"] for chunk in payload["chunks"]}
        observations = [e for e in payload["entities"] if e["entity_type"] == "paper_observation"]
        self.assertEqual(len(source_ids), 309)
        self.assertEqual(len(observations), 27)
        self.assertEqual(len(payload["relationships"]), 54)
        self.assertTrue(all(edge["source_id"] in source_ids for edge in payload["relationships"]))
        self.assertTrue(all(
            relation["sample_context"] in next(e["description"] for e in observations if relation["id"] in e["entity_name"])
            for relation in dataset["relations"]
        ))

    def test_sdk_failure_is_not_reported_as_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "index.json"
            marker.write_text(json.dumps({"dataset_sha256": hashlib.sha256(backend.DATASET.read_bytes()).hexdigest(), "extraction_version": backend.EXTRACTION_VERSION}))
            records = Path(directory) / "experiments.json"
            records.write_text(json.dumps({"experiments": [], "measurements": []}))
            rag = unittest.mock.Mock()
            rag.initialize_storages = AsyncMock()
            rag.finalize_storages = AsyncMock()
            rag.aquery_data = AsyncMock(return_value={"status": "failure", "message": "query failed", "data": {}})
            with patch.object(backend, "MARKER", marker), patch.object(backend, "RECORDS", records), patch.object(backend, "_rag", return_value=rag), patch.object(
                backend, "_search_queries", new=AsyncMock(return_value=["test question"])
            ):
                with self.assertRaisesRegex(RuntimeError, "LightRAG 검색 실패"):
                    asyncio.run(backend._ask("test question"))

    def test_same_paragraph_keeps_distinct_experiments_and_measurement_citations(self):
        dataset = backend._dataset()
        doc = dataset["documents"][0]
        paragraph = doc["paragraphs"][0]
        quote = paragraph["text"][: min(40, len(paragraph["text"]))]
        evidence = [{"paragraph_id": paragraph["id"], "quote": quote}]
        raw = {
            "experiments": [
                {"id": "E1", "sample": "sample A", "varied_parameter": "100 C", "fixed_conditions": "same substrate", "purpose": "author-stated test", "purpose_evidence": evidence, "evidence": evidence},
                {"id": "E2", "sample": "sample B", "varied_parameter": "200 C", "fixed_conditions": "same substrate", "purpose": "unknown", "purpose_evidence": [], "evidence": evidence},
            ],
            "measurements": [
                {"id": "M1", "experiment_id": "E1", "metric": "thickness", "value": "1", "unit": "nm", "conditions": "room temperature", "evidence": evidence},
                {"id": "M2", "experiment_id": "E2", "metric": "thickness", "value": "2", "unit": "nm", "conditions": "room temperature", "evidence": evidence},
                {"id": "M3", "experiment_id": "E2", "metric": "thickness", "value": "3", "unit": "nm", "conditions": "unknown", "evidence": [{"paragraph_id": paragraph["id"], "quote": "not present in source"}]},
            ],
        }
        records = validate(doc, raw)
        self.assertEqual(records["rejected"], 1)
        self.assertEqual(len(records["experiments"]), 2)
        self.assertEqual(len(records["measurements"]), 2)
        self.assertEqual(records["experiments"][0]["purpose_evidence"], evidence)
        source = {paragraph["id"]: (doc, paragraph)}
        chunks = backend._measurement_chunks(records["measurements"], records["experiments"], source)
        self.assertNotEqual(chunks[0]["id"], chunks[1]["id"])
        self.assertEqual([chunk["original_paragraph_id"] for chunk in chunks], [paragraph["id"]] * 2)
        self.assertEqual([chunk["measurement_ids"] for chunk in chunks], [[f"{doc['id']}:M1"], [f"{doc['id']}:M2"]])
        self.assertIn("100 C", chunks[0]["context_text"])
        self.assertIn("200 C", chunks[1]["context_text"])
        self.assertIn("author-stated test", chunks[0]["context_text"])
        self.assertEqual(chunks[0]["context_text"].count("변경 조건"), 1)
        self.assertEqual(chunks[1]["context_text"].count("변경 조건"), 1)
        payload = backend._payload(dataset, records)
        measured_edges = [edge for edge in payload["relationships"] if edge["tgt_id"] in {f"{doc['id']}:M1", f"{doc['id']}:M2"}]
        self.assertEqual({edge["src_id"] for edge in measured_edges}, {f"{doc['id']}:E1", f"{doc['id']}:E2"})

    def test_context_budget_never_discards_retrieved_chunks_for_expansion(self):
        raw = [
            {"id": name, "doc_id": "D", "title": "Paper", "section": "Results", "text": name * 100}
            for name in ("r1", "r2", "extra")
        ]
        fact = [{"id": "fact", "measurement_ids": ["M1"], "context_text": "measurement " * 100}]
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        limit = len(encoding.encode(backend._context([raw[1], raw[0]]))) + 5
        chunks, omitted_raw, omitted_facts, count = backend._fit_context(
            raw, fact, ["r2", "r1"], ["D"], limit
        )
        self.assertEqual([chunk["id"] for chunk in chunks[:2]], ["r2", "r1"])
        self.assertEqual(omitted_raw, ["extra"])
        self.assertEqual(omitted_facts, ["M1"])
        self.assertLessEqual(count, limit)

    def test_unquoted_purpose_becomes_unknown_without_losing_experiment(self):
        doc = backend._dataset()["documents"][0]
        paragraph = doc["paragraphs"][0]
        records = validate(doc, {"experiments": [{
            "id": "E1", "sample": "sample", "varied_parameter": "condition",
            "fixed_conditions": "fixed", "purpose": "gate dielectric",
            "purpose_evidence": [{"paragraph_id": paragraph["id"], "quote": "not in the source paragraph"}],
            "evidence": [{"paragraph_id": paragraph["id"], "quote": paragraph["text"][:20]}],
        }], "measurements": []})
        self.assertEqual(len(records["experiments"]), 1)
        self.assertEqual(records["experiments"][0]["purpose"], "unknown")
        self.assertEqual(records["experiments"][0]["purpose_evidence"], [])
        self.assertEqual(records["purpose_rejected"], 1)
        self.assertEqual(records["rejected"], 0)

if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path

from lightrag_app.concepts import build_concepts, match_query


RECORDS = Path(__file__).resolve().parent / "storage-experiments-v3/experiments.json"
DATASET = Path(__file__).resolve().parent.parent / "mechanism_poc/sources/dataset.json"


class ConceptTest(unittest.TestCase):
    def test_field_boundaries_and_evidence_edges(self):
        records = {
            "experiments": [{
                "id": "P1:E1", "doc_id": "P1", "sample": "HfO2 on SiO2; MOS capacitor",
                "varied_parameter": "ALD purge time series", "fixed_conditions": "annealing temperature 500 C",
                "source_ids": ["P1:p1"],
            }],
            "measurements": [{
                "id": "P1:M1", "doc_id": "P1", "experiment_id": "P1:E1",
                "metric": "Interface trap density (Dit)", "conditions": "deposition temperature 250 C",
                "source_ids": ["P1:p2"],
            }],
        }
        result = build_concepts(records)
        experiment = set(result["experiment_concepts"]["P1:E1"])
        measurement = set(result["measurement_concepts"]["P1:M1"])
        self.assertIn("concept:material:hfo2", experiment)
        self.assertIn("concept:substrate:sio2", experiment)
        self.assertIn("concept:device:mos-capacitor", experiment)
        self.assertIn("concept:process:ald-purge-time", experiment)
        self.assertNotIn("concept:substrate:si", experiment)
        self.assertNotIn("concept:process:annealing-temperature", experiment)
        self.assertEqual(measurement, {"concept:metric:interface-trap-density"})
        self.assertEqual(len(result["relationships"]), len(experiment) + len(measurement))
        self.assertEqual(
            {(edge["src_id"], edge["source_id"], edge["file_path"]) for edge in result["relationships"]},
            {("P1:E1", "P1:p1", "P1")} | {("P1:M1", "P1:p2", "P1")},
        )
        self.assertTrue(all(edge["tgt_id"].startswith("concept:") for edge in result["relationships"]))
        self.assertTrue(all(entity["source_id"] == "P1:p1" or entity["source_id"] == "P1:p2" for entity in result["entities"]))

    def test_shared_concept_and_unmapped_breakdown(self):
        records = {"experiments": [
            {"id": "A:E1", "doc_id": "A", "sample": "HfO₂ on Si", "varied_parameter": "mystery step", "source_ids": ["A:p1"]},
            {"id": "B:E1", "doc_id": "B", "sample": "HfO2 on silicon", "varied_parameter": "RTA temperature", "source_ids": ["B:p1"]},
        ], "measurements": [
            {"id": "A:M1", "doc_id": "A", "metric": "unclassified readout", "source_ids": ["A:p2"]},
            {"id": "B:M1", "doc_id": "B", "metric": "Dit", "source_ids": ["B:p2"]},
        ]}
        result = build_concepts(records)
        self.assertEqual({"concept:material:hfo2", "concept:substrate:si"}, set(result["experiment_concepts"]["A:E1"]))
        self.assertEqual(result["measurement_concepts"]["A:M1"], [])
        self.assertGreaterEqual(result["stats"]["multi_paper_concepts"], 2)
        self.assertEqual(result["stats"]["unmapped"]["process"], 1)
        self.assertEqual(result["stats"]["unmapped"]["metric"], 1)
        self.assertEqual(result["stats"]["experiments_mapped"], 2)
        self.assertEqual(result["stats"]["measurements_mapped"], 1)
        self.assertEqual(result["stats"]["experiment_mapping_rate"], 1.0)
        self.assertEqual(result["stats"]["measurement_mapping_rate"], 0.5)

    def test_query_aliases_are_static_and_boundary_aware(self):
        self.assertIn("concept:material:hfo2", match_query("산화하프늄 HfO₂ 박막의 Dit를 비교"))
        self.assertIn("concept:metric:interface-trap-density", match_query("산화하프늄 HfO₂ 박막의 Dit를 비교"))
        self.assertIn("concept:process:ald-purge-time", match_query("ALD 퍼지 시간에 따른 두께"))
        self.assertNotIn("concept:substrate:si", match_query("SiO2 막 위의 HfO2"))
        self.assertNotIn("concept:substrate:si", match_query("silicon dioxide 막"))
        self.assertEqual(match_query("아무 관련 없는 질문 12345"), [])

    def test_real_records_have_practical_coverage(self):
        if not RECORDS.exists():
            self.skipTest("ignored extraction cache is unavailable")
        result = build_concepts(json.loads(RECORDS.read_text(encoding="utf-8")))
        stats = result["stats"]
        self.assertEqual(stats["experiment_count"], 93)
        self.assertEqual(stats["measurement_count"], 291)
        self.assertEqual(stats["experiments_mapped"], 93)
        self.assertGreaterEqual(stats["measurement_mapping_rate"], 0.9)
        self.assertGreater(stats["multi_paper_concepts"], 0)
        self.assertEqual(len(result["experiment_concepts"]), 93)
        self.assertEqual(len(result["measurement_concepts"]), 291)
        paragraphs = {
            paragraph["id"]: document["id"]
            for document in json.loads(DATASET.read_text(encoding="utf-8"))["documents"]
            for paragraph in document["paragraphs"]
        }
        for item in result["entities"] + result["relationships"]:
            self.assertEqual(paragraphs[item["source_id"]], item["file_path"])


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from evaluation.run import run

class EvaluationTest(unittest.TestCase):
    def test_existing_report_from_other_questions_is_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            questions=Path(directory)/'questions.json'
            report=Path(directory)/'report.json'
            questions.write_text(json.dumps([{'id':'H01'}]))
            original=json.dumps([{'question_id':'Q01','mode':'graph'}])
            report.write_text(original)
            with self.assertRaisesRegex(AssertionError,'different questions'):
                asyncio.run(run(questions,report,['graph']))
            self.assertEqual(report.read_text(),original)

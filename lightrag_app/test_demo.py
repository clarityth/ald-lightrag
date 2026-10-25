import unittest
from lightrag_app.serve import load_examples

class DemoTest(unittest.TestCase):
    def test_public_example_has_real_answer_and_resolvable_measurements(self):
        rows = load_examples()
        self.assertTrue(rows)
        row = rows[0]
        self.assertTrue(row['answer'])
        self.assertNotIn('metadata', row)
        measurements = {m['id']: m for m in row['measurements']}
        experiments = {e['id'] for e in row['experiments']}
        self.assertTrue(measurements)
        for chunk in row['chunks']:
            for mid in chunk.get('measurement_ids', []):
                self.assertIn(mid, measurements)
                self.assertIn(measurements[mid]['experiment_id'], experiments)

if __name__ == '__main__':
    unittest.main()

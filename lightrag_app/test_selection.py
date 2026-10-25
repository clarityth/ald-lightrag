import unittest
from lightrag_app.selection import select_measurements

class SelectionTest(unittest.TestCase):
    def test_sdk_concept_traversal_crosses_papers_without_full_document_expansion(self):
        records={'experiments':[{'id':'E1','doc_id':'D1'},{'id':'E2','doc_id':'D2'},{'id':'E3','doc_id':'D1'}], 'measurements':[{'id':'M1','experiment_id':'E1','doc_id':'D1','source_ids':['p1']},{'id':'M2','experiment_id':'E2','doc_id':'D2','source_ids':['p2']},{'id':'M3','experiment_id':'E3','doc_id':'D1','source_ids':['p3']}]}
        c='concept:variable:purge'; other='concept:variable:anneal'
        graph={'entities':[{'entity_name':c,'description':'Purge time'}], 'experiment_concepts':{'E1':[c],'E2':[c],'E3':[other]},'measurement_concepts':{}}
        sdk={'entities':[], 'relationships':[{'src_id':'E1','tgt_id':c}]}
        chosen,paths=select_measurements(records,graph,[c],sdk)
        self.assertEqual({x['id'] for x in chosen},{'M1','M2'})
        self.assertEqual({x['doc_id'] for x in paths},{'D1','D2'})
        self.assertEqual(select_measurements(records,graph,[c],{})[0],[])

    def test_missing_query_category_does_not_drop_retrieved_measurement(self):
        variable = 'concept:variable:purge'
        metric = 'concept:metric:dit'
        records = {'measurements': [
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['p1']},
        ]}
        graph = {'entities': [{'entity_name': variable, 'description': 'Purge time'}],
                 'experiment_concepts': {'E1': [variable]}, 'measurement_concepts': {}}
        sdk = {'entities': [{'entity_name': 'M1'}, {'entity_name': variable}], 'relationships': []}
        chosen, paths = select_measurements(records, graph, [variable, metric], sdk)
        self.assertEqual([item['id'] for item in chosen], ['M1'])
        self.assertEqual([item['concept_id'] for item in paths], [variable])

    def test_sdk_measurement_and_source_paragraph_survive_without_query_concepts(self):
        records = {'measurements': [
            {'id': 'M3', 'experiment_id': 'E3', 'doc_id': 'D3', 'source_ids': ['p3']},
            {'id': 'M2', 'experiment_id': 'E2', 'doc_id': 'D2', 'source_ids': ['p2']},
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['p1']},
        ]}
        graph = {'entities': [], 'experiment_concepts': {}, 'measurement_concepts': {}}
        sdk = {'entities': [{'entity_name': 'M1'}], 'relationships': [],
               'chunks': [{'content': '[p2] Source paragraph'}]}
        for _ in range(2):
            chosen, paths = select_measurements(records, graph, [], sdk)
            self.assertEqual([item['id'] for item in chosen], ['M1', 'M2'])
            self.assertEqual(paths, [])

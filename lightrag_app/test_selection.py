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

    def test_retrieved_measurements_outlast_experiment_links_within_document(self):
        records = {'experiments': [
            {'id': 'E1', 'source_ids': ['e1']},
            {'id': 'E2', 'source_ids': ['e2']},
            {'id': 'E3', 'source_ids': ['e3']},
            {'id': 'E4', 'source_ids': ['e4']},
        ], 'measurements': [
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['m1']},
            {'id': 'M2', 'experiment_id': 'E2', 'doc_id': 'D1', 'source_ids': ['m2']},
            {'id': 'M3', 'experiment_id': 'E3', 'doc_id': 'D1', 'source_ids': ['m3']},
            {'id': 'M4', 'experiment_id': 'E4', 'doc_id': 'D1', 'source_ids': ['m4']},
        ]}
        graph = {'entities': [], 'experiment_concepts': {}, 'measurement_concepts': {}}
        sdk = {'entities': [{'entity_name': 'E1'}, {'entity_name': 'M4'}],
               'chunks': [{'content': '[e2] Experiment source'},
                          {'content': '[m3] Measurement source'}]}
        chosen, _ = select_measurements(records, graph, [], sdk, limit=2)
        self.assertEqual([item['id'] for item in chosen], ['M3', 'M4'])

    def test_concept_neighbor_from_other_document_survives_limit(self):
        concept = 'concept:variable:purge'
        records = {'measurements': [
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['m1']},
            {'id': 'M2', 'experiment_id': 'E2', 'doc_id': 'D1', 'source_ids': ['m2']},
            {'id': 'M3', 'experiment_id': 'E3', 'doc_id': 'D2', 'source_ids': ['m3']},
        ]}
        graph = {'entities': [{'entity_name': concept, 'description': 'Purge time'}],
                 'experiment_concepts': {'E1': [concept], 'E2': [concept], 'E3': [concept]},
                 'measurement_concepts': {}}
        sdk = {'entities': [{'entity_name': 'M1'}, {'entity_name': 'M2'}]}
        chosen, paths = select_measurements(records, graph, [concept], sdk, limit=2)
        self.assertEqual([item['id'] for item in chosen], ['M1', 'M3'])
        self.assertEqual([item['doc_id'] for item in paths], ['D1', 'D2'])

    def test_retrieved_paragraph_expands_only_through_shared_concept(self):
        concept = 'concept:process:anneal'
        records = {'measurements': [
            {'id': 'M3', 'experiment_id': 'E3', 'doc_id': 'D3', 'source_ids': ['p3']},
            {'id': 'M2', 'experiment_id': 'E2', 'doc_id': 'D2', 'source_ids': ['p2']},
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['p1']},
        ]}
        graph = {'entities': [{'entity_name': concept, 'description': 'Anneal'}],
                 'experiment_concepts': {'E1': [concept], 'E2': [concept]},
                 'measurement_concepts': {}}
        chosen, paths = select_measurements(records, graph, [], {}, retrieved_ids=['p1'])
        self.assertEqual([item['id'] for item in chosen], ['M1', 'M2'])
        self.assertEqual([item['measurement_id'] for item in paths], ['M1', 'M2'])

    def test_experiment_source_and_purpose_paragraphs_select_linked_measurements(self):
        records = {'experiments': [
            {'id': 'E1', 'source_ids': ['e1'], 'purpose_evidence': []},
            {'id': 'E2', 'source_ids': ['e2'], 'purpose_evidence': []},
            {'id': 'E3', 'source_ids': ['e3'], 'purpose_evidence': [{'paragraph_id': 'purpose3'}]},
        ], 'measurements': [
            {'id': 'M1', 'experiment_id': 'E1', 'doc_id': 'D1', 'source_ids': ['m1']},
            {'id': 'M2', 'experiment_id': 'E2', 'doc_id': 'D2', 'source_ids': ['m2']},
            {'id': 'M3', 'experiment_id': 'E3', 'doc_id': 'D3', 'source_ids': ['m3']},
        ]}
        graph = {'entities': [], 'experiment_concepts': {}, 'measurement_concepts': {}}
        sdk = {'chunks': [{'content': '[e1] Experiment description'},
                          {'content': '[purpose3] Experiment purpose'}]}
        chosen, paths = select_measurements(records, graph, [], sdk)
        self.assertEqual([item['id'] for item in chosen], ['M1', 'M3'])
        self.assertEqual(paths, [])

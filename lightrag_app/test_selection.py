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

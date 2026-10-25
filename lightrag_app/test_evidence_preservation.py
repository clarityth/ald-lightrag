import unittest
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from lightrag_app import backend

class EvidencePreservationTest(unittest.TestCase):
    def setUp(self):
        self.dataset={'documents':[{'id':'D','title':'Paper','url':'https://example.org/paper','paragraphs':[
            {'id':'p1','section':'Methods','text':'Sample A was grown at 220 C.'},
            {'id':'p2','section':'Results','text':'Sample A had a dielectric constant of 12.'},
            {'id':'p3','section':'Results','text':'A separate study reports a result.'}]}]}
        self.experiment={'id':'E','doc_id':'D','sample':'Sample A','varied_parameter':'temperature', 'fixed_conditions':'220 C', 'purpose':'unknown','purpose_evidence':[], 'source_ids':['p1']}
        self.measurement={'id':'M','experiment_id':'E','doc_id':'D','metric':'dielectric constant','value':12,'unit':'1','conditions':'220 C','source_ids':['p2'],'evidence':[{'paragraph_id':'p2','quote':'Sample A had a dielectric constant of 12.'}]}
        self.records={'experiments':[self.experiment],'measurements':[self.measurement]}

    def test_raw_search_survives_missing_concepts(self):
        selected,chunks=backend.graph_context(self.dataset,self.records,[],retrieved_ids=['p3'])
        self.assertEqual([c['id'] for c in chunks],['p3'])

    def test_citations_are_unique_original_paragraphs_with_separate_conditions(self):
        selected,chunks=backend.graph_context(self.dataset,self.records,[self.measurement],retrieved_ids=['p2'])
        self.assertEqual([c['id'] for c in chunks],['p2','p1'])
        self.assertEqual(chunks[0]['measurement_ids'],['M'])
        self.assertNotIn('context_text',chunks[0])
        self.assertNotIn('220 C',backend._chunk_context(1,chunks[0]))
        self.assertIn('220 C',backend._chunk_context(2,chunks[1]))

    def test_budget_keeps_raw_and_drops_whole_expansion(self):
        raw=backend.raw_chunks(self.dataset,{'p3'})
        limit=len(backend.tiktoken.encoding_for_model('gpt-4o-mini').encode(backend._context(raw)))
        selected,chunks=backend.graph_context(self.dataset,self.records,[self.measurement],limit=limit,retrieved_ids=['p3'])
        self.assertEqual([c['id'] for c in chunks],['p3'])
        self.assertEqual(selected,[])
        with self.assertRaisesRegex(RuntimeError,'검색 원문'):
            backend.graph_context(self.dataset,self.records,[],limit=1,retrieved_ids=['p3'])

    def test_prepare_keeps_sdk_paragraph_when_no_concept_is_recognized(self):
        dataset=backend._dataset()
        pid=dataset['documents'][0]['paragraphs'][0]['id']
        rag=Mock(initialize_storages=AsyncMock(),finalize_storages=AsyncMock(),
                 aquery_data=AsyncMock(return_value={'status':'success','data':{'entities':[], 'relationships':[], 'chunks':[{'content':f'[{pid}] raw source'}]}}))
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'index.json'
            marker.write_text(json.dumps({'dataset_sha256':hashlib.sha256(backend.DATASET.read_bytes()).hexdigest(),'graph_signature':backend.index_signature()}))
            with patch.object(backend,'MARKER',marker),patch.object(backend,'_rag',return_value=rag),patch.object(backend,'match_query',return_value=[]):
                result=asyncio.run(backend.prepare('an unclassified user question'))
        self.assertEqual(result['chunks'][0]['id'],pid)
        self.assertEqual(result['metadata']['retrieved_paragraph_ids'],[pid])

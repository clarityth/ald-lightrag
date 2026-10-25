"""Model transport configured through environment variables."""
import json, os, time
import requests

MODEL = 'gpt-6-luna'
EMBED_MODEL = 'text-embedding-3-small'

def post(endpoint, payload):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('API 인증 설정이 없습니다.')
    start = time.perf_counter()
    r = requests.post('https://api.openai.com/v1/' + endpoint,
        headers={'Authorization': 'Bearer ' + key}, json=payload, timeout=(10, 120))
    if r.status_code != 200:
        raise RuntimeError(f'모델 API 응답 오류: HTTP {r.status_code}')
    return r.json(), time.perf_counter() - start

def generate(system, content, max_tokens=1800, schema=None, *, model=None):
    payload = {'model': model or MODEL, 'store': False,
        'input': [{'role':'system','content':system},{'role':'user','content':content}],
        'reasoning': {'effort':'low'}, 'max_output_tokens':max_tokens}
    if schema: payload['text']={'format':{'type':'json_schema','name':'result','strict':True,'schema':schema}}
    data, seconds = post('responses',payload)
    if data.get('status') != 'completed':
        raise RuntimeError('모델 응답이 완료되지 않았습니다.')
    text = ''.join(c.get('text','') for o in data.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
    if not text: raise RuntimeError('모델 응답이 비어 있습니다.')
    return text, {'seconds':seconds,'usage':data.get('usage'),'model':data.get('model')}

def embed(texts):
    data, seconds = post('embeddings', {'model':EMBED_MODEL,'input':texts})
    rows=sorted(data['data'],key=lambda r:r['index'])
    if len(rows)!=len(texts): raise RuntimeError('임베딩 개수가 일치하지 않습니다.')
    return [r['embedding'] for r in rows], {'seconds':seconds,'usage':data.get('usage'),'model':EMBED_MODEL}

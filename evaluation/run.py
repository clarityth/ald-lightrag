"""Run each frozen question once per method; publish answers, evidence IDs and metrics."""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from lightrag_app import backend

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'reports/comparison.json'
PRICES = {'input':2.0,'cached_input':0.2,'output':10.0,'embedding':0.02}


def usage_metrics(calls):
    inputs = outputs = cached = embeddings = 0
    for call in calls:
        usage = call.get('usage') or {}
        if call['kind'] == 'embedding':
            embeddings += usage.get('total_tokens', usage.get('prompt_tokens',0))
        elif call['kind'] == 'answer_llm':
            inputs += usage.get('input_tokens',0)
            outputs += usage.get('output_tokens',0)
            cached += usage.get('input_tokens_details',{}).get('cached_tokens',0)
        else:
            raise RuntimeError('Unexpected extra generation call')
    cost=((inputs-cached)*PRICES['input']+cached*PRICES['cached_input']+outputs*PRICES['output']+embeddings*PRICES['embedding'])/1e6
    return {'input_tokens':inputs,'cached_input_tokens':cached,'output_tokens':outputs,'embedding_tokens':embeddings,'estimated_cost_usd':cost}


async def run(question_file=ROOT/"evaluation/questions.json", report=REPORT, modes=("all","naive","graph")):
    questions=json.loads(question_file.read_text())
    assert questions and len({q["id"] for q in questions}) == len(questions)
    assert set(modes) <= {"all", "naive", "graph"}
    existing=json.loads(report.read_text()) if report.exists() else []
    done={(r['question_id'],r['mode']) for r in existing}
    assert len(existing)==len(done)
    for q in questions:
        # Rotate execution order to avoid always giving one method warm caches.
        offset=(int(q['id'][1:])-1)%len(modes)
        for mode in modes[offset:]+modes[:offset]:
            if (q['id'],mode) in done: continue
            start=time.perf_counter()
            row={'question_id':q['id'],'question':q['question'],'kind':q['kind'],'mode':mode}
            try:
                result=await backend._ask(q['question'],mode)
                paragraphs={c.get('original_paragraph_id',c['id'].split('#')[0]) for c in result['chunks']}
                found=[m['measurement_id'] for m in q['required_measurements'] if set(m['paragraph_ids'])<=paragraphs]
                row.update(answer=result['answer'],required_count=len(q['required_measurements']),recovered_measurements=found,
                    evidence_recall=len(found)/len(q['required_measurements']),
                    context_tokens=result['metadata']['context_tokens'],seconds=result['metadata']['elapsed_seconds'],
                    citations=[{'number':i,'paragraph_id':c.get('original_paragraph_id',c['id'].split('#')[0]),'measurement_ids':c.get('measurement_ids',[])} for i,c in enumerate(result['chunks'],1)],
                    paths=result['paths'],retrieved_paragraph_ids=result['metadata']['retrieved_paragraph_ids'],context_paragraph_ids=sorted(paragraphs),**usage_metrics(result['metadata']['model_calls']))
                if mode=='graph' and ((question_file.name == 'questions.json' and q['id']=='Q08') or (question_file.name != 'questions.json' and q['id']==questions[0]['id'])):
                    public={k:result[k] for k in ('question','answer','chunks','documents','experiments','measurements','paths')}
                    (ROOT/'lightrag_app/demo/example.json').write_text(json.dumps([public],ensure_ascii=False,indent=2)+'\n')
            except Exception as error:
                row.update(error=type(error).__name__,seconds=time.perf_counter()-start,required_count=len(q['required_measurements']),recovered_measurements=[],evidence_recall=0)
            existing.append(row);done.add((q['id'],mode))
            report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps(existing,ensure_ascii=False,indent=2)+'\n')
            print(q['id'],mode,'ERROR' if 'error' in row else f"recall={row['evidence_recall']:.2f} tokens={row['context_tokens']} seconds={row['seconds']:.1f}",flush=True)
    assert len(existing)==len(questions)*len(modes)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--questions',type=Path,default=ROOT/'evaluation/questions.json')
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--modes',nargs='+',choices=['all','naive','graph'],default=['all','naive','graph'])
    args=parser.parse_args()
    assert len(args.modes)==len(set(args.modes))
    asyncio.run(run(args.questions,args.report,args.modes))

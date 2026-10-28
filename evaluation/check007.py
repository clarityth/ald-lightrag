"""Reproduce the frozen held-out evidence and answer audit totals without API calls."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
REPORT=ROOT/'reports/evidence-preservation'

def summarize(implementation_ref=None):
    questions={q['id']:q for q in json.loads((ROOT/'evaluation/questions-007.json').read_text())}
    rows=json.loads((REPORT/'comparison.json').read_text())
    audits=[r for f in sorted(REPORT.glob('audit-*.json')) for r in json.loads(f.read_text())]
    frozen=json.loads((REPORT/'frozen-inputs.json').read_text())
    for name, digest in frozen['files'].items():
        # Historical code is checked at its evaluated revision, not relabeled as current performance.
        content = subprocess.check_output(['git', 'show', f'{implementation_ref}:{name}'], cwd=ROOT) \
            if implementation_ref and name.endswith('.py') else (ROOT/name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest, name
    expected={(qid,mode) for qid in questions for mode in ('naive','graph')}
    keyed={(r['question_id'],r['mode']):r for r in rows}
    reviewed={(r['question_id'],r['mode']):r for r in audits}
    assert len(rows)==len(audits)==24 and set(keyed)==set(reviewed)==expected
    for key,row in keyed.items():
        gold={m['measurement_id'] for m in questions[key[0]]['required_measurements']}
        audit=reviewed[key]
        assert set(audit['answered_required_measurements'])<=gold
        assert set(row['recovered_measurements'])<=gold
        source={c['paragraph_id'] for c in row.get('citations',[])}
        assert all(set(c['paragraph_ids'])<=source for c in audit['claims'])
        assert set(row.get('retrieved_paragraph_ids',[]))<=set(row.get('context_paragraph_ids',[]))
    result={}
    for mode in ('naive','graph'):
        selected=[r for r in rows if r['mode']==mode]
        good=[r for r in selected if 'error' not in r]
        review=[r for r in audits if r['mode']==mode]
        counts=Counter(c['status'] for r in review for c in r['claims'])
        assert set(counts)<={'supported','unsupported','unclear'}
        result[mode]={
            'successful_runs':len(good),'failed_runs':len(selected)-len(good),
            'required':sum(len(q['required_measurements']) for q in questions.values()),
            'recovered':sum(len(r['recovered_measurements']) for r in selected),
            'answered':sum(len(r['answered_required_measurements']) for r in review),
            'numeric_claims':dict(counts),
            'context_mean_success':sum(r['context_tokens'] for r in good)/len(good),
            'seconds_mean_success':sum(r['seconds'] for r in good)/len(good),
            'cost_recorded':sum(r.get('estimated_cost_usd',0) for r in selected),
            'retrieved_paragraphs':sum(len(r['retrieved_paragraph_ids']) for r in good),
            'preserved_paragraphs':sum(len(set(r['retrieved_paragraph_ids'])&set(r['context_paragraph_ids'])) for r in good),
        }
    return result

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation-ref')
    args = parser.parse_args()
    print(json.dumps(summarize(args.implementation_ref),ensure_ascii=False,indent=2))

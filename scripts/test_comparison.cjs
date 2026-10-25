const assert = require('node:assert/strict');
const test = require('node:test');
const { buildComparisonRows, comparisonCsv } = require('../lightrag_app/comparison.js');

const chunk = (id, doc_id, text) => ({ id, doc_id, title: `Title ${doc_id}`, url: `https://example.org/${doc_id}`, section: 'Results', text });

test('every measurement becomes one row with document-scoped experiments and distinct source paragraphs', () => {
  const result = {
    documents: [{ id: 'A', title: 'Paper A', url: 'https://example.org/A' }, { id: 'B', title: 'Paper B', url: 'https://example.org/B' }],
    experiments: [
      { id: 'E1', doc_id: 'A', sample: 'A sample', varied_parameter: 'temperature', fixed_conditions: 'pressure', evidence: [{ paragraph_id: 'p1', quote: 'Condition A' }] },
      { id: 'E1', doc_id: 'B', sample: 'B sample', evidence: [{ paragraph_id: 'p1', quote: 'Condition B' }] },
    ],
    measurements: [
      { id: 'M1', doc_id: 'A', experiment_id: 'E1', metric: 'thickness', value: 3, unit: 'nm', conditions: 'room temperature', evidence: [{ paragraph_id: 'p2', quote: 'Measured A' }, { paragraph_id: 'p2', quote: 'Measured A' }] },
      { id: 'M1', doc_id: 'B', experiment_id: 'E1', metric: 'density', value: 4, unit: 'g/cm3', conditions: 'hot', evidence: [{ paragraph_id: 'p2', quote: 'Measured B' }] },
      { id: 'M1', doc_id: 'A', experiment_id: 'E1', metric: 'thickness', value: 3, unit: 'nm', conditions: 'room temperature' },
    ],
    chunks: [chunk('p1', 'A', 'Condition A'), chunk('p2', 'A', 'Measured A'), chunk('p1', 'B', 'Condition B'), chunk('p2', 'B', 'Measured B')],
  };
  const rows = buildComparisonRows(result);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map(({ measurementId, docId, sample }) => [measurementId, docId, sample]), [['M1', 'A', 'A sample'], ['M1', 'B', 'B sample']]);
  assert.deepEqual(rows[0].conditionSources.map(({ doc_id, id, quote }) => [doc_id, id, quote]), [['A', 'p1', 'Condition A']]);
  assert.deepEqual(rows[0].measurementSources.map(({ doc_id, id, quote }) => [doc_id, id, quote]), [['A', 'p2', 'Measured A']]);
  assert.equal(rows[0].value, 3);
  assert.equal(rows[0].title, 'Paper A');
});

test('missing fields remain explicit and foreign or absent paragraphs never become source links', () => {
  const rows = buildComparisonRows({
    experiments: [{ id: 'E', doc_id: 'B', sample: 'foreign' }],
    measurements: [{ id: 'M', doc_id: 'A', experiment_id: 'E', evidence: [{ paragraph_id: 'p', quote: 'foreign text' }] }],
    chunks: [chunk('p', 'B', 'foreign text')],
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].sample, '정보 없음');
  assert.equal(rows[0].metric, '정보 없음');
  assert.deepEqual(rows[0].conditionSources, []);
  assert.deepEqual(rows[0].measurementSources, []);
});

test('saved example exposes all measurements and their original condition and measurement sources', () => {
  const result = require('../lightrag_app/demo/example.json')[0];
  const rows = buildComparisonRows(result);
  assert.equal(rows.length, result.measurements.length);
  const row = rows.find(item => item.measurementId === 'PMC10005305:M9');
  assert.equal(row.docId, 'PMC10005305');
  assert.ok(row.conditionSources.length > 0);
  assert.ok(row.measurementSources.length > 0);
  assert.ok(row.conditionSources.every(source => source.doc_id === row.docId && source.text));
  assert.ok(row.measurementSources.every(source => source.doc_id === row.docId && source.text));
});

test('CSV includes question and both excerpts, quotes multiline cells, and neutralizes formulas', () => {
  const rows = [{ title: 'A "paper"', url: 'https://example.org/A', sample: '=RUN()', variedParameter: '+SUM(1,2)', fixedConditions: 'line 1\nline 2', metric: 'thickness', value: 3, unit: 'nm', conditions: 'room temp', conditionSources: [chunk('p1', 'A', 'Condition, "quoted"')], measurementSources: [{ ...chunk('p2', 'A', 'Measured\nvalue'), quote: 'Measured\nvalue' }] }];
  const csv = comparisonCsv(rows, '@question');
  assert.match(csv, /^\uFEFF?질문,/);
  assert.ok(csv.includes("'@question"));
  assert.ok(csv.includes("'=RUN()"));
  assert.ok(csv.includes("'+SUM(1,2)"));
  assert.ok(csv.includes('"line 1\nline 2"'));
  assert.ok(csv.includes('"Condition, ""quoted"""'));
  assert.ok(csv.includes('"Measured\nvalue"'));
  assert.ok(csv.includes('https://example.org/A'));
});

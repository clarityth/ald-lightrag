const assert = require('node:assert/strict');
const test = require('node:test');
const { load, save, clear, rowId } = require('../lightrag_app/workspace.js');

function memoryStorage() {
  const data = new Map();
  return {
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => data.set(key, value),
    removeItem: key => data.delete(key),
  };
}

const answer = question => ({ question, answer: '답변', chunks: [], experiments: [], measurements: [] });

test('the saved answer and its selected row IDs survive reload', () => {
  const storage = memoryStorage();
  const first = answer('첫 질문');
  const second = answer('다른 질문');
  const selected = rowId({ docId: 'paper', measurementId: 'M1' }, 0);
  assert.equal(save(storage, first, []), true);
  assert.equal(save(storage, second, [selected]), true);
  assert.deepEqual(load(storage), { value: { result: second, selectedIds: [selected] }, error: false });
  assert.equal(save(storage, { question: '실패한 질문' }, []), false);
  assert.deepEqual(load(storage).value, { result: second, selectedIds: [selected] });
  assert.equal(save(storage, first, []), true);
  assert.deepEqual(load(storage).value, { result: first, selectedIds: [] });
  assert.equal(clear(storage), true);
  assert.deepEqual(load(storage), { value: null, error: false });
});

test('row IDs distinguish documents and rows without a measurement ID', () => {
  assert.notEqual(rowId({ docId: 'A', measurementId: 'M' }, 0), rowId({ docId: 'B', measurementId: 'M' }, 0));
  assert.notEqual(rowId({ docId: 'A' }, 0), rowId({ docId: 'A' }, 1));
  assert.notEqual(rowId({ docId: 'A', measurementId: 1 }, 0), rowId({ docId: 'A' }, 1));
});

test('damaged data is rejected and storage failures do not throw', () => {
  const storage = memoryStorage();
  storage.setItem('processlens.workspace.v1', '{broken');
  assert.deepEqual(load(storage), { value: null, error: true });
  storage.setItem('processlens.workspace.v1', JSON.stringify({ version: 1, result: answer('질문'), selectedIds: [1] }));
  assert.deepEqual(load(storage), { value: null, error: true });
  const denied = { getItem() { throw Error('denied'); }, setItem() { throw Error('denied'); }, removeItem() { throw Error('denied'); } };
  assert.deepEqual(load(denied), { value: null, error: true });
  assert.equal(save(denied, answer('질문'), []), false);
  assert.equal(clear(denied), false);
});

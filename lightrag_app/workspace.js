(function () {
  const key = 'processlens.workspace.v1';
  const rowId = (row, index) => JSON.stringify([row.docId ?? null, row.measurementId == null ? 'row' : 'measurement', row.measurementId ?? index]);

  function valid(value) {
    if (!value || value.version !== 1 || !value.result || typeof value.result !== 'object') return false;
    const result = value.result;
    return typeof result.question === 'string' && typeof result.answer === 'string'
      && ['chunks', 'documents', 'experiments', 'measurements', 'paths'].every(name =>
        result[name] === undefined || Array.isArray(result[name]) && result[name].every(row => row && typeof row === 'object'))
      && Array.isArray(value.selectedIds) && value.selectedIds.every(id => typeof id === 'string');
  }

  function load(storage) {
    try {
      const raw = storage.getItem(key);
      if (raw === null) return { value: null, error: false };
      const data = JSON.parse(raw);
      return valid(data) ? { value: { result: data.result, selectedIds: data.selectedIds }, error: false }
        : { value: null, error: true };
    } catch { return { value: null, error: true }; }
  }

  function save(storage, result, selectedIds) {
    try {
      const data = { version: 1, result, selectedIds };
      if (!valid(data)) return false;
      storage.setItem(key, JSON.stringify(data));
      return true;
    } catch { return false; }
  }

  function clear(storage) {
    try { storage.removeItem(key); return true; }
    catch { return false; }
  }

  const api = { load, save, clear, rowId };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.ProcessLensWorkspace = api;
})();

(function () {
  const unknown = '정보 없음';
  const shown = value => value === undefined || value === null || value === '' ? unknown : value;
  const key = (docId, id) => `${docId}\u0000${id}`;

  function buildComparisonRows(result) {
    const chunks = new Map((result.chunks || []).map(chunk => [key(chunk.doc_id, chunk.id), chunk]));
    const documents = new Map((result.documents || []).map(doc => [doc.id, doc]));
    const experiments = new Map((result.experiments || []).map(experiment => [key(experiment.doc_id, experiment.id), experiment]));
    const seen = new Set();

    function sources(record) {
      if (!record) return [];
      const evidence = record.evidence?.length ? record.evidence : (record.source_ids || []).map(paragraph_id => ({ paragraph_id }));
      const used = new Set();
      return evidence.flatMap(item => {
        const sourceKey = key(record.doc_id, item.paragraph_id);
        const chunk = chunks.get(sourceKey);
        if (!chunk || used.has(sourceKey)) return [];
        used.add(sourceKey);
        return [{ ...chunk, ...(item.quote && chunk.text?.includes(item.quote) ? { quote: item.quote } : {}) }];
      });
    }

    return (result.measurements || []).flatMap((measurement, index) => {
      const measurementKey = key(measurement.doc_id, measurement.id ?? index);
      if (seen.has(measurementKey)) return [];
      seen.add(measurementKey);
      const experiment = experiments.get(key(measurement.doc_id, measurement.experiment_id));
      const doc = documents.get(measurement.doc_id) || [...chunks.values()].find(chunk => chunk.doc_id === measurement.doc_id) || {};
      return [{
        measurementId: measurement.id,
        experimentId: experiment?.id || measurement.experiment_id,
        docId: measurement.doc_id,
        title: shown(doc.title),
        url: shown(doc.url),
        sample: shown(experiment?.sample),
        variedParameter: shown(experiment?.varied_parameter),
        fixedConditions: shown(experiment?.fixed_conditions),
        metric: shown(measurement.metric),
        value: shown(measurement.value),
        unit: shown(measurement.unit),
        conditions: shown(measurement.conditions),
        conditionSources: sources(experiment),
        measurementSources: sources(measurement),
      }];
    });
  }

  function comparisonCsv(rows, question) {
    const headers = ['질문', '문헌', '문헌 URL', '문헌 ID', '실험 ID', '측정 ID', '시료', '변경 조건', '고정 공정 조건', '측정 지표', '값', '단위', '측정 조건', '조건 원문 발췌', '측정 원문 발췌'];
    const excerpts = sources => (sources || []).map(source => source.quote || source.text || '').filter(Boolean).join('\n\n');
    const cell = value => {
      let text = String(value ?? '');
      if (/^[\s\uFEFF]*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text)) text = `'${text}`;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const body = (rows || []).map(row => [question, row.title, row.url, row.docId, row.experimentId, row.measurementId, row.sample, row.variedParameter, row.fixedConditions, row.metric, row.value, row.unit, row.conditions, excerpts(row.conditionSources), excerpts(row.measurementSources)].map(cell).join(','));
    return '\uFEFF' + [headers.join(','), ...body].join('\r\n');
  }

  const api = { buildComparisonRows, comparisonCsv };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.ProcessLensComparison = api;
})();

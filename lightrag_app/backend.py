"""Local LightRAG index and source-grounded question answering."""

import asyncio
import contextvars
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import tiktoken
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

import provider
from lightrag_app.experiments import VERSION as EXTRACTION_VERSION, extract_all


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "mechanism_poc/sources/dataset.json"
STORE = ROOT / "storage-experiments-v3"
MARKER = STORE / "index.json"
RECORDS = STORE / "experiments.json"
ANSWER_MODEL = "gpt-6-sol"
CONTEXT_LIMIT = 128000
CHUNK_ID = re.compile(r"^\[([^\]]+)\] ")
CALLS = contextvars.ContextVar("lightrag_calls", default=None)
PLAN_PROMPT = (
    "Decompose the literature question into at most three concise, self-contained English "
    "search queries. For comparisons, give each requested comparison arm its own query, "
    "preserving the material, sample, process stage, and requested measurements. "
    "Do not answer, infer evidence, or mix conditions across arms. Return JSON with queries."
)


def _dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _context(chunks):
    return "\n\n".join(
        _chunk_context(i, chunk) for i, chunk in enumerate(chunks, 1)
    )


def _chunk_context(number, chunk):
    if "context_text" in chunk:
        return f"[{number}] {chunk['id']}\n{chunk['context_text']}"
    return f"[{number}] {chunk['title']} ({chunk['id']}, {chunk['section']})\n{chunk['text']}"


async def _search_queries(question, calls):
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["queries"],
        "additionalProperties": False,
    }
    response, info = await asyncio.to_thread(
        provider.generate, PLAN_PROMPT,
        question,
        700, schema,
    )
    calls.append({"kind": "query_plan_llm", **info})
    planned = json.loads(response)
    queries = [question]
    for query in planned["queries"]:
        query = query.strip()
        if query and query not in queries:
            queries.append(query)
    return queries


async def _embed(texts: list[str]) -> np.ndarray:
    vectors, info = await asyncio.to_thread(provider.embed, texts)
    if CALLS.get() is not None:
        CALLS.get().append({"kind": "embedding", **info})
    return np.asarray(vectors)


async def _complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    history = "\n".join(
        f"{message['role']}: {message['content']}" for message in (history_messages or [])
    )
    content = f"{history}\n{prompt}" if history else prompt
    answer, info = await asyncio.to_thread(
        provider.generate, system_prompt or "", content, 1800
    )
    if CALLS.get() is not None:
        CALLS.get().append({"kind": "retrieval_llm", **info})
    return answer


def _rag():
    return LightRAG(
        working_dir=str(STORE),
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8191,
            model_name=provider.EMBED_MODEL,
            func=_embed,
        ),
        llm_model_func=_complete,
        llm_model_name=provider.MODEL,
    )


def _payload(dataset, records=None):
    chunks = []
    paragraphs = {}
    for doc in dataset["documents"]:
        for paragraph in doc["paragraphs"]:
            pid = paragraph["id"]
            if pid in paragraphs:
                raise ValueError(f"중복 원문 ID: {pid}")
            paragraphs[pid] = (doc, paragraph)
            chunks.append({
                "content": f"[{pid}] {doc['title']} | {paragraph['section']}\n{paragraph['text']}",
                "source_id": pid,
                "file_path": doc["id"],
            })

    entities = {}
    relationships = []
    for relation in dataset["relations"]:
        evidence = relation["evidence"]
        if not evidence:
            raise ValueError(f"원문이 없는 관계: {relation['id']}")
        pid = evidence[0]["paragraph_id"]
        if pid not in paragraphs or paragraphs[pid][0]["id"] != relation["doc_id"]:
            raise ValueError(f"관계의 문헌과 원문이 다릅니다: {relation['id']}")
        if evidence[0]["quote"] not in paragraphs[pid][1]["text"]:
            raise ValueError(f"관계 인용이 원문에 없습니다: {relation['id']}")

        observation = f"{relation['doc_id']} {relation['id']} observation"
        context = (
            f"{relation['doc_id']} {relation['id']}: {relation['subject']} "
            f"{relation['predicate'].replace('_', ' ')} {relation['object']}. "
            f"Sample and measurement context: {relation['sample_context']}. "
            f"Assertion type: {relation['assertion_type']}. Evidence: {pid}."
        )
        entities[observation] = {
            "entity_name": observation,
            "entity_type": "paper_observation",
            "description": context,
            "source_id": pid,
            "file_path": relation["doc_id"],
        }
        for concept in (relation["subject"], relation["object"]):
            entities.setdefault(concept, {
                "entity_name": concept,
                "entity_type": "concept",
                "description": concept,
                "source_id": pid,
                "file_path": relation["doc_id"],
            })
        for src, dst in ((relation["subject"], observation), (observation, relation["object"])):
            relationships.append({
                "src_id": src,
                "tgt_id": dst,
                "description": context,
                "keywords": f"{relation['subject']}, {relation['object']}, {relation['predicate']}",
                "source_id": pid,
                "file_path": relation["doc_id"],
            })
    if records:
        for experiment in records["experiments"]:
            source_id = experiment["source_ids"][0]
            if source_id not in paragraphs or paragraphs[source_id][0]["id"] != experiment["doc_id"]:
                raise ValueError(f"실험군 근거가 원문에 없습니다: {experiment['id']}")
            entities[experiment["id"]] = {
                "entity_name": experiment["id"], "entity_type": "experimental_group",
                "description": (
                    f"Paper {experiment['doc_id']}; sample {experiment['sample']}; "
                    f"varied parameter {experiment['varied_parameter']}; "
                    f"fixed conditions {experiment['fixed_conditions']}; "
                    f"author-stated purpose {experiment['purpose']}; "
                    f"purpose evidence paragraphs {', '.join(item['paragraph_id'] for item in experiment['purpose_evidence']) or 'unknown'}."
                ),
                "source_id": source_id, "file_path": experiment["doc_id"],
            }
        for measurement in records["measurements"]:
            source_id = measurement["source_ids"][0]
            experiment = entities.get(measurement["experiment_id"])
            if not experiment or source_id not in paragraphs or paragraphs[source_id][0]["id"] != measurement["doc_id"]:
                raise ValueError(f"측정의 실험군 또는 원문 근거가 없습니다: {measurement['id']}")
            description = (
                f"Paper {measurement['doc_id']}; measurement {measurement['metric']} "
                f"= {measurement['value']} {measurement['unit']}; measurement conditions "
                f"{measurement['conditions']}; linked experimental group: {experiment['description']} "
                f"Evidence paragraphs: {', '.join(measurement['source_ids'])}."
            )
            entities[measurement["id"]] = {
                "entity_name": measurement["id"], "entity_type": "measurement_observation",
                "description": description, "source_id": source_id,
                "file_path": measurement["doc_id"],
            }
            relationships.append({
                "src_id": measurement["experiment_id"], "tgt_id": measurement["id"],
                "description": description, "keywords": measurement["metric"],
                "source_id": source_id, "file_path": measurement["doc_id"],
            })
    return {"chunks": chunks, "entities": list(entities.values()), "relationships": relationships}


async def _index():
    dataset_bytes = DATASET.read_bytes()
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    if MARKER.exists() and RECORDS.exists() and (marker := json.loads(MARKER.read_text())).get("dataset_sha256") == digest and marker.get("extraction_version") == EXTRACTION_VERSION:
        return
    dataset = json.loads(dataset_bytes)
    records = extract_all(dataset["documents"])
    failures = [call for call in records["calls"] if "error" in call]
    if failures or not records["experiments"] or not records["measurements"]:
        raise RuntimeError(f"실험군 추출이 완료되지 않았습니다: {len(failures)}편 실패, {len(records['experiments'])}개 실험군, {len(records['measurements'])}개 측정")
    payload = _payload(dataset, records)
    STORE.mkdir(exist_ok=True)
    rag = _rag()
    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(payload)
    finally:
        await rag.finalize_storages()
    RECORDS.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKER.write_text(json.dumps({
        "dataset_sha256": digest,
        "extraction_version": EXTRACTION_VERSION,
        "documents": len(dataset["documents"]),
        "chunks": len(payload["chunks"]),
        "relations": len(dataset["relations"]),
        "experiments": len(records["experiments"]),
        "measurements": len(records["measurements"]),
        "rejected": records["rejected"],
        "purpose_rejected": records["purpose_rejected"],
        "extraction_calls": records["calls"],
    }, indent=2) + "\n", encoding="utf-8")


def index():
    """Build or reuse the local index from the frozen source dataset."""
    asyncio.run(_index())


def _measurement_chunks(measurements, experiments, paragraphs):
    """Give each measured outcome a distinct citation while retaining the source text."""
    by_id = {experiment["id"]: experiment for experiment in experiments}
    order = {experiment["id"]: index for index, experiment in enumerate(experiments)}
    measurements = sorted(measurements, key=lambda item: order.get(item["experiment_id"], len(order)))
    chunks = []
    described_groups = set()
    for measurement in measurements:
        experiment = by_id.get(measurement["experiment_id"])
        if not experiment:
            continue
        if experiment["id"] not in described_groups:
            group_description = (
                f"실험군 {experiment['id']}: 시료 {experiment['sample']}; "
                f"변경 조건 {experiment['varied_parameter']}; 고정 조건 {experiment['fixed_conditions']}; "
                f"저자가 명시한 시험 목적 {experiment['purpose']}; "
                f"목적 근거 {' | '.join(item['paragraph_id'] + ': ' + item['quote'] for item in experiment['purpose_evidence']) or 'unknown'}.\n"
            )
            described_groups.add(experiment["id"])
        else:
            group_description = ""
        paragraph_id = measurement["source_ids"][0]
        doc, paragraph = paragraphs[paragraph_id]
        quotes = " | ".join(
            f"{item['paragraph_id']}: {item['quote']}" for item in measurement["evidence"]
        )
        chunks.append({
            "id": f"{paragraph_id}#{measurement['id']}",
            "original_paragraph_id": paragraph_id,
            "measurement_ids": [measurement["id"]],
            "doc_id": doc["id"], "title": doc["title"], "url": doc["url"],
            "section": paragraph["section"], "text": paragraph["text"],
            "context_text": (
                f"{group_description}측정 {measurement['metric']} = {measurement['value']} {measurement['unit']}; "
                f"측정 조건 {measurement['conditions']}; 실험군 {experiment['id']}; "
                f"원문 인용 {quotes}"
            ),
        })
    return chunks


def _fit_context(raw_chunks, fact_chunks, retrieved_ids, selected_doc_ids, limit=CONTEXT_LIMIT):
    """Keep retrieved evidence first, then linked facts, then full-paper expansion."""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    by_id = {chunk["id"]: chunk for chunk in raw_chunks}
    retrieved = [by_id[pid] for pid in dict.fromkeys(retrieved_ids) if pid in by_id]
    retrieved_set = {chunk["id"] for chunk in retrieved}
    if len(encoding.encode(_context(retrieved))) > limit:
        raise RuntimeError("LightRAG가 회수한 원문 청크가 답변 문맥 상한을 초과했습니다.")
    omitted_measurements = []
    fact_chunks = list(fact_chunks)
    while fact_chunks and len(encoding.encode(_context(fact_chunks + retrieved))) > limit:
        omitted_measurements.extend(fact_chunks.pop()["measurement_ids"])
    extras = [chunk for chunk in raw_chunks if chunk["id"] not in retrieved_set]
    candidate = fact_chunks + retrieved + extras
    if len(encoding.encode(_context(candidate))) <= limit:
        return candidate, [], omitted_measurements, len(encoding.encode(_context(candidate)))
    by_doc = {doc_id: [chunk for chunk in extras if chunk["doc_id"] == doc_id] for doc_id in selected_doc_ids}
    limited = list(retrieved)
    omitted_paragraphs = []
    for rank in range(max(map(len, by_doc.values()), default=0)):
        for doc_id in selected_doc_ids:
            if rank >= len(by_doc[doc_id]):
                continue
            chunk = by_doc[doc_id][rank]
            if len(encoding.encode(_context(fact_chunks + limited + [chunk]))) <= limit:
                limited.append(chunk)
            else:
                omitted_paragraphs.append(chunk["id"])
    final = fact_chunks + limited
    return final, omitted_paragraphs, omitted_measurements, len(encoding.encode(_context(final)))


async def _ask(question: str):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("질문을 입력해 주세요.")
    if not MARKER.exists():
        raise RuntimeError("LightRAG 색인이 없습니다. 먼저 index()를 실행하세요.")
    if not RECORDS.exists():
        raise RuntimeError("실험군 연결 기록이 없습니다. 먼저 index()를 실행하세요.")
    marker = json.loads(MARKER.read_text())
    if marker["dataset_sha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest() or marker.get("extraction_version") != EXTRACTION_VERSION:
        raise RuntimeError("원문 데이터가 바뀌었습니다. 색인을 다시 생성하세요.")

    started = time.perf_counter()
    calls = []
    dataset = _dataset()
    records = json.loads(RECORDS.read_text(encoding="utf-8"))
    paragraphs = {
        p["id"]: (d, p) for d in dataset["documents"] for p in d["paragraphs"]
    }
    queries = await _search_queries(question.strip(), calls)
    token = CALLS.set(calls)
    try:
        rag = _rag()
        await rag.initialize_storages()
        try:
            param = QueryParam(mode="mix", top_k=20, chunk_top_k=10, enable_rerank=False)
            results = [await rag.aquery_data(query, param=param) for query in queries]
        finally:
            await rag.finalize_storages()
    finally:
        CALLS.reset(token)
    retrieval_seconds = time.perf_counter() - started
    for result in results:
        if result.get("status") == "failure":
            raise RuntimeError(f"LightRAG 검색 실패: {result.get('message', '원인 미상')}")

    groups = [result.get("data", {}).get("chunks", []) for result in results]
    retrieved_chunks = []
    seen_chunks = set()
    for rank in range(max(map(len, groups), default=0)):
        for group in groups:
            if rank >= len(group):
                continue
            retrieved = group[rank]
            key = retrieved.get("chunk_id") or retrieved.get("content")
            if key not in seen_chunks:
                seen_chunks.add(key)
                retrieved_chunks.append(retrieved)

    chunks = []
    for retrieved in retrieved_chunks:
        match = CHUNK_ID.match(retrieved.get("content", ""))
        if not match or match.group(1) not in paragraphs:
            raise RuntimeError("LightRAG 검색 청크를 원문 ID에 연결할 수 없습니다.")
        doc, paragraph = paragraphs[match.group(1)]
        if retrieved.get("file_path") != doc["id"] or paragraph["text"] not in retrieved["content"]:
            raise RuntimeError(f"LightRAG 검색 청크의 원문 검증에 실패했습니다: {paragraph['id']}")
        chunks.append({
            "id": paragraph["id"],
            "doc_id": doc["id"],
            "title": doc["title"],
            "url": doc["url"],
            "section": paragraph["section"],
            "text": paragraph["text"],
        })

    retrieved_chunk_ids = [chunk["id"] for chunk in chunks]
    selected_doc_ids = list(dict.fromkeys(chunk["doc_id"] for chunk in chunks))
    full_chunks = []
    for doc_id in selected_doc_ids:
        doc = next(doc for doc in dataset["documents"] if doc["id"] == doc_id)
        full_chunks.extend({
            "id": paragraph["id"], "doc_id": doc["id"],
            "title": doc["title"], "url": doc["url"],
            "section": paragraph["section"], "text": paragraph["text"],
        } for paragraph in doc["paragraphs"])
    seen_raw = set()
    retrieved_source_chunks = []
    for chunk in chunks:
        if chunk["id"] not in seen_raw:
            retrieved_source_chunks.append(chunk)
            seen_raw.add(chunk["id"])
    chunks = retrieved_source_chunks + [chunk for chunk in full_chunks if chunk["id"] not in seen_raw]

    selected_experiments = [record for record in records["experiments"] if record["doc_id"] in selected_doc_ids]
    selected_measurements = [record for record in records["measurements"] if record["doc_id"] in selected_doc_ids]
    fact_chunks = _measurement_chunks(selected_measurements, selected_experiments, paragraphs)
    chunks, omitted_ids, omitted_measurement_ids, context_tokens = _fit_context(
        chunks, fact_chunks, retrieved_chunk_ids, selected_doc_ids
    )
    context = _context(chunks)
    included_measurement_ids = {mid for chunk in chunks for mid in chunk.get("measurement_ids", [])}
    selected_measurements = [m for m in selected_measurements if m["id"] in included_measurement_ids]

    documents = []
    seen = set()
    for chunk in chunks:
        if chunk["doc_id"] in seen:
            continue
        seen.add(chunk["doc_id"])
        doc = next(d for d in dataset["documents"] if d["id"] == chunk["doc_id"])
        documents.append({key: doc[key] for key in ("id", "title", "url", "license")})

    if chunks:
        answer, info = await asyncio.to_thread(
            provider.generate,
            "검색된 논문 원문과 원문 인용을 확인한 실험군·측정 연결을 근거로 한국어로 답하세요. 질문에 직접 답하는 핵심 결과와 수치를 간결하게 쓰고, 문헌별 시료와 공정·측정 조건을 구분하세요. "
            "측정 수치를 적을 때는 각 값의 바로 뒤에 해당 측정 결과 ID의 근거 청크 번호를 붙이세요. 한 문장에 값이 여러 개면 값마다 별도 번호를 붙이고, 해당 측정 청크가 있는 수치를 원문 문단 번호 하나로만 묶어 인용하지 마세요. "
            "같은 원문 문단에 여러 측정 결과가 있을 때도 반드시 해당 실험군에 연결된 측정 결과 번호를 고르세요. "
            "측정 결과 연결이 없는 정성적 결과도 원문 문단에서 확인하고 해당 번호를 인용하세요. "
            "실험군에 명시되지 않은 온도나 조건은 같은 논문의 다른 실험군에서도 가져오지 마세요. 장치 구조와 용도가 다른 실험의 결과는 별개의 연구 또는 실험으로 밝혀 주세요. "
            "MIS나 MOS 같은 시험 구조 이름만으로 응용 용도를 추정하지 말고, 저자가 명시한 용도 또는 시험 구조 자체로 서술하세요. "
            "측정 결과 연결이 없는 사실은 원문 문단 번호를 인용하세요. 불명확한 조건은 추정하지 마세요. "
            "서로 다른 시료의 수치를 합치거나 논문 사이의 인과를 단정하지 마세요. "
            "평문 단락으로 쓰고 Markdown 문법은 쓰지 마세요. 각 주장 뒤에 원문 번호 [1][2]처럼 개별 인용하세요. "
            "[1]–[9] 같은 범위 표기는 쓰지 마세요. "
            "정보가 없다고 단정하기 전에 제공된 방법과 결과 문단을 함께 확인하고, 평가 부재는 해당 조건 비교의 범위에서만 기술하세요. 질문에 필요하지 않은 미확인 항목이나 반복적인 주의 문장은 덧붙이지 마세요.",
            f"질문: {question.strip()}\n\n실험군 정의와 측정 결과의 근거 청크 및 원문 문단:\n{context}",
            1800,
            model=ANSWER_MODEL,
        )
        calls.append({"kind": "answer_llm", **info})
        references = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
        if any(number < 1 or number > len(chunks) for number in references):
            raise RuntimeError("답변의 인용 번호를 검색된 원문과 연결할 수 없습니다.")
    else:
        answer = "검색된 원문 근거가 없어 답변할 수 없습니다."
    result = {
        "question": question.strip(), "answer": answer, "chunks": chunks,
        "documents": documents,
        "experiments": selected_experiments,
        "measurements": selected_measurements,
        "metadata": {
            "mode": "mix", "retrieval_queries": queries,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": retrieved_chunks,
            "selected_fulltext_doc_ids": selected_doc_ids,
            "context_tokens": context_tokens,
            "context_limit_tokens": CONTEXT_LIMIT,
            "omitted_paragraph_ids": omitted_ids,
            "omitted_measurement_ids": omitted_measurement_ids,
            "extraction_summary": {key: marker.get(key) for key in ("experiments", "measurements", "rejected", "purpose_rejected")},
            "elapsed_seconds": time.perf_counter() - started,
            "retrieval_seconds": retrieval_seconds,
            "model_calls": calls,
        },
    }
    return result


def ask(question: str):
    """Return an answer and the exact source paragraphs retrieved by LightRAG mix mode."""
    return asyncio.run(_ask(question))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "ask"])
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()
    if args.command == "index":
        index()
    else:
        print(json.dumps(ask(args.question), ensure_ascii=False, indent=2))

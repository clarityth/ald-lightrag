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
from lightrag_app.experiments import VERSION as EXTRACTION_VERSION
from lightrag_app.concepts import build_concepts, match_query
from lightrag_app.selection import select_measurements


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "mechanism_poc/sources/dataset.json"
STORE = ROOT / "storage-graph-v1"
MARKER = STORE / "index.json"
RECORDS = STORE / "experiments.json"
ANSWER_MODEL = "gpt-6-sol"
CONTEXT_LIMIT = 128000
GRAPH_LIMIT = 16000
SOURCE_RECORDS = ROOT.parent / "data/experiments.json"
ANSWER_PROMPT = (
    "제공된 근거로 질문에 한국어로 답하세요. 문헌별 시료, 공정 조건과 측정 조건을 구분하고 핵심 결과와 수치를 간결하게 설명하세요. "
    "각 수치와 주장 바로 뒤에 그 내용을 뒷받침하는 근거 번호 [1]을 붙이세요. 여러 값은 각각 맞는 번호를 사용하세요. "
    "인용 번호는 원문 문단 하나를 가리킵니다. 결과값과 공정·측정 조건이 다른 문단에 있으면 각각 해당 문단을 인용하세요. "
    "같은 논문에서도 다른 시료의 조건을 옮기지 말고 MIS/MOS 구조만으로 응용 목적을 추정하지 마세요. "
    "서로 다른 논문의 인과를 단정하지 말고 근거가 없는 내용은 구체적으로 한정하세요. 방법과 결과를 함께 읽고 질문에 직접 답하세요. "
    "평문 단락과 개별 인용 [1][2]를 사용하고 인용 범위 표기는 사용하지 마세요."
)
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
    if records:
        concepts = build_concepts(records)
        entities.update({e["entity_name"]: e for e in concepts["entities"]})
        relationships.extend(concepts["relationships"])
    return {"chunks": chunks, "entities": list(entities.values()), "relationships": relationships}


def index_signature():
    dictionary = ROOT.parent / "config/concepts.json"
    return hashlib.sha256(DATASET.read_bytes() + SOURCE_RECORDS.read_bytes() + dictionary.read_bytes()).hexdigest()


async def _index():
    dataset_bytes = DATASET.read_bytes()
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    if MARKER.exists() and RECORDS.exists() and (marker := json.loads(MARKER.read_text())).get("dataset_sha256") == digest and marker.get("extraction_version") == EXTRACTION_VERSION and marker.get("graph_signature") == index_signature():
        return
    dataset = json.loads(dataset_bytes)
    records = json.loads(SOURCE_RECORDS.read_text())
    records.update(rejected=0, purpose_rejected=0, calls=[])
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
        "graph_signature": index_signature(),
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


def raw_chunks(dataset, ids=None):
    return [{"id":p["id"], "doc_id":d["id"], "title":d["title"], "url":d["url"],
             "section":p["section"], "text":p["text"]}
            for d in dataset["documents"] for p in d["paragraphs"] if ids is None or p["id"] in ids]


def graph_context(dataset, records, measurements, limit=GRAPH_LIMIT, retrieved_ids=()):
    """Preserve search evidence; add complete source packs with one citation per paragraph."""
    by_id = {c["id"]: c for c in raw_chunks(dataset)}
    chunks = [by_id[pid] for pid in dict.fromkeys(retrieved_ids) if pid in by_id]
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    if len(encoding.encode(_context(chunks))) > limit:
        raise RuntimeError("검색 원문이 문맥 상한을 초과했습니다.")
    experiments = {e["id"]: e for e in records["experiments"]}
    kept = []
    for measurement in measurements:
        experiment = experiments[measurement["experiment_id"]]
        ids = list(dict.fromkeys(measurement["source_ids"] + experiment["source_ids"] +
                    [item["paragraph_id"] for item in experiment.get("purpose_evidence", [])]))
        present = {c["id"] for c in chunks}
        candidate = chunks + [by_id[pid] for pid in ids if pid not in present and pid in by_id]
        if len(encoding.encode(_context(candidate))) <= limit:
            kept.append(measurement)
            chunks = candidate
    for chunk in chunks:
        chunk["measurement_ids"] = [m["id"] for m in kept if chunk["id"] in m["source_ids"]]
    return kept, chunks


async def prepare(question, mode="graph"):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("질문을 입력해 주세요.")
    if mode not in {"all", "naive", "graph", "full_documents"}:
        raise ValueError("지원하지 않는 검색 방식입니다.")
    dataset = _dataset()
    records = json.loads(SOURCE_RECORDS.read_text())
    graph = build_concepts(records)
    calls, sdk_data, retrieved_ids = [], {}, []
    concepts = match_query(question)
    if mode != "all":
        if not MARKER.exists():
            raise RuntimeError("LightRAG 색인이 없습니다. 먼저 index 명령을 실행하세요.")
        marker = json.loads(MARKER.read_text())
        if marker["dataset_sha256"] != hashlib.sha256(DATASET.read_bytes()).hexdigest() or marker.get("graph_signature") != index_signature():
            raise RuntimeError("원문 데이터가 바뀌었습니다. 색인을 다시 생성하세요.")
        labels = {e["entity_name"]:e["description"] for e in graph["entities"]}
        keywords = [labels[c] for c in concepts if c in labels] or [question]
        token = CALLS.set(calls)
        rag = _rag()
        await rag.initialize_storages()
        try:
            result = await rag.aquery_data(question, param=QueryParam(
                mode="naive" if mode == "naive" else "mix", top_k=40, chunk_top_k=10,
                hl_keywords=keywords, ll_keywords=keywords,
                max_entity_tokens=12000, max_relation_tokens=12000, max_total_tokens=30000,
                enable_rerank=False))
        finally:
            await rag.finalize_storages()
            CALLS.reset(token)
        if result.get("status") == "failure":
            raise RuntimeError(f"LightRAG 검색 실패: {result.get('message', '원인 미상')}")
        sdk_data = result.get("data", {})
        for chunk in sdk_data.get("chunks", []):
            match = CHUNK_ID.match(chunk.get("content", ""))
            if match:
                retrieved_ids.append(match.group(1))
    paths = []
    if mode == "graph":
        chosen, paths = select_measurements(records, graph, concepts, sdk_data, retrieved_ids=retrieved_ids)
        measurements, chunks = graph_context(dataset, records, chosen, retrieved_ids=retrieved_ids)
        included = {m["id"] for m in measurements}
        paths = [p for p in paths if p["measurement_id"] in included]
    elif mode == "naive":
        by_id = {c["id"]:c for c in raw_chunks(dataset)}
        chunks = [by_id[pid] for pid in dict.fromkeys(retrieved_ids[:10]) if pid in by_id]
        measurements = []
    else:
        ids = None
        if mode == "full_documents":
            docs = {c["doc_id"] for c in raw_chunks(dataset, set(retrieved_ids))}
            ids = {p["id"] for d in dataset["documents"] if d["id"] in docs for p in d["paragraphs"]}
        chunks = raw_chunks(dataset, ids)
        measurements = []
    experiments = [e for e in records["experiments"] if e["id"] in {m["experiment_id"] for m in measurements}]
    doc_ids = {c["doc_id"] for c in chunks}
    documents = [{k:d[k] for k in ("id","title","url","license")} for d in dataset["documents"] if d["id"] in doc_ids]
    context_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(_context(chunks)))
    if context_tokens > CONTEXT_LIMIT:
        raise RuntimeError("문맥 상한을 초과했습니다.")
    return {"question":question.strip(), "chunks":chunks, "documents":documents,
            "experiments":experiments,"measurements":measurements,"paths":paths,
            "metadata":{"mode":mode,"context_tokens":context_tokens,"query_concepts":concepts,
                        "retrieved_paragraph_ids":list(dict.fromkeys(retrieved_ids)),
                        "sdk_entity_count":len(sdk_data.get("entities",[])),
                        "sdk_relationship_count":len(sdk_data.get("relationships",[])),
                        "model_calls":calls}}


async def _ask(question: str, mode="graph"):
    started = time.perf_counter()
    result = await prepare(question, mode)
    retrieval_seconds = time.perf_counter() - started
    answer, info = await asyncio.to_thread(provider.generate, ANSWER_PROMPT,
        f"질문: {question.strip()}\n\n문헌 원문 문단:\n{_context(result['chunks'])}",
        1800, model=ANSWER_MODEL)
    references = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if any(n < 1 or n > len(result["chunks"]) for n in references):
        raise RuntimeError("답변의 인용 번호를 검색된 원문과 연결할 수 없습니다.")
    result["answer"] = answer
    result["metadata"]["model_calls"].append({"kind":"answer_llm",**info})
    result["metadata"].update(elapsed_seconds=time.perf_counter()-started,retrieval_seconds=retrieval_seconds)
    return result


def ask(question: str, mode="graph"):
    return asyncio.run(_ask(question, mode))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "ask"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=["all","naive","graph","full_documents"], default="graph")
    args = parser.parse_args()
    if args.command == "index":
        index()
    else:
        print(json.dumps(ask(args.question, args.mode), ensure_ascii=False, indent=2))

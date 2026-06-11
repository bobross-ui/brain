"""LOCOMO evaluation harness for Brain.

Three phases, all driving Brain's public API:

  1. Ingest   - replay each conversation session-by-session into MemoryService.ingest_session(),
                using one Scope(user_id=sample_id) per conversation for isolation
                (search filters on user_id + namespace, not agent_id).
  2. Retrieve - for each question, either MemoryService.recall_evidence(...)
                over fused distilled memories + raw turns, or
                MemoryService.search_turns(...) over raw turns only. The raw-turn mode
                reports retrieval recall only and skips answer/judge.
  3. Score    - true evidence recall from gold/retrieved dia_ids, the previous
                answer-token recall@k proxy for comparison, and correctness via an
                optional LLM judge (falls back to a token-overlap heuristic).
                Predictions are also written to JSONL for offline grading.

The memory backend (extraction + reconciliation) is whatever LLM_PROVIDER / LLM_MODEL
point at (ollama or deepseek). The answerer and judge are chosen independently via flags
so you can, e.g., run a local memory backend but a stronger answerer.

Dataset: the public LOCOMO file (commonly locomo10.json). Pull it and confirm the exact
schema / category mapping from the LOCOMO repo (snap-research/locomo); this loader is
tolerant but the CATEGORY_NAMES mapping below is best-effort.

Usage:
    uv run python scripts/eval_locomo.py --dataset path/to/locomo10.json
    uv run python scripts/eval_locomo.py --dataset locomo10.json --max-conversations 1 \
        --answerer-provider deepseek --answerer-model deepseek-v4-pro \
        --judge deepseek --judge-model deepseek-v4-pro

Start small: a full LOCOMO conversation is hundreds of turns; through a local 3b model
each conversation ingests slowly. The default processes one conversation.
"""

import argparse
import asyncio
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from brain.config import settings
from brain.llm import LLMClient, build_llm_client
from brain.memory import build_memory
from brain.models import RetrievedEvidence, Scope, ScoredMemory, SessionInput, Turn
from brain.temporal import query_date_filters


# Best-effort; confirm against the LOCOMO dataset/repo before citing per-category numbers.
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "that", "this", "it",
    "do", "does", "did", "has", "have", "had", "what", "when", "where", "who", "how",
    "why", "which", "they", "their", "them", "she", "he", "her", "his", "you", "your",
}

_ABSTAIN_MARKERS = (
    "no information", "not mentioned", "not available", "don't know", "do not know",
    "cannot answer", "can't answer", "no answer", "unknown", "not stated", "not specified",
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
ANSWER_SYSTEM = (
    "You answer questions using ONLY the evidence provided about the speakers. "
    "If the evidence does not contain enough information to answer, reply with exactly: "
    "No information available. Keep the answer short and factual."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["correct"],
}
JUDGE_SYSTEM = (
    "You grade a predicted answer against a reference answer for a question. "
    "Mark correct=true if the prediction conveys the same information as the reference, "
    "even if worded differently. If the reference indicates no information is available, "
    "the prediction is correct only if it also declines to answer. Respond as json."
)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _is_abstention(text: str) -> bool:
    low = str(text).lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


def _as_bool(value: object) -> bool:
    """Coerce a judge verdict to bool.

    The judge schema asks for a boolean, but DeepSeek's JSON mode does not enforce
    schemas, so ``correct`` can come back as the string "false" — and ``bool("false")``
    is ``True``, which would silently score every wrong answer as correct.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def _session_keys(conversation: dict) -> list[str]:
    keyed = []
    for key in conversation:
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(conversation[key], list):
            keyed.append((int(match.group(1)), key))
    return [key for _, key in sorted(keyed)]


def load_locomo(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("data") or data.get("conversations") or [data]
    return data


def _conversation_block(sample: dict) -> dict:
    block = sample.get("conversation")
    return block if isinstance(block, dict) else sample


def _qa_items(sample: dict) -> list[dict]:
    return sample.get("qa") or sample.get("questions") or []


def _gold_answer(item: dict) -> str:
    answer = item.get("answer")
    if answer is None:
        answer = item.get("adversarial_answer")
    return "" if answer is None else str(answer)


def _gold_evidence(item: dict) -> list[str]:
    evidence = item.get("evidence")
    if evidence is None:
        return []
    if isinstance(evidence, str):
        return [evidence] if evidence else []
    if not isinstance(evidence, (list, tuple, set)):
        return []
    return [str(source_turn_id) for source_turn_id in evidence if source_turn_id]


async def ingest_conversation(
    service,
    sample: dict,
    scope: Scope,
    max_sessions: int,
) -> int:
    conversation = _conversation_block(sample)
    keys = _session_keys(conversation)
    if max_sessions:
        keys = keys[:max_sessions]

    sessions_ingested = 0
    for key in keys:
        date_time = conversation.get(f"{key}_date_time")
        turns: list[Turn] = []
        for turn in conversation[key]:
            speaker = turn.get("speaker") or turn.get("role") or "user"
            text = turn.get("text") or turn.get("content") or ""
            if text:
                source_turn_id = turn.get("dia_id") or turn.get("source_turn_id")
                turns.append(
                    Turn(
                        speaker=str(speaker),
                        text=str(text),
                        source_turn_id=str(source_turn_id)
                        if source_turn_id is not None
                        else None,
                        observed_at=str(date_time) if date_time else None,
                    )
                )

        if turns:
            speaker_roster = {
                roster_key: conversation[roster_key]
                for roster_key in ("speaker_a", "speaker_b")
                if roster_key in conversation
            }
            await service.ingest_session(
                SessionInput(
                    turns=turns,
                    source_session_id=key,
                    observed_at=str(date_time) if date_time else None,
                    speaker_roster=speaker_roster or None,
                ),
                scope,
            )
            sessions_ingested += 1
    return sessions_ingested


async def answer_question(
    llm: LLMClient,
    question: str,
    retrieved: list[RetrievedEvidence],
) -> str:
    evidence = "\n".join(
        f"{index}. {_format_evidence(item)}"
        for index, item in enumerate(retrieved)
    ) or "(no evidence retrieved)"
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"Evidence:\n{evidence}\n\nQuestion: {question}",
        },
    ]
    result = await llm.chat_json(messages, ANSWER_SCHEMA, temperature=0.0)
    return str(result.get("answer", "")).strip()


def _format_evidence(item: RetrievedEvidence) -> str:
    if item.event_date is not None:
        return f"[{item.event_date}] {item.content}"
    if item.event_date_start is not None and item.event_date_end is not None:
        return (
            f"[{item.event_date_start} to {item.event_date_end}] "
            f"{item.content}"
        )
    return item.content


async def judge_answer(llm: LLMClient, question: str, gold: str, prediction: str) -> bool:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Reference answer: {gold}\n"
                f"Predicted answer: {prediction}"
            ),
        },
    ]
    result = await llm.chat_json(messages, JUDGE_SCHEMA, temperature=0.0)
    return _as_bool(result.get("correct"))


def _retrieved_text(retrieved: list[ScoredMemory] | list[RetrievedEvidence]) -> str:
    chunks: list[str] = []
    for item in retrieved:
        if hasattr(item, "memory"):
            chunks.append(item.memory.content)
        else:
            chunks.append(item.content)
    return " ".join(chunks)


def recall_at_k(
    gold: str,
    retrieved: list[ScoredMemory] | list[RetrievedEvidence],
) -> bool | None:
    """Proxy: are the gold answer's content words present in the retrieved memories?

    Returns None when recall is not meaningful (empty/abstention answers). This is a
    cheap stand-in for evidence-based recall, which would require tracking which source
    turn each memory came from.
    """
    if not gold or _is_abstention(gold):
        return None
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return None
    found = gold_tokens & _content_tokens(_retrieved_text(retrieved))
    return len(found) / len(gold_tokens) >= 0.6


def evidence_recall(
    gold_evidence: list[str],
    retrieved_turn_ids: set[str],
) -> float | None:
    gold_turn_ids = set(gold_evidence)
    if not gold_turn_ids:
        return None
    return len(gold_turn_ids & retrieved_turn_ids) / len(gold_turn_ids)


def evidence_hit(
    gold_evidence: list[str],
    retrieved_turn_ids: set[str],
) -> bool | None:
    recall = evidence_recall(gold_evidence, retrieved_turn_ids)
    return None if recall is None else recall > 0.0


def _as_retrieved_evidence(
    retrieved: list[ScoredMemory] | list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    for item in retrieved:
        if isinstance(item, RetrievedEvidence):
            evidence.append(item)
            continue

        memory = item.memory
        evidence.append(
            RetrievedEvidence(
                kind="memory",
                content=memory.content,
                score=item.score,
                memory_id=memory.id,
                source_turn_ids=memory.source_turn_ids,
                source_session_id=memory.source_session_id,
                observed_at=memory.observed_at,
                event_date=getattr(memory, "event_date", None),
                event_date_start=getattr(memory, "event_date_start", None),
                event_date_end=getattr(memory, "event_date_end", None),
            )
        )
    return evidence


def _retrieved_turn_ids(retrieved: list[RetrievedEvidence]) -> set[str]:
    return {
        source_turn_id
        for hit in retrieved
        for source_turn_id in hit.source_turn_ids
        if source_turn_id
    }


def _evidence_fields(
    gold_evidence: list[str],
    retrieved: list[ScoredMemory] | list[RetrievedEvidence],
) -> dict:
    retrieved_turn_ids = _retrieved_turn_ids(_as_retrieved_evidence(retrieved))
    return {
        "gold_evidence": gold_evidence,
        "retrieved_turn_ids": sorted(retrieved_turn_ids),
        "evidence_recall": evidence_recall(gold_evidence, retrieved_turn_ids),
        "evidence_hit": evidence_hit(gold_evidence, retrieved_turn_ids),
    }


def heuristic_correct(gold: str, prediction: str) -> bool | None:
    """Token-overlap fallback when no LLM judge is configured."""
    if _is_abstention(gold):
        return _is_abstention(prediction)
    gold_tokens = _content_tokens(gold)
    if not gold_tokens:
        return None
    return len(gold_tokens & _content_tokens(prediction)) / len(gold_tokens) >= 0.6


async def score_prediction(
    category: str,
    question: str,
    gold: str,
    prediction: str,
    judge: LLMClient | None,
) -> tuple[bool | None, bool]:
    """Score a prediction, returning (correct, judged).

    Adversarial (category 5) items carry no gold answer — `_gold_answer` falls back
    to `adversarial_answer`, which is the trap a fooled system would give. Judging
    against it inverts the score, so the correct behavior is abstention and no LLM
    judge runs for these items.
    """
    if category == "adversarial":
        return _is_abstention(prediction), False
    if judge is not None:
        return await judge_answer(judge, question, gold, prediction), True
    return heuristic_correct(gold, prediction), False


def _mean(values: list[bool | float | None]) -> float | None:
    scored = [float(value) for value in values if value is not None]
    return sum(scored) / len(scored) if scored else None


def _summary_evidence_recall(record: dict) -> float | bool | None:
    if "evidence_recall" in record:
        return record["evidence_recall"]
    return record.get("recall_hit")


def print_summary(records: list[dict], judged: bool) -> None:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    label = "judge" if judged else "heuristic"
    print("\n" + "=" * 76)
    print(f"LOCOMO results  ({len(records)} questions)")
    print("=" * 76)
    header = (
        f"{'category':<14}{'n':>5}{'  acc(' + label + ')':>18}"
        f"{'recall@k':>12}{'ev_recall':>12}"
    )
    print(header)
    print("-" * 76)
    for category in sorted(by_category):
        rows = by_category[category]
        acc = _mean([row["correct"] for row in rows])
        recall = _mean([row["recall_hit"] for row in rows])
        ev_recall = _mean([_summary_evidence_recall(row) for row in rows])
        acc_text = f"{acc:.3f}" if acc is not None else "n/a"
        recall_text = f"{recall:.3f}" if recall is not None else "n/a"
        ev_recall_text = f"{ev_recall:.3f}" if ev_recall is not None else "n/a"
        print(
            f"{category:<14}{len(rows):>5}{acc_text:>18}"
            f"{recall_text:>12}{ev_recall_text:>12}"
        )
    print("-" * 76)
    overall_acc = _mean([row["correct"] for row in records])
    overall_recall = _mean([row["recall_hit"] for row in records])
    overall_ev_recall = _mean(
        [_summary_evidence_recall(row) for row in records]
    )
    acc_text = f"{overall_acc:.3f}" if overall_acc is not None else "n/a"
    recall_text = f"{overall_recall:.3f}" if overall_recall is not None else "n/a"
    ev_recall_text = (
        f"{overall_ev_recall:.3f}" if overall_ev_recall is not None else "n/a"
    )
    print(
        f"{'OVERALL':<14}{len(records):>5}{acc_text:>18}"
        f"{recall_text:>12}{ev_recall_text:>12}"
    )
    print("=" * 76)


async def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    samples = load_locomo(dataset)
    if args.max_conversations:
        samples = samples[: args.max_conversations]

    answerer_provider = args.answerer_provider or settings.llm_provider
    answerer_model = args.answerer_model or settings.llm_model
    answerer = None
    if args.retrieve_from == "memories":
        answerer = build_llm_client(answerer_provider, answerer_model)

    judge = None
    if args.retrieve_from == "memories" and args.judge != "none":
        judge_model = args.judge_model or (
            "deepseek-v4-pro" if args.judge == "deepseek" else settings.llm_model
        )
        judge = build_llm_client(args.judge, judge_model)

    print(
        f"Memory backend: {settings.llm_provider}/{settings.llm_model}  |  "
        f"retrieve_from: {args.retrieve_from}  |  "
        f"answerer: {answerer_provider + '/' + answerer_model if answerer else 'skipped'}  |  "
        f"judge: {args.judge if judge else 'none (heuristic)'}"
    )
    print(f"Conversations: {len(samples)}  |  retrieval k={args.k}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    if args.db_path:
        settings.brain_db_path = args.db_path
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="brain-eval-locomo-")
        settings.brain_db_path = str(Path(temp_dir.name) / "eval.db")

    records: list[dict] = []
    try:
        service = build_memory()  # one embedder load, scope isolates conversations
        for conv_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id") or f"conv-{conv_index}")
            scope = Scope(user_id=sample_id)

            started = time.time()
            sessions = await ingest_conversation(service, sample, scope, args.max_sessions)
            qa_items = _qa_items(sample)
            if args.max_questions:
                qa_items = qa_items[: args.max_questions]
            print(
                f"[{conv_index + 1}/{len(samples)}] {sample_id}: ingested {sessions} "
                f"session(s) in {time.time() - started:.1f}s, {len(qa_items)} question(s)"
            )

            for item in qa_items:
                question = item.get("question")
                if not question:
                    continue
                gold = _gold_answer(item)
                gold_evidence = _gold_evidence(item)
                category = CATEGORY_NAMES.get(item.get("category"), str(item.get("category")))

                if args.retrieve_from == "turns":
                    retrieved_turns = await service.search_turns(
                        question,
                        scope,
                        limit=args.k,
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "category": category,
                            "question": question,
                            "gold": gold,
                            "prediction": "",
                            "retrieved": [hit.content for hit in retrieved_turns],
                            **_evidence_fields(gold_evidence, retrieved_turns),
                            "recall_hit": recall_at_k(gold, retrieved_turns),
                            "correct": None,
                            "judged": False,
                        }
                    )
                else:
                    date_filters = query_date_filters(question)
                    if date_filters:
                        retrieved = await service.recall_evidence(
                            question,
                            scope,
                            limit=args.k,
                            filters=date_filters,
                        )
                    else:
                        retrieved = await service.recall_evidence(
                            question,
                            scope,
                            limit=args.k,
                        )
                    prediction = await answer_question(answerer, question, retrieved)

                    correct, was_judged = await score_prediction(
                        category,
                        question,
                        gold,
                        prediction,
                        judge,
                    )

                    records.append(
                        {
                            "sample_id": sample_id,
                            "category": category,
                            "question": question,
                            "gold": gold,
                            "prediction": prediction,
                            "retrieved": [hit.content for hit in retrieved],
                            **_evidence_fields(gold_evidence, retrieved),
                            "recall_hit": recall_at_k(gold, retrieved),
                            "correct": correct,
                            "judged": was_judged,
                        }
                    )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print_summary(records, judged=judge is not None)
    print(f"\nPredictions written to {out_path}")
    if args.retrieve_from == "turns":
        print("Raw-turn mode skips answer generation and judge scoring.")
    elif judge is None:
        print(
            "No LLM judge was used; 'acc' is a token-overlap heuristic. For real numbers, "
            "rerun with --judge deepseek (or grade the JSONL with the LOCOMO judge)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Brain on LOCOMO.")
    parser.add_argument("--dataset", required=True, help="Path to LOCOMO json (e.g. locomo10.json)")
    parser.add_argument("--out", default="eval_results/locomo_predictions.jsonl")
    parser.add_argument("--k", type=int, default=10, help="Retrieval limit at QA time")
    parser.add_argument(
        "--max-conversations", type=int, default=1, help="Process at most N conversations (0=all)"
    )
    parser.add_argument(
        "--max-questions", type=int, default=0, help="Per conversation, at most N questions (0=all)"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0, help="Per conversation, ingest at most N sessions (0=all)"
    )
    parser.add_argument("--answerer-provider", choices=("ollama", "deepseek"), default=None)
    parser.add_argument("--answerer-model", default=None)
    parser.add_argument("--judge", choices=("none", "ollama", "deepseek"), default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--db-path", default=None, help="Memory DB path (default: fresh temp DB)")
    parser.add_argument(
        "--retrieve-from",
        choices=("memories", "turns"),
        default="memories",
        help="Retrieve distilled memories or raw turns at QA time",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))

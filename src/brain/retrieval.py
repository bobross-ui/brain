import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence, TypeVar


RRF_K = 60
FILTER_OVERFETCH_MIN = 200
FILTER_OVERFETCH_MULTIPLIER = 10
# Fusion only needs enough list depth for the retrievers to overlap; past ~50
# the RRF contribution (1/(RRF_K + rank)) is too small to move the top-k.
FUSION_DEPTH_MIN = 50
FUSION_DEPTH_MULTIPLIER = 5
RERANK_DEPTH = 50

T = TypeVar("T")


@dataclass(frozen=True)
class FilterSpec:
    subject: str | None = None
    event_after: str | None = None
    event_before: str | None = None
    event_on: str | None = None

    @classmethod
    def from_dict(cls, filters: dict | None) -> "FilterSpec":
        if not filters:
            return cls()

        unsupported = set(filters) - {
            "subject",
            "event_after",
            "event_before",
            "event_on",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported memory filters: {names}")
        for name in ("event_after", "event_before", "event_on"):
            value = filters.get(name)
            if value is not None:
                try:
                    date.fromisoformat(str(value))
                except ValueError as exc:
                    raise ValueError(
                        f"Memory filter {name} must be an ISO-8601 date"
                    ) from exc
        return cls(
            subject=filters.get("subject"),
            event_after=filters.get("event_after"),
            event_before=filters.get("event_before"),
            event_on=filters.get("event_on"),
        )

    @property
    def active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.subject,
                self.event_after,
                self.event_before,
                self.event_on,
            )
        )


def candidate_limit(limit: int, *, overfetch: bool) -> int:
    if not overfetch:
        return limit
    return max(FILTER_OVERFETCH_MIN, FILTER_OVERFETCH_MULTIPLIER * limit)


def fusion_limit(limit: int) -> int:
    return max(FUSION_DEPTH_MIN, FUSION_DEPTH_MULTIPLIER * limit)


def search_pool_limit(limit: int, *, filters_active: bool, mode: str) -> int:
    """Per-retriever candidate depth for store search.

    Non-partition filters need the deep pool (the matching row may rank far
    below the unfiltered top-k); plain hybrid fusion only needs overlap depth;
    single-retriever modes need no extra candidates at all.
    """
    if filters_active:
        return candidate_limit(limit, overfetch=True)
    if mode == "hybrid":
        return fusion_limit(limit)
    return limit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = RRF_K,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class Reranker(Protocol):
    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class CrossEncoderReranker:
    """Local sentence-transformers cross-encoder, loaded on first use."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        def _work() -> list[float]:
            model = self._load_model()
            scores = model.predict([(query, document) for document in documents])
            return [float(score) for score in scores]

        return await asyncio.to_thread(_work)


async def rerank(
    query: str,
    items: Sequence[T],
    documents: Sequence[str],
    reranker: Reranker | None,
    *,
    depth: int = RERANK_DEPTH,
) -> list[tuple[T, float]] | None:
    if reranker is None or not items:
        return None
    items = list(items)[:depth]
    documents = list(documents)[:depth]
    scores = await reranker.score(query, documents)
    if len(scores) != len(items):
        raise RuntimeError("Reranker returned an unexpected number of scores")
    return sorted(
        zip(items, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

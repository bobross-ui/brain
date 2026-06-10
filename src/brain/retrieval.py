import asyncio
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar


RRF_K = 60
FILTER_OVERFETCH_MIN = 200
FILTER_OVERFETCH_MULTIPLIER = 10

T = TypeVar("T")


@dataclass(frozen=True)
class FilterSpec:
    subject: str | None = None

    @classmethod
    def from_dict(cls, filters: dict | None) -> "FilterSpec":
        if not filters:
            return cls()

        unsupported = set(filters) - {"subject"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported memory filters: {names}")
        return cls(subject=filters.get("subject"))

    @property
    def active(self) -> bool:
        return self.subject is not None


def candidate_limit(limit: int, *, overfetch: bool) -> int:
    if not overfetch:
        return limit
    return max(FILTER_OVERFETCH_MIN, FILTER_OVERFETCH_MULTIPLIER * limit)


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
) -> list[tuple[T, float]] | None:
    if reranker is None or not items:
        return None
    scores = await reranker.score(query, documents)
    if len(scores) != len(items):
        raise RuntimeError("Reranker returned an unexpected number of scores")
    return sorted(
        zip(items, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

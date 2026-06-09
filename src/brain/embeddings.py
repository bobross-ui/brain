import asyncio
import math
from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of vectors produced by ``embed``."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedder(Embedder):
    """Local free embedder. Downloads model once on first run."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        vec = await asyncio.to_thread(
            self._model.encode,
            text,
            normalize_embeddings=True,
        )
        return vec.tolist()


class FakeEmbedder(Embedder):
    """Deterministic, semantically meaningless D=384 embedder for tests."""

    D = 384

    @property
    def dim(self) -> int:
        return self.D

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.D
        for index, char in enumerate(text):
            vec[index % self.D] += ord(char)

        magnitude = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / magnitude for value in vec]

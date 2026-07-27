from __future__ import annotations
from abc import ABC, abstractmethod

JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        raise NotImplementedError

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """批量获取 embedding；未覆盖时保持逐条兼容。"""

        return [self.get_embeddings(text) for text in texts]

    async def agenerate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        import asyncio
        return await asyncio.to_thread(
            self.generate,
            system,
            user,
            response_format=response_format,
        )

    async def aget_embeddings(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.to_thread(self.get_embeddings, text)

    async def aget_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self.get_embeddings_batch, texts)

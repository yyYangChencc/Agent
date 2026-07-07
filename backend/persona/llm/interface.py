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

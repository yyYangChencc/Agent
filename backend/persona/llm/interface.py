from __future__ import annotations
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        raise NotImplementedError

    async def agenerate(self, system: str, user: str) -> str:
        import asyncio
        return await asyncio.to_thread(self.generate, system, user)

    async def aget_embeddings(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.to_thread(self.get_embeddings, text)

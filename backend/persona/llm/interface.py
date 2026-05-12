from __future__ import annotations
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        raise NotImplementedError

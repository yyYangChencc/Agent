from __future__ import annotations
from openai import OpenAI, AsyncOpenAI
from persona.llm.interface import LLMClient
from persona.config import AgentConfig
from persona.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str, config: AgentConfig | None = None):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._config = config or AgentConfig()

    def get_embeddings(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._config.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


class AsyncOpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str, config: AgentConfig | None = None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._config = config or AgentConfig()

    def generate(self, system: str, user: str) -> str:
        import asyncio
        return asyncio.run(self.agenerate(system, user))

    def get_embeddings(self, text: str) -> list[float]:
        import asyncio
        return asyncio.run(self.aget_embeddings(text))

    async def agenerate(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    async def aget_embeddings(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self._config.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

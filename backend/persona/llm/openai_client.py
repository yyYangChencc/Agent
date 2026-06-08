from __future__ import annotations
from openai import OpenAI, AsyncOpenAI
from persona.llm.interface import LLMClient
from persona.config import AgentConfig
from persona.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, base_url: str,embedding_key: str, embedding_base_url: str, config: AgentConfig | None = None):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._embedding_client = OpenAI(api_key=embedding_key, base_url=embedding_base_url)
        self._config = config or AgentConfig()

    def get_embeddings(self, text: str) -> list[float]:
        resp = self._embedding_client.embeddings.create(
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
    def __init__(self, api_key: str, base_url: str, embedding_key: str, embedding_base_url: str, config: AgentConfig | None = None):
        # 同时持有同步/异步客户端：
        # - 异步路径使用 AsyncOpenAI（await 调用）
        # - 同步路径使用 OpenAI（避免在运行中的事件循环里再调用 asyncio.run）
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._sclient = OpenAI(api_key=api_key, base_url=base_url)
        self._embedding_client = OpenAI(api_key=embedding_key, base_url=embedding_base_url)
        self._config = config or AgentConfig()

    def generate(self, system: str, user: str) -> str:
        resp = self._sclient.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def get_embeddings(self, text: str) -> list[float]:
        resp = self._sclient.embeddings.create(
            model=self._config.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    async def agenerate(self, system: str, user: str) -> str:
        resp = await self._aclient.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    async def aget_embeddings(self, text: str) -> list[float]:
        resp = await self._embedding_client.embeddings.create(  
            model=self._config.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

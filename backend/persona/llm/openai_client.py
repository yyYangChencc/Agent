from __future__ import annotations

import time

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI

from persona.config import AgentConfig
from persona.llm.interface import LLMClient
from persona.logger import get_logger

logger = get_logger(__name__)


def _build_timeout(timeout_seconds: float, connect_timeout_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=timeout_seconds, connect=connect_timeout_seconds)


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        embedding_key: str,
        embedding_base_url: str,
        config: AgentConfig | None = None,
    ):
        self._config = config or AgentConfig()
        self._base_url = base_url
        self._embedding_base_url = embedding_base_url
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_build_timeout(
                self._config.llm_timeout_seconds,
                self._config.llm_connect_timeout_seconds,
            ),
            max_retries=self._config.llm_max_retries,
        )
        self._embedding_client = OpenAI(
            api_key=embedding_key,
            base_url=embedding_base_url,
            timeout=_build_timeout(
                self._config.embedding_timeout_seconds,
                self._config.embedding_connect_timeout_seconds,
            ),
            max_retries=self._config.embedding_max_retries,
        )

    def get_embeddings(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        try:
            resp = self._embedding_client.embeddings.create(
                model=self._config.embedding_model,
                input=text,
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
            )
            return []
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
                e,
            )
            return []
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI embedding completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.embedding_model,
            self._embedding_base_url,
        )
        return resp.data[0].embedding

    def generate(self, system: str, user: str) -> str:
        started_at = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
            )
            return ""
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
                e,
            )
            return ""
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI chat completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.llm_model,
            self._base_url,
        )
        return resp.choices[0].message.content or ""


class AsyncOpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        embedding_key: str,
        embedding_base_url: str,
        config: AgentConfig | None = None,
    ):
        self._config = config or AgentConfig()
        self._base_url = base_url
        self._embedding_base_url = embedding_base_url
        llm_timeout = _build_timeout(
            self._config.llm_timeout_seconds,
            self._config.llm_connect_timeout_seconds,
        )
        embedding_timeout = _build_timeout(
            self._config.embedding_timeout_seconds,
            self._config.embedding_connect_timeout_seconds,
        )
        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=llm_timeout,
            max_retries=self._config.llm_max_retries,
        )
        self._sclient = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=llm_timeout,
            max_retries=self._config.llm_max_retries,
        )
        self._embedding_client = OpenAI(
            api_key=embedding_key,
            base_url=embedding_base_url,
            timeout=embedding_timeout,
            max_retries=self._config.embedding_max_retries,
        )
        self._a_embedding_client = AsyncOpenAI(
            api_key=embedding_key,
            base_url=embedding_base_url,
            timeout=embedding_timeout,
            max_retries=self._config.embedding_max_retries,
        )

    def generate(self, system: str, user: str) -> str:
        started_at = time.perf_counter()
        try:
            resp = self._sclient.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
            )
            return ""
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
                e,
            )
            return ""
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI chat completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.llm_model,
            self._base_url,
        )
        return resp.choices[0].message.content or ""

    def get_embeddings(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        try:
            resp = self._embedding_client.embeddings.create(
                model=self._config.embedding_model,
                input=text,
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
            )
            return []
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
                e,
            )
            return []
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI embedding completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.embedding_model,
            self._embedding_base_url,
        )
        return resp.data[0].embedding

    async def agenerate(self, system: str, user: str) -> str:
        started_at = time.perf_counter()
        try:
            resp = await self._aclient.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
            )
            return ""
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
                e,
            )
            return ""
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI chat completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.llm_model,
            self._base_url,
        )
        return resp.choices[0].message.content or ""

    async def aget_embeddings(self, text: str) -> list[float]:
        started_at = time.perf_counter()
        try:
            resp = await self._a_embedding_client.embeddings.create(
                model=self._config.embedding_model,
                input=text,
            )
        except APITimeoutError:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding timeout after %.3fs; model=%s base_url=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
            )
            return []
        except APIConnectionError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding connection error after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
                e,
            )
            return []
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI embedding completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.embedding_model,
            self._embedding_base_url,
        )
        return resp.data[0].embedding

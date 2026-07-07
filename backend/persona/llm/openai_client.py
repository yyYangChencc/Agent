from __future__ import annotations

import asyncio
import time

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from persona.config import AgentConfig
from persona.llm.interface import LLMClient
from persona.logger import get_logger

logger = get_logger(__name__)


def _build_timeout(timeout_seconds: float, connect_timeout_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=timeout_seconds, connect=connect_timeout_seconds)


def _positive_int(value, default: int) -> int:
    """把配置值归一化为正整数，防止无效配置破坏并发控制。"""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _non_negative_int(value, default: int) -> int:
    """把配置值归一化为非负整数，用于有限次重试。"""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _retry_delay(base_delay: float, attempt: int) -> float:
    """429 退避时间；attempt 从 0 开始。"""

    try:
        base = float(base_delay)
    except (TypeError, ValueError):
        base = 1.0
    return max(0.1, base) * (2 ** attempt)


def _chat_kwargs(config: AgentConfig, system: str, user: str, response_format: dict | None = None) -> dict:
    """组装 chat.completions 参数；仅 JSON 调用显式传入 response_format。"""

    kwargs = {
        "model": config.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


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
        except RateLimitError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding rate limit after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
                e,
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

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        started_at = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                **_chat_kwargs(self._config, system, user, response_format)
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
        except RateLimitError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat rate limit after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
                e,
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
        self._chat_semaphore = asyncio.Semaphore(
            _positive_int(self._config.llm_max_concurrent_requests, 3)
        )
        self._embedding_semaphore = asyncio.Semaphore(
            _positive_int(self._config.embedding_max_concurrent_requests, 3)
        )
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

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        started_at = time.perf_counter()
        try:
            resp = self._sclient.chat.completions.create(
                **_chat_kwargs(self._config, system, user, response_format)
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
        except RateLimitError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI chat rate limit after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.llm_model,
                self._base_url,
                e,
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
        except RateLimitError as e:
            elapsed = time.perf_counter() - started_at
            logger.warning(
                "OpenAI embedding rate limit after %.3fs; model=%s base_url=%s error=%s",
                elapsed,
                self._config.embedding_model,
                self._embedding_base_url,
                e,
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

    async def agenerate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        started_at = time.perf_counter()
        retries = _non_negative_int(self._config.llm_rate_limit_retries, 3)
        for attempt in range(retries + 1):
            try:
                async with self._chat_semaphore:
                    resp = await self._aclient.chat.completions.create(
                        **_chat_kwargs(self._config, system, user, response_format)
                    )
                break
            except RateLimitError as e:
                elapsed = time.perf_counter() - started_at
                if attempt >= retries:
                    logger.warning(
                        "OpenAI chat rate limit after %.3fs; attempts=%d model=%s base_url=%s error=%s",
                        elapsed,
                        attempt + 1,
                        self._config.llm_model,
                        self._base_url,
                        e,
                    )
                    return ""
                delay = _retry_delay(self._config.llm_rate_limit_backoff_seconds, attempt)
                logger.warning(
                    "OpenAI chat rate limit; retrying in %.2fs attempt=%d/%d model=%s base_url=%s",
                    delay,
                    attempt + 1,
                    retries,
                    self._config.llm_model,
                    self._base_url,
                )
                await asyncio.sleep(delay)
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
        else:
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
        retries = _non_negative_int(self._config.embedding_rate_limit_retries, 2)
        for attempt in range(retries + 1):
            try:
                async with self._embedding_semaphore:
                    resp = await self._a_embedding_client.embeddings.create(
                        model=self._config.embedding_model,
                        input=text,
                    )
                break
            except RateLimitError as e:
                elapsed = time.perf_counter() - started_at
                if attempt >= retries:
                    logger.warning(
                        "OpenAI embedding rate limit after %.3fs; attempts=%d model=%s base_url=%s error=%s",
                        elapsed,
                        attempt + 1,
                        self._config.embedding_model,
                        self._embedding_base_url,
                        e,
                    )
                    return []
                delay = _retry_delay(self._config.embedding_rate_limit_backoff_seconds, attempt)
                logger.warning(
                    "OpenAI embedding rate limit; retrying in %.2fs attempt=%d/%d model=%s base_url=%s",
                    delay,
                    attempt + 1,
                    retries,
                    self._config.embedding_model,
                    self._embedding_base_url,
                )
                await asyncio.sleep(delay)
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
        else:
            return []
        elapsed = time.perf_counter() - started_at
        logger.debug(
            "OpenAI embedding completed in %.3fs; model=%s base_url=%s",
            elapsed,
            self._config.embedding_model,
            self._embedding_base_url,
        )
        return resp.data[0].embedding

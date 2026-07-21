from __future__ import annotations

import asyncio
import threading
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError

from persona.config import AgentConfig
from persona.llm.interface import LLMClient
from persona.logger import get_logger

logger = get_logger(__name__)
_RETRYABLE_SERVER_STATUS = {502, 503, 504}


class _FailureCircuit:
    """连续服务失败达到阈值后开启短暂冷却。"""

    def __init__(self, threshold: int, cooldown_seconds: float):
        self.threshold = max(1, int(threshold))
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))
        self.failures = 0
        self.blocked_until = 0.0
        self.probe_in_flight = False
        self.lock = threading.Lock()

    def is_blocked(self) -> bool:
        return self.cooldown_remaining() > 0

    def cooldown_remaining(self) -> float:
        """返回剩余冷却秒数，供异步请求等待服务恢复。"""

        with self.lock:
            return max(0.0, self.blocked_until - time.monotonic())

    def request_permission(self) -> tuple[bool, float, bool]:
        """冷却到期后只放行一个探测请求。"""

        with self.lock:
            now = time.monotonic()
            if now < self.blocked_until:
                return False, self.blocked_until - now, False
            if self.blocked_until > 0:
                if self.probe_in_flight:
                    return False, 0.05, False
                self.probe_in_flight = True
                return True, 0.0, True
            return True, 0.0, False

    def release_probe(self, probe: bool) -> None:
        """重试前释放探测资格，让下一次请求重新竞争。"""

        if not probe:
            return
        with self.lock:
            self.probe_in_flight = False

    def success(self, probe: bool = False) -> None:
        with self.lock:
            if probe:
                # half-open 探测成功后才恢复正常流量。
                self.failures = 0
                self.blocked_until = 0.0
                self.probe_in_flight = False
            elif self.blocked_until <= 0:
                self.failures = 0

    def failure(self, probe: bool = False) -> None:
        with self.lock:
            if probe:
                self.blocked_until = time.monotonic() + self.cooldown_seconds
                self.probe_in_flight = False
                self.failures = 0
                return
            self.failures += 1
            if self.failures >= self.threshold:
                self.blocked_until = time.monotonic() + self.cooldown_seconds
                self.probe_in_flight = False
                self.failures = 0


def _is_retryable_server_error(exc: APIStatusError) -> bool:
    """只重试日志中已经确认出现的网关和服务错误。"""

    return exc.status_code in _RETRYABLE_SERVER_STATUS


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


def _sync_request(
    operation,
    *,
    circuit: _FailureCircuit,
    kind: str,
    model: str,
    base_url: str,
    rate_retries: int,
    rate_backoff: float,
    server_retries: int,
    server_backoff: float,
):
    """执行同步请求，并统一处理限流、服务错误和冷却。"""

    rate_attempt = 0
    server_attempt = 0
    started_at = time.perf_counter()
    while True:
        allowed, delay, probe = circuit.request_permission()
        if not allowed:
            # 同步归档串行等待冷却恢复，避免快速跳过后续全部投票。
            time.sleep(delay)
            continue
        try:
            response = operation()
            circuit.success(probe)
            return response
        except RateLimitError as exc:
            if rate_attempt >= rate_retries:
                circuit.failure(probe)
                logger.warning("OpenAI %s rate limit after %.3fs; model=%s base_url=%s error=%s", kind, time.perf_counter() - started_at, model, base_url, exc)
                return None
            circuit.release_probe(probe)
            delay = _retry_delay(rate_backoff, rate_attempt)
            rate_attempt += 1
            time.sleep(delay)
        except APIStatusError as exc:
            if not _is_retryable_server_error(exc):
                # 非重试状态说明服务已响应，half-open 可以结束。
                circuit.success(probe)
                logger.warning("OpenAI %s status error status=%s; model=%s base_url=%s error=%s", kind, exc.status_code, model, base_url, exc)
                return None
            if server_attempt >= server_retries:
                circuit.failure(probe)
                logger.warning("OpenAI %s server error after %.3fs status=%s attempts=%d; model=%s base_url=%s", kind, time.perf_counter() - started_at, exc.status_code, server_attempt + 1, model, base_url)
                return None
            circuit.release_probe(probe)
            delay = _retry_delay(server_backoff, server_attempt)
            server_attempt += 1
            time.sleep(delay)
        except APITimeoutError:
            circuit.failure(probe)
            logger.warning("OpenAI %s timeout after %.3fs; model=%s base_url=%s", kind, time.perf_counter() - started_at, model, base_url)
            return None
        except APIConnectionError as exc:
            if probe:
                circuit.failure(probe=True)
            logger.warning("OpenAI %s connection error after %.3fs; model=%s base_url=%s error=%s", kind, time.perf_counter() - started_at, model, base_url, exc)
            return None
        except Exception:
            # 未归类异常保持原有抛出语义，但不能永久占用探测资格。
            circuit.release_probe(probe)
            raise


async def _async_request(
    operation,
    *,
    semaphore: asyncio.Semaphore,
    circuit: _FailureCircuit,
    kind: str,
    model: str,
    base_url: str,
    rate_retries: int,
    rate_backoff: float,
    server_retries: int,
    server_backoff: float,
    total_timeout: float,
):
    """总超时覆盖信号量排队、退避和 HTTP 请求。"""

    async def run():
        rate_attempt = 0
        server_attempt = 0
        started_at = time.perf_counter()
        while True:
            allowed = False
            delay = 0.0
            probe = False
            try:
                async with semaphore:
                    # 只有拿到服务信号量后才竞争 half-open 探测资格。
                    allowed, delay, probe = circuit.request_permission()
                    if allowed:
                        response = await operation()
                if not allowed:
                    await asyncio.sleep(delay)
                    continue
                circuit.success(probe)
                return response
            except RateLimitError as exc:
                if rate_attempt >= rate_retries:
                    circuit.failure(probe)
                    logger.warning("OpenAI %s rate limit after %.3fs; model=%s base_url=%s error=%s", kind, time.perf_counter() - started_at, model, base_url, exc)
                    return None
                circuit.release_probe(probe)
                delay = _retry_delay(rate_backoff, rate_attempt)
                rate_attempt += 1
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                if not _is_retryable_server_error(exc):
                    circuit.success(probe)
                    logger.warning("OpenAI %s status error status=%s; model=%s base_url=%s error=%s", kind, exc.status_code, model, base_url, exc)
                    return None
                if server_attempt >= server_retries:
                    circuit.failure(probe)
                    logger.warning("OpenAI %s server error after %.3fs status=%s attempts=%d; model=%s base_url=%s", kind, time.perf_counter() - started_at, exc.status_code, server_attempt + 1, model, base_url)
                    return None
                circuit.release_probe(probe)
                delay = _retry_delay(server_backoff, server_attempt)
                server_attempt += 1
                await asyncio.sleep(delay)
            except APITimeoutError:
                circuit.failure(probe)
                logger.warning("OpenAI %s timeout after %.3fs; model=%s base_url=%s", kind, time.perf_counter() - started_at, model, base_url)
                return None
            except APIConnectionError as exc:
                if probe:
                    circuit.failure(probe=True)
                logger.warning("OpenAI %s connection error after %.3fs; model=%s base_url=%s error=%s", kind, time.perf_counter() - started_at, model, base_url, exc)
                return None
            except asyncio.CancelledError:
                circuit.release_probe(probe)
                raise
            except Exception:
                # 未归类异常保持原有抛出语义，但不能永久占用探测资格。
                circuit.release_probe(probe)
                raise

    try:
        return await asyncio.wait_for(run(), timeout=max(0.1, float(total_timeout)))
    except asyncio.TimeoutError:
        # 总超时包含本地排队和冷却等待，不能直接归因于上游服务失败。
        logger.warning("OpenAI %s total timeout; model=%s base_url=%s", kind, model, base_url)
        return None


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
        self._chat_circuit = _FailureCircuit(self._config.service_failure_threshold, self._config.service_cooldown_seconds)
        self._embedding_circuit = _FailureCircuit(self._config.service_failure_threshold, self._config.service_cooldown_seconds)

    def get_embeddings(self, text: str) -> list[float]:
        resp = _sync_request(
            lambda: self._embedding_client.embeddings.create(model=self._config.embedding_model, input=text),
            circuit=self._embedding_circuit, kind="embedding", model=self._config.embedding_model,
            base_url=self._embedding_base_url,
            rate_retries=_non_negative_int(self._config.embedding_rate_limit_retries, 2),
            rate_backoff=self._config.embedding_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.embedding_server_error_retries, 2),
            server_backoff=self._config.embedding_server_error_backoff_seconds,
        )
        return resp.data[0].embedding if resp is not None else []

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        resp = _sync_request(
            lambda: self._client.chat.completions.create(**_chat_kwargs(self._config, system, user, response_format)),
            circuit=self._chat_circuit, kind="chat", model=self._config.llm_model, base_url=self._base_url,
            rate_retries=_non_negative_int(self._config.llm_rate_limit_retries, 3),
            rate_backoff=self._config.llm_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.llm_server_error_retries, 2),
            server_backoff=self._config.llm_server_error_backoff_seconds,
        )
        return (resp.choices[0].message.content or "") if resp is not None else ""


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
        self._chat_concurrency = _positive_int(self._config.llm_max_concurrent_requests, 20)
        self._embedding_concurrency = _positive_int(self._config.embedding_max_concurrent_requests, 20)
        self._chat_semaphore: asyncio.Semaphore | None = None
        self._chat_semaphore_loop = None
        self._embedding_semaphore: asyncio.Semaphore | None = None
        self._embedding_semaphore_loop = None
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
        self._chat_circuit = _FailureCircuit(self._config.service_failure_threshold, self._config.service_cooldown_seconds)
        self._embedding_circuit = _FailureCircuit(self._config.service_failure_threshold, self._config.service_cooldown_seconds)

    def _current_chat_semaphore(self) -> asyncio.Semaphore:
        """为当前事件循环返回 Chat 信号量。"""

        loop = asyncio.get_running_loop()
        if self._chat_semaphore is None or self._chat_semaphore_loop is not loop:
            self._chat_semaphore = asyncio.Semaphore(self._chat_concurrency)
            self._chat_semaphore_loop = loop
        return self._chat_semaphore

    def _current_embedding_semaphore(self) -> asyncio.Semaphore:
        """为当前事件循环返回 Embedding 信号量。"""

        loop = asyncio.get_running_loop()
        if self._embedding_semaphore is None or self._embedding_semaphore_loop is not loop:
            self._embedding_semaphore = asyncio.Semaphore(self._embedding_concurrency)
            self._embedding_semaphore_loop = loop
        return self._embedding_semaphore

    def generate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        resp = _sync_request(
            lambda: self._sclient.chat.completions.create(**_chat_kwargs(self._config, system, user, response_format)),
            circuit=self._chat_circuit, kind="chat", model=self._config.llm_model, base_url=self._base_url,
            rate_retries=_non_negative_int(self._config.llm_rate_limit_retries, 3), rate_backoff=self._config.llm_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.llm_server_error_retries, 2), server_backoff=self._config.llm_server_error_backoff_seconds,
        )
        return (resp.choices[0].message.content or "") if resp is not None else ""

    def get_embeddings(self, text: str) -> list[float]:
        resp = _sync_request(
            lambda: self._embedding_client.embeddings.create(model=self._config.embedding_model, input=text),
            circuit=self._embedding_circuit, kind="embedding", model=self._config.embedding_model, base_url=self._embedding_base_url,
            rate_retries=_non_negative_int(self._config.embedding_rate_limit_retries, 2), rate_backoff=self._config.embedding_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.embedding_server_error_retries, 2), server_backoff=self._config.embedding_server_error_backoff_seconds,
        )
        return resp.data[0].embedding if resp is not None else []

    async def agenerate(self, system: str, user: str, *, response_format: dict | None = None) -> str:
        resp = await _async_request(
            lambda: self._aclient.chat.completions.create(**_chat_kwargs(self._config, system, user, response_format)),
            semaphore=self._current_chat_semaphore(), circuit=self._chat_circuit, kind="chat",
            model=self._config.llm_model, base_url=self._base_url,
            rate_retries=_non_negative_int(self._config.llm_rate_limit_retries, 3), rate_backoff=self._config.llm_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.llm_server_error_retries, 2), server_backoff=self._config.llm_server_error_backoff_seconds,
            total_timeout=self._config.llm_total_timeout_seconds,
        )
        return (resp.choices[0].message.content or "") if resp is not None else ""

    async def aget_embeddings(self, text: str) -> list[float]:
        resp = await _async_request(
            lambda: self._a_embedding_client.embeddings.create(model=self._config.embedding_model, input=text),
            semaphore=self._current_embedding_semaphore(), circuit=self._embedding_circuit, kind="embedding",
            model=self._config.embedding_model, base_url=self._embedding_base_url,
            rate_retries=_non_negative_int(self._config.embedding_rate_limit_retries, 2), rate_backoff=self._config.embedding_rate_limit_backoff_seconds,
            server_retries=_non_negative_int(self._config.embedding_server_error_retries, 2), server_backoff=self._config.embedding_server_error_backoff_seconds,
            total_timeout=self._config.embedding_total_timeout_seconds,
        )
        return resp.data[0].embedding if resp is not None else []

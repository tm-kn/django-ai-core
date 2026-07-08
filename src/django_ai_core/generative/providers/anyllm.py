"""Concrete ``GenerativeProvider`` backed by any-llm.

Shipped with django-ai-core as a batteries-included provider. ``any-llm-sdk``
is a core dependency. Error handling is provider-agnostic: we enable any-llm's
unified exceptions (``ANY_LLM_UNIFIED_EXCEPTIONS``) so it normalises every
vendor's failures into its ``AnyLLMError`` hierarchy, which we then map to
django-ai-core's own ``AICoreProviderError`` types. Consumers only ever see our
exceptions — never raw vendor errors, never ``AnyLLMError``. No vendor SDK is
imported here.

Talking to a specific vendor still needs that vendor's client installed, pulled
via an optional extra, e.g. ``django-ai-core[anthropic]`` / ``[openai]``.

Reference by dotted path from ``AI_CORE['GENERATIVE_MODELS']``::

    "provider": "django_ai_core.generative.providers.anyllm.AnyLLMProvider",
    "params": {"provider": "anthropic", "model": "claude-haiku-4-5"},
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

# any-llm only converts vendor SDK errors into its unified ``AnyLLMError``
# hierarchy when this flag is set (otherwise it re-raises the raw vendor error
# with a deprecation warning). We depend on the unified types, so enable it by
# default — ``setdefault`` leaves an explicit consumer choice untouched. Must be
# set before any completion call; any-llm reads the env var at raise time.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

from any_llm import AnyLLM
from any_llm.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    GatewayTimeoutError,
    InsufficientFundsError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
    UpstreamProviderError,
)

from ...exceptions import (
    AICoreProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)
from .base import GenerativeProvider, UsageCapture

# Unified any-llm exception -> semantic AICoreProviderError. any-llm normalises
# every vendor's failure into these types, so mapping is provider-agnostic: no
# vendor SDK import needed. any-llm's ``ProviderError`` is its own junk drawer:
# raw transport failures (connection/timeout/network), 5xx server errors, and
# anything it couldn't classify all land there, with no way to tell them apart
# short of sniffing SDK class names. We refuse to guess and map the whole bucket
# to ProviderUnexpectedError. Anything not matched here (content/finish-reason
# errors, non-AnyLLMError exceptions) also becomes ProviderUnexpectedError via
# _translate's fallback.
_EXCEPTION_MAP = (
    (RateLimitError, ProviderRateLimitError),
    (GatewayTimeoutError, ProviderTimeoutError),
    (UpstreamProviderError, ProviderUnavailableError),
    (ProviderError, ProviderUnexpectedError),
    (
        (
            AuthenticationError,
            MissingApiKeyError,
            UnsupportedProviderError,
            UnsupportedParameterError,
            InvalidRequestError,
            ModelNotFoundError,
            ContextLengthExceededError,
            InsufficientFundsError,
        ),
        ProviderConfigurationError,
    ),
)


def _translate(exc: Exception) -> AICoreProviderError:
    """Map a raw any-llm error to a semantic provider error.

    Already-semantic errors pass through. Unified any-llm exceptions map to
    their semantic equivalent; anything else is a real-but-unclassified failure.
    """
    if isinstance(exc, AICoreProviderError):
        return exc
    for exc_types, target in _EXCEPTION_MAP:
        if isinstance(exc, exc_types):
            return target(str(exc))
    return ProviderUnexpectedError(str(exc))


@contextmanager
def _translate_errors():
    """Re-raise any raw provider failure as a semantic ``AICoreProviderError``."""
    try:
        yield
    except AICoreProviderError:
        raise
    except Exception as exc:
        raise _translate(exc) from exc


def build_messages(
    prompt: object,
    *,
    system: str | None,
    history: list[dict] | None,
) -> list[dict]:
    """Assemble an OpenAI-style messages list from (system, history, prompt)."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": str(prompt)})
    return messages


# any-llm normalises every provider's response to the OpenAI-shaped
# ``ChatCompletion`` / ``ChatCompletionChunk`` (its types subclass the openai
# SDK's), so the extraction paths below are vendor-uniform: a completion always
# exposes ``.choices[0].message.content`` and a stream chunk
# ``.choices[0].delta.content``, regardless of the underlying provider. Only the
# *error* path is handled by ``_translate``. test_anyllm asserts this against
# real any-llm types.
def _delta_text(chunk: object) -> str:
    """Extract incremental text from a streaming chunk, tolerating gaps."""
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _message_text(response: object) -> str:
    return response.choices[0].message.content or ""


def _fill_usage(capture: UsageCapture, chunk: object) -> None:
    """Copy any usage on this chunk into ``capture`` (best-effort, no raise).

    any-llm sets ``.usage`` (OpenAI-shaped: ``prompt_tokens`` /
    ``completion_tokens``) only on the terminal chunk of a clean finish. This is
    the sole confinement point for any-llm's untyped surface: we read ``Any``
    here and write typed ``int | None`` onto ``capture``. A cancelled stream
    never reaches the terminal chunk, so ``capture`` stays ``None``."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is not None:
        capture.input_tokens = prompt_tokens
    if completion_tokens is not None:
        capture.output_tokens = completion_tokens


def _require_content(text: str) -> str:
    if not text:
        raise ProviderResponseError("provider returned an empty response")
    return text


class AnyLLMProvider(GenerativeProvider):
    """Generative provider over any-llm. Translates failures to the
    ``AICoreProviderError`` hierarchy so consumers never see raw SDK errors."""

    def __init__(self, *, provider: str, model: str, **client_kwargs):
        self._provider = provider
        self.model = model
        self._client = AnyLLM.create(provider=provider, **client_kwargs)

    def completion(self, prompt, *, system=None, history=None, **kwargs) -> str:
        messages = build_messages(prompt, system=system, history=history)
        with _translate_errors():
            resp = self._client.completion(
                model=self.model, messages=messages, stream=False, **kwargs
            )
        return _require_content(_message_text(resp))

    async def acompletion(self, prompt, *, system=None, history=None, **kwargs) -> str:
        messages = build_messages(prompt, system=system, history=history)
        with _translate_errors():
            resp = await self._client.acompletion(
                model=self.model, messages=messages, stream=False, **kwargs
            )
        return _require_content(_message_text(resp))

    def stream(
        self,
        prompt,
        *,
        system=None,
        history=None,
        usage_capture: UsageCapture | None = None,
        **kwargs,
    ) -> Iterator[str]:
        messages = build_messages(prompt, system=system, history=history)
        with _translate_errors():
            for chunk in self._client.completion(
                model=self.model, messages=messages, stream=True, **kwargs
            ):
                if usage_capture is not None:
                    _fill_usage(usage_capture, chunk)
                text = _delta_text(chunk)
                if text:
                    yield text

    async def astream(
        self,
        prompt,
        *,
        system=None,
        history=None,
        usage_capture: UsageCapture | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        messages = build_messages(prompt, system=system, history=history)
        with _translate_errors():
            stream = await self._client.acompletion(
                model=self.model, messages=messages, stream=True, **kwargs
            )
            async for chunk in stream:
                if usage_capture is not None:
                    _fill_usage(usage_capture, chunk)
                text = _delta_text(chunk)
                if text:
                    yield text

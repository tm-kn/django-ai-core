# Embedding Package Clean Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split embedding into its own top-level package (`django_ai_core.embedding`), backed by a neutral shared kernel, with a concrete any-llm embedding provider — so embedding and generative are peers that never import each other.

**Architecture:** Hoist the provider-agnostic plumbing (role resolution, AI_CORE settings reader, `UsageCapture`, any-llm error/usage glue) out of `generative/` into neutral flat modules at the package root, beside the existing `exceptions.py`. `generative/` and `embedding/` each depend inward on that kernel and on the any-llm SDK — never on each other. Cross-domain orchestration (if ever needed) sits above both. This is a **refactor**: the existing test suite is the safety net; each task ends with the full suite green.

**Tech Stack:** Python 3.13, Django 5.2, `any-llm-sdk`, pytest / pytest-django.

## Global Constraints

- No back-compat shims. This is a young internal library with no external consumers of these paths; update imports at call sites directly (matches the shim-deletion done earlier this session).
- `generative/` MUST NOT import from `embedding/` and vice versa. Both import only from the neutral kernel modules (`django_ai_core.{exceptions,usage,settings,resolve,anyllm}`) and the any-llm SDK.
- The kernel modules MUST NOT import any domain ABC (`GenerativeProvider`/`EmbeddingProvider`) — dependencies point domain → kernel only.
- Consumers only ever see `AICoreProviderError` subclasses — never raw `AnyLLMError` or vendor SDK errors.
- Run the full suite with `uv run pytest -q` (there is no `python` on PATH; use `uv run`).
- Preserve exact behavior. Pure code moves must not change logic; only new code (Task 7) is TDD-first.

---

## File Structure

**New neutral kernel modules (flat at `src/django_ai_core/`, beside `exceptions.py`):**
- `usage.py` — `UsageCapture` dataclass (moved from `generative/providers/base.py`).
- `settings.py` — `get_ai_core_setting`, `get_generative_models`, `get_embedding_models` (moved from `generative/settings.py`).
- `resolve.py` — generic `resolve_provider(name, *, base, models, models_key, expect=None)` (extracted from `generative/resolve.py`'s `_resolve`).
- `anyllm.py` — shared any-llm glue: env flag, `_EXCEPTION_MAP`, `translate_error`, `translate_errors`, `fill_usage` (moved from `generative/providers/anyllm.py`).

**Generative (rewired onto kernel):**
- `generative/providers/base.py` — `GenerativeProvider` only; import `UsageCapture` from kernel.
- `generative/providers/anyllm.py` — `AnyLLMProvider` + completion-only helpers (`build_messages`, `_message_text`, `_delta_text`, `_require_content`); import glue from kernel.
- `generative/resolve.py` — thin `resolve_generative_provider` wrapper over kernel.
- `generative/service.py` — import `UsageCapture` from kernel.
- `generative/settings.py` — **deleted**.
- `generative/__init__.py`, `generative/providers/__init__.py` — drop embedding re-exports.

**New embedding package (`src/django_ai_core/embedding/`):**
- `__init__.py` — public API.
- `providers/__init__.py`, `providers/base.py` — `EmbeddingProvider` ABC (moved).
- `providers/anyllm.py` — `AnyLLMEmbeddingProvider` (new).
- `resolve.py` — `resolve_embedding_provider`.

**Tests:**
- New: `tests/unit/test_anyllm_glue.py`, `tests/unit/embedding/` package.
- Moved fakes: shared `FakeEmbeddingProvider` follows the ABC into an embedding test fixture module.
- Updated imports in `tests/unit/generative/*`.

---

### Task 1: Kernel — `UsageCapture`

**Files:**
- Create: `src/django_ai_core/usage.py`
- Modify: `src/django_ai_core/generative/providers/base.py` (remove `UsageCapture`, import from kernel)
- Modify: `src/django_ai_core/generative/service.py:10` (import from kernel)
- Modify: `src/django_ai_core/generative/providers/anyllm.py:58` (import from kernel)

**Interfaces:**
- Produces: `django_ai_core.usage.UsageCapture` — dataclass, fields `input_tokens: int | None = None`, `output_tokens: int | None = None`.

- [ ] **Step 1: Create the kernel module**

```python
# src/django_ai_core/usage.py
"""Provider-agnostic token-usage sink, shared by every provider layer."""

from dataclasses import dataclass


@dataclass
class UsageCapture:
    """Best-effort token-usage sink passed into a provider stream.

    A provider populates whatever usage it can read locally, synchronously
    (no ``await``), at every exit path. Fields stay ``None`` when unavailable —
    e.g. a stream cancelled before its terminal usage frame arrived. The
    consuming service reads this in its own ``finally`` for lifecycle logging.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
```

- [ ] **Step 2: Remove `UsageCapture` from `generative/providers/base.py`**

Delete the `UsageCapture` dataclass and its `@dataclass`/`from dataclasses import dataclass` usage from `generative/providers/base.py`. Add at the top of that file:

```python
from django_ai_core.usage import UsageCapture
```

Keep `UsageCapture` referenced in the `GenerativeProvider.stream`/`astream` annotations (now resolved via the import). Remove the now-unused `from dataclasses import dataclass` if nothing else uses it.

- [ ] **Step 3: Update `service.py` import**

In `generative/service.py`, replace:

```python
from .providers.base import UsageCapture
```

with:

```python
from django_ai_core.usage import UsageCapture
```

- [ ] **Step 4: Update `anyllm.py` import**

In `generative/providers/anyllm.py:58`, change:

```python
from .base import GenerativeProvider, UsageCapture
```

to:

```python
from django_ai_core.usage import UsageCapture

from .base import GenerativeProvider
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (same count as before this task).

- [ ] **Step 6: Commit**

```bash
git add src/django_ai_core/usage.py src/django_ai_core/generative
git commit -m "refactor(generative): move UsageCapture to neutral kernel module"
```

---

### Task 2: Kernel — settings reader

**Files:**
- Create: `src/django_ai_core/settings.py` (move content from `generative/settings.py`)
- Delete: `src/django_ai_core/generative/settings.py`
- Modify: `src/django_ai_core/generative/resolve.py:14` (import from kernel)
- Modify: `tests/unit/generative/test_settings.py` (update import path)

**Interfaces:**
- Produces: `django_ai_core.settings.get_ai_core_setting(key) -> Any`, `get_generative_models() -> dict`, `get_embedding_models() -> dict`.

- [ ] **Step 1: Create the kernel module**

Create `src/django_ai_core/settings.py` with the exact current content of `src/django_ai_core/generative/settings.py` (the `get_ai_core_setting`, `_get_models_dict`, `get_generative_models`, `get_embedding_models` definitions — unchanged).

- [ ] **Step 2: Delete the old module**

```bash
git rm src/django_ai_core/generative/settings.py
```

- [ ] **Step 3: Repoint `generative/resolve.py`**

In `generative/resolve.py`, change:

```python
from .settings import get_embedding_models, get_generative_models
```

to:

```python
from django_ai_core.settings import get_embedding_models, get_generative_models
```

(This import is temporary — Task 4 removes the embedding half from generative.)

- [ ] **Step 4: Repoint the settings test**

In `tests/unit/generative/test_settings.py`, change any `from django_ai_core.generative.settings import ...` to `from django_ai_core.settings import ...`. Consider moving the file to `tests/unit/test_settings_reader.py` since it now tests a kernel module; if moved, create `tests/unit/__init__.py` if absent.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A src/django_ai_core tests/unit
git commit -m "refactor: move AI_CORE settings reader to neutral kernel"
```

---

### Task 3: Kernel — generic role resolution

**Files:**
- Create: `src/django_ai_core/resolve.py` (generic `resolve_provider`)
- Modify: `src/django_ai_core/generative/resolve.py` (thin wrapper over kernel)

**Interfaces:**
- Produces: `django_ai_core.resolve.resolve_provider(name, *, base: type, models: dict, models_key: str, expect: type | None = None)` — returns a validated provider instance; raises `django.core.exceptions.ImproperlyConfigured` on misconfiguration.
- Consumes (later tasks): `generative/resolve.py` and `embedding/resolve.py` call `resolve_provider`.

- [ ] **Step 1: Create the kernel resolver**

```python
# src/django_ai_core/resolve.py
"""Generic role-name → provider-instance resolution.

Domain packages wrap this with their own ABC and settings key. Duck-typed on
``base=`` so the kernel never imports a domain ABC.

Each call returns a *fresh* instance — no caching at this layer. Providers are
responsible for their own lazy SDK client construction if they need cheap
repeated instantiation.
"""

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string


def resolve_provider(
    name: str,
    *,
    base: type,
    models: dict,
    models_key: str,
    expect: type | None = None,
):
    """Resolve ``name`` from ``models`` (an ``AI_CORE[models_key]`` map) to an
    instance of a ``base`` subclass. ``expect`` optionally narrows the accepted
    concrete type."""
    label = f"AI_CORE['{models_key}']"
    if name not in models:
        raise ImproperlyConfigured(
            f"Role '{name}' not configured in {label}. "
            f"Available roles: {sorted(models)!r}."
        )
    spec = models[name]
    if not isinstance(spec, dict) or "provider" not in spec:
        raise ImproperlyConfigured(
            f"{label}['{name}'] missing 'provider' key. "
            "Expected {'provider': 'dotted.path', 'params': {...}}."
        )
    provider_path = spec["provider"]
    params = spec.get("params", {}) or {}

    try:
        cls = import_string(provider_path)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"Cannot import provider '{provider_path}' for role '{name}': {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ImproperlyConfigured(
            f"Role '{name}' resolves to {cls!r}, expected subclass of {base.__name__}."
        )

    try:
        instance = cls(**params)
    except TypeError as exc:
        raise ImproperlyConfigured(
            f"Cannot instantiate '{provider_path}' for role '{name}' with "
            f"params={params!r}: {exc}"
        ) from exc

    if expect is not None and not isinstance(instance, expect):
        raise ImproperlyConfigured(
            f"Role '{name}' resolves to {type(instance).__name__}, "
            f"expected {expect.__name__}."
        )

    return instance
```

- [ ] **Step 2: Reduce `generative/resolve.py` to a thin wrapper**

Replace the whole file with:

```python
# src/django_ai_core/generative/resolve.py
"""Resolve generative role names to instantiated provider objects."""

from typing import TypeVar

from django_ai_core.resolve import resolve_provider
from django_ai_core.settings import get_generative_models

from .providers import GenerativeProvider

G = TypeVar("G", bound=GenerativeProvider)


def resolve_generative_provider(
    name: str,
    *,
    expect: type[G] | None = None,
) -> GenerativeProvider:
    """Resolve a role from ``AI_CORE['GENERATIVE_MODELS']`` to an instance."""
    return resolve_provider(
        name,
        base=GenerativeProvider,
        models=get_generative_models(),
        models_key="GENERATIVE_MODELS",
        expect=expect,
    )
```

Note: `resolve_embedding_provider` disappears from generative here — Task 5 recreates it under `embedding/`. Any test importing it from `generative.resolve` is repaired in Task 5. To keep the suite green *in this task*, temporarily keep an embedding wrapper too:

```python
from django_ai_core.settings import get_embedding_models
from .providers import EmbeddingProvider

E = TypeVar("E", bound=EmbeddingProvider)


def resolve_embedding_provider(name, *, expect: type[E] | None = None) -> EmbeddingProvider:
    return resolve_provider(
        name,
        base=EmbeddingProvider,
        models=get_embedding_models(),
        models_key="EMBEDDING_MODELS",
        expect=expect,
    )
```

(This temporary wrapper is removed in Task 5 once `embedding/` owns it.)

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/django_ai_core/resolve.py src/django_ai_core/generative/resolve.py
git commit -m "refactor: extract generic resolve_provider into neutral kernel"
```

---

### Task 4: Kernel — shared any-llm glue

**Files:**
- Create: `src/django_ai_core/anyllm.py` (env flag, `_EXCEPTION_MAP`, `translate_error`, `translate_errors`, `fill_usage`)
- Modify: `src/django_ai_core/generative/providers/anyllm.py` (import glue from kernel; keep completion helpers)
- Create: `tests/unit/test_anyllm_glue.py`
- Modify: `tests/unit/generative/providers/test_anyllm.py` (repoint glue imports)

**Interfaces:**
- Produces:
  - `django_ai_core.anyllm.translate_error(exc: Exception) -> AICoreProviderError`
  - `django_ai_core.anyllm.translate_errors()` — context manager re-raising raw failures as `AICoreProviderError`
  - `django_ai_core.anyllm.fill_usage(capture: UsageCapture, chunk: object) -> None`
- Consumes: `django_ai_core.usage.UsageCapture`, `django_ai_core.exceptions.*`.

- [ ] **Step 1: Create the kernel glue module**

```python
# src/django_ai_core/anyllm.py
"""Shared any-llm adapter glue: error translation + usage extraction.

Domain-neutral — used by every concrete any-llm-backed provider (generative
and embedding). No completion- or embedding-specific logic lives here. We
enable any-llm's unified exceptions so it normalises every vendor's failures
into its ``AnyLLMError`` hierarchy, which we map to django-ai-core's own
``AICoreProviderError`` types. Consumers never see raw vendor errors or
``AnyLLMError``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# any-llm only converts vendor SDK errors into its unified ``AnyLLMError``
# hierarchy when this flag is set (otherwise it re-raises the raw vendor error
# with a deprecation warning). We depend on the unified types, so enable it by
# default — ``setdefault`` leaves an explicit consumer choice untouched. Must be
# set before any call; any-llm reads the env var at raise time.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

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

from .exceptions import (
    AICoreProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)
from .usage import UsageCapture

# Maps any-llm's unified exceptions to our semantic ``AICoreProviderError`` types.
# any-llm's ``ProviderError`` is a catch-all (transport failures, 5xx,
# unclassified) we can't split without sniffing SDK class names, so it maps whole
# to ``ProviderUnexpectedError`` — as does anything unmatched, via the fallback.
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


def translate_error(exc: Exception) -> AICoreProviderError:
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
def translate_errors():
    """Re-raise any raw provider failure as a semantic ``AICoreProviderError``."""
    try:
        yield
    except AICoreProviderError:
        raise
    except Exception as exc:
        raise translate_error(exc) from exc


def fill_usage(capture: UsageCapture, chunk: object) -> None:
    """Copy any usage on this chunk into ``capture`` (best-effort, no raise).

    any-llm sets ``.usage`` (``prompt_tokens`` / ``completion_tokens``) only on
    the terminal chunk of a clean finish. A cancelled stream never reaches it, so
    ``capture`` stays ``None``."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is not None:
        capture.input_tokens = prompt_tokens
    if completion_tokens is not None:
        capture.output_tokens = completion_tokens
```

- [ ] **Step 2: Strip glue from `generative/providers/anyllm.py`**

Remove from that file: the `os.environ.setdefault(...)` line, the `from any_llm.exceptions import (...)` block, the `from ...exceptions import (...)` block, `_EXCEPTION_MAP`, `_translate`, `_translate_errors`, and `_fill_usage`. Keep `AnyLLM`, `build_messages`, `_delta_text`, `_message_text`, `_require_content`, and the `AnyLLMProvider` class. Add near the top:

```python
from django_ai_core.anyllm import fill_usage, translate_errors
from django_ai_core.usage import UsageCapture
```

Then update in-file call sites: `_translate_errors()` → `translate_errors()`, `_fill_usage(...)` → `fill_usage(...)`. Keep `ProviderResponseError` imported from `...exceptions` (still used by `_require_content`):

```python
from ...exceptions import ProviderResponseError
```

- [ ] **Step 3: Write the failing kernel-glue test**

```python
# tests/unit/test_anyllm_glue.py
from types import SimpleNamespace

import pytest
from any_llm.exceptions import ProviderError, RateLimitError

from django_ai_core.anyllm import fill_usage, translate_error, translate_errors
from django_ai_core.exceptions import (
    AICoreProviderError,
    ProviderRateLimitError,
    ProviderUnexpectedError,
)
from django_ai_core.usage import UsageCapture


def test_translate_maps_rate_limit():
    assert isinstance(translate_error(RateLimitError("x")), ProviderRateLimitError)


def test_translate_catch_all_maps_to_unexpected():
    assert isinstance(translate_error(ProviderError("x")), ProviderUnexpectedError)


def test_translate_passes_through_semantic():
    err = ProviderRateLimitError("keep me")
    assert translate_error(err) is err


def test_translate_errors_ctx_wraps_raw():
    with pytest.raises(AICoreProviderError):
        with translate_errors():
            raise RateLimitError("boom")


def test_fill_usage_reads_terminal_chunk():
    cap = UsageCapture()
    chunk = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7))
    fill_usage(cap, chunk)
    assert (cap.input_tokens, cap.output_tokens) == (3, 7)


def test_fill_usage_no_usage_is_noop():
    cap = UsageCapture()
    fill_usage(cap, SimpleNamespace())
    assert (cap.input_tokens, cap.output_tokens) == (None, None)
```

- [ ] **Step 4: Run the glue test**

Run: `uv run pytest tests/unit/test_anyllm_glue.py -v`
Expected: PASS (glue module created in Step 1).

- [ ] **Step 5: Repoint the existing any-llm provider test**

In `tests/unit/generative/providers/test_anyllm.py`, change glue imports from `django_ai_core.generative.providers.anyllm` (`_translate`, and any `_translate_errors`/`_fill_usage`) to `django_ai_core.anyllm` (`translate_error`, `translate_errors`, `fill_usage`). Keep `AnyLLMProvider`, `_delta_text`, `_message_text` imported from `django_ai_core.generative.providers.anyllm`. Update any references to the renamed symbols in the test body.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/django_ai_core/anyllm.py src/django_ai_core/generative/providers/anyllm.py tests/unit
git commit -m "refactor: extract shared any-llm glue into neutral kernel module"
```

---

### Task 5: Embedding package — ABC, resolver, public API

**Files:**
- Create: `src/django_ai_core/embedding/__init__.py`
- Create: `src/django_ai_core/embedding/providers/__init__.py`
- Create: `src/django_ai_core/embedding/providers/base.py`
- Create: `src/django_ai_core/embedding/resolve.py`
- Modify: `src/django_ai_core/generative/providers/base.py` (remove `EmbeddingProvider`)
- Modify: `src/django_ai_core/generative/providers/__init__.py` (export only `GenerativeProvider`)
- Modify: `src/django_ai_core/generative/resolve.py` (remove temporary embedding wrapper)
- Modify: `src/django_ai_core/generative/__init__.py` (drop `EmbeddingProvider`, `resolve_embedding_provider`)
- Create: `tests/unit/embedding/__init__.py`, `tests/unit/embedding/_fakes.py`, `tests/unit/embedding/test_resolve.py`
- Modify: `tests/unit/generative/_fakes.py`, `tests/unit/generative/conftest.py`, `tests/unit/generative/test_resolve.py`, `tests/unit/generative/test_abc.py` (drop embedding-side pieces)

**Interfaces:**
- Produces:
  - `django_ai_core.embedding.EmbeddingProvider` — ABC with `embedding(self, input, **kwargs)` and `async aembedding(self, input, **kwargs)`.
  - `django_ai_core.embedding.resolve_embedding_provider(name, *, expect=None) -> EmbeddingProvider`.
- Consumes: `django_ai_core.resolve.resolve_provider`, `django_ai_core.settings.get_embedding_models`.

- [ ] **Step 1: Create the embedding ABC**

```python
# src/django_ai_core/embedding/providers/base.py
"""Provider abstract class for the embedding module."""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Embed inputs into vector representations."""

    #: Human-readable model identifier for audit/lifecycle logging. Providers
    #: set it; ``None`` when the provider doesn't track one.
    model: str | None = None

    @abstractmethod
    def embedding(self, input: Any, **kwargs: Any) -> Any:
        """Synchronous embedding."""

    @abstractmethod
    async def aembedding(self, input: Any, **kwargs: Any) -> Any:
        """Async embedding."""
```

```python
# src/django_ai_core/embedding/providers/__init__.py
from .base import EmbeddingProvider

__all__ = ["EmbeddingProvider"]
```

- [ ] **Step 2: Create the embedding resolver**

```python
# src/django_ai_core/embedding/resolve.py
"""Resolve embedding role names to instantiated provider objects."""

from typing import TypeVar

from django_ai_core.resolve import resolve_provider
from django_ai_core.settings import get_embedding_models

from .providers import EmbeddingProvider

E = TypeVar("E", bound=EmbeddingProvider)


def resolve_embedding_provider(
    name: str,
    *,
    expect: type[E] | None = None,
) -> EmbeddingProvider:
    """Resolve a role from ``AI_CORE['EMBEDDING_MODELS']`` to an instance."""
    return resolve_provider(
        name,
        base=EmbeddingProvider,
        models=get_embedding_models(),
        models_key="EMBEDDING_MODELS",
        expect=expect,
    )
```

- [ ] **Step 3: Create the embedding package public API**

```python
# src/django_ai_core/embedding/__init__.py
"""Embedding module.

Public API:
- ``EmbeddingProvider`` — abstract class to subclass for a custom provider.
- ``resolve_embedding_provider`` — resolve a configured embedding role to an
  instance.
"""

from .providers import EmbeddingProvider
from .resolve import resolve_embedding_provider

__all__ = ["EmbeddingProvider", "resolve_embedding_provider"]
```

- [ ] **Step 4: Remove `EmbeddingProvider` from generative**

In `generative/providers/base.py`, delete the `EmbeddingProvider` class (keep `GenerativeProvider`). In `generative/providers/__init__.py`:

```python
from .base import GenerativeProvider

__all__ = ["GenerativeProvider"]
```

In `generative/resolve.py`, delete the temporary `resolve_embedding_provider`, its `E` TypeVar, `EmbeddingProvider` import, and `get_embedding_models` import (added in Task 3) — leaving only the generative wrapper.

In `generative/__init__.py`, remove `EmbeddingProvider` and `resolve_embedding_provider` from both the imports and `__all__`.

- [ ] **Step 5: Move embedding test fakes**

In `tests/unit/generative/_fakes.py`, delete `FakeEmbeddingProvider` and change its import line to `from django_ai_core.generative.providers import GenerativeProvider`. In `tests/unit/generative/conftest.py`, drop `FakeEmbeddingProvider` from the imports and `__all__`.

Create `tests/unit/embedding/__init__.py` (empty) and:

```python
# tests/unit/embedding/_fakes.py
"""Test doubles for embedding provider resolution."""

from django_ai_core.embedding import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, model: str = "fake-embed", **extra):
        self.model = model
        self.extra = extra

    def embedding(self, input, **kwargs):
        return [[0.0] for _ in input]

    async def aembedding(self, input, **kwargs):
        return self.embedding(input, **kwargs)


class NotAProvider:
    """Used to test the abstract-class-subclass check."""

    def __init__(self, **kwargs):
        pass
```

- [ ] **Step 6: Write the embedding resolve test**

```python
# tests/unit/embedding/test_resolve.py
import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_core.embedding import resolve_embedding_provider

from ._fakes import FakeEmbeddingProvider

_EMB_FAKE = "embedding._fakes.FakeEmbeddingProvider"


@override_settings(
    AI_CORE={"EMBEDDING_MODELS": {"default": {"provider": _EMB_FAKE, "params": {"model": "m"}}}}
)
def test_resolves_configured_role():
    provider = resolve_embedding_provider("default")
    assert isinstance(provider, FakeEmbeddingProvider)
    assert provider.model == "m"


@override_settings(AI_CORE={"EMBEDDING_MODELS": {}})
def test_unknown_role_raises():
    with pytest.raises(ImproperlyConfigured):
        resolve_embedding_provider("missing")
```

Note: the dotted path `embedding._fakes.FakeEmbeddingProvider` assumes `tests/unit/embedding` is importable as `embedding` (mirrors the existing `generative._fakes...` convention in `test_resolve.py`). Verify the test rootdir/`sys.path` makes `embedding` importable exactly as `generative` is today; if the existing suite reaches `generative._fakes` via a `pythonpath = tests/unit` (or similar) pytest setting, the same setting covers `embedding`.

- [ ] **Step 7: Repair the generative resolve test**

In `tests/unit/generative/test_resolve.py`, remove `resolve_embedding_provider` from the import, drop `FakeEmbeddingProvider` from the `._fakes` import, remove `_EMB_FAKE`, and delete any test cases exercising embedding resolution (they now live in `tests/unit/embedding/test_resolve.py`). Do the same in `tests/unit/generative/test_abc.py` for any `EmbeddingProvider` ABC assertions — move them to a new `tests/unit/embedding/test_abc.py` if present.

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add -A src/django_ai_core tests/unit
git commit -m "feat(embedding): split EmbeddingProvider into its own top-level package"
```

---

### Task 6: Embedding package — `AnyLLMEmbeddingProvider`

**Files:**
- Create: `src/django_ai_core/embedding/providers/anyllm.py`
- Create: `tests/unit/embedding/providers/__init__.py`, `tests/unit/embedding/providers/test_anyllm.py`

**Interfaces:**
- Produces: `django_ai_core.embedding.providers.anyllm.AnyLLMEmbeddingProvider(*, provider: str, model: str, **client_kwargs)` — implements `EmbeddingProvider`; `embedding`/`aembedding` return any-llm's raw `CreateEmbeddingResponse` (normalisation deferred, mirroring the generative provider's staged approach). Failures surface as `AICoreProviderError`.
- Consumes: `any_llm.embedding` / `any_llm.aembedding` (module-level), `django_ai_core.anyllm.translate_errors`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/providers/test_anyllm.py
import asyncio

import pytest

import django_ai_core.embedding.providers.anyllm as mod
from django_ai_core.embedding import EmbeddingProvider
from django_ai_core.embedding.providers.anyllm import AnyLLMEmbeddingProvider
from django_ai_core.exceptions import ProviderRateLimitError


def test_is_embedding_provider():
    assert issubclass(AnyLLMEmbeddingProvider, EmbeddingProvider)


def test_embedding_calls_any_llm_with_provider_and_model(monkeypatch):
    seen = {}

    def fake_embedding(model, inputs, *, provider, **kwargs):
        seen.update(model=model, inputs=inputs, provider=provider, kwargs=kwargs)
        return "RESP"

    monkeypatch.setattr(mod.any_llm, "embedding", fake_embedding)
    p = AnyLLMEmbeddingProvider(provider="openai", model="text-embed", api_key="k")
    out = p.embedding(["a", "b"])
    assert out == "RESP"
    assert seen["model"] == "text-embed"
    assert seen["inputs"] == ["a", "b"]
    assert seen["provider"] == "openai"
    assert seen["kwargs"]["api_key"] == "k"


def test_embedding_translates_errors(monkeypatch):
    from any_llm.exceptions import RateLimitError

    def boom(*a, **k):
        raise RateLimitError("slow down")

    monkeypatch.setattr(mod.any_llm, "embedding", boom)
    p = AnyLLMEmbeddingProvider(provider="openai", model="m")
    with pytest.raises(ProviderRateLimitError):
        p.embedding(["a"])


def test_aembedding_calls_any_llm(monkeypatch):
    async def fake_aembedding(model, inputs, *, provider, **kwargs):
        return "ARESP"

    monkeypatch.setattr(mod.any_llm, "aembedding", fake_aembedding)
    p = AnyLLMEmbeddingProvider(provider="openai", model="m")
    assert asyncio.run(p.aembedding(["a"])) == "ARESP"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/embedding/providers/test_anyllm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'django_ai_core.embedding.providers.anyllm'`.

- [ ] **Step 3: Implement the provider**

```python
# src/django_ai_core/embedding/providers/anyllm.py
"""Concrete ``EmbeddingProvider`` backed by any-llm.

Shipped with django-ai-core as a batteries-included provider. Error handling
is provider-agnostic via the shared any-llm glue: consumers only ever see
``AICoreProviderError`` types. Talking to a specific vendor still needs that
vendor's client installed via an optional extra, e.g. ``[openai]``.

Reference by dotted path from ``AI_CORE['EMBEDDING_MODELS']``::

    "provider": "django_ai_core.embedding.providers.anyllm.AnyLLMEmbeddingProvider",
    "params": {"provider": "openai", "model": "text-embedding-3-small"},
"""

from __future__ import annotations

from typing import Any

import any_llm

from django_ai_core.anyllm import translate_errors

from .base import EmbeddingProvider


class AnyLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider over any-llm. Translates failures to the
    ``AICoreProviderError`` hierarchy so consumers never see raw SDK errors.

    Returns any-llm's raw ``CreateEmbeddingResponse`` for now; a normalised
    response type can land later without changing this contract's call shape.
    """

    def __init__(self, *, provider: str, model: str, **client_kwargs: Any):
        self._provider = provider
        self.model = model
        self._client_kwargs = client_kwargs

    def embedding(self, input: Any, **kwargs: Any) -> Any:
        with translate_errors():
            return any_llm.embedding(
                self.model, input, provider=self._provider, **self._client_kwargs, **kwargs
            )

    async def aembedding(self, input: Any, **kwargs: Any) -> Any:
        with translate_errors():
            return await any_llm.aembedding(
                self.model, input, provider=self._provider, **self._client_kwargs, **kwargs
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/embedding/providers/test_anyllm.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/django_ai_core/embedding/providers/anyllm.py tests/unit/embedding
git commit -m "feat(embedding): add any-llm-backed AnyLLMEmbeddingProvider"
```

---

### Task 7: Docs + import-hygiene sweep

**Files:**
- Modify: generative module docs (wherever the module is documented — search `docs/` and any module docstrings referencing the old embedding location)
- Verify: no stray imports of moved symbols remain

- [ ] **Step 1: Grep for stale references**

Run:
```bash
grep -rn "generative.settings\|generative.providers.anyllm import.*_translate\|generative import.*EmbeddingProvider\|generative.resolve import.*embedding\|generative.providers.base import UsageCapture" src/ tests/ docs/
```
Expected: no matches. Fix any that appear.

- [ ] **Step 2: Update module docs**

Find generative-module docs (`grep -rln "EmbeddingProvider\|resolve_embedding_provider" docs/`) and update them to point at `django_ai_core.embedding`, plus document `AnyLLMEmbeddingProvider` and the `EMBEDDING_MODELS` role map. Add a short embedding section mirroring the generative one.

- [ ] **Step 3: Run the full suite + linters**

Run: `uv run pytest -q`
Expected: PASS.
Run the project's linters (e.g. `uv run ruff check .` / `uv run ruff format --check .` if configured).
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: document the embedding package and any-llm embedding provider"
```

---

## Follow-up (separate plan): rewire `contrib/index` onto `EmbeddingProvider`

Out of scope here — it changes a live subsystem with its own consumers. `contrib/index/embedding.py`'s `CoreEmbeddingTransformer` takes `llm_service: LLMService` and calls `.embedding(text).data[0].embedding`. `AnyLLMEmbeddingProvider.embedding` returns the same `CreateEmbeddingResponse` shape, so the migration is mechanical but ripples through `CachedEmbeddingTransformer`, `tests/testapp/indexes.py`, and index query/vector tests. Write that as its own plan once this lands.

## Self-Review

- **Spec coverage:** kernel usage (T1), settings (T2), resolve (T3), any-llm glue (T4); embedding ABC/resolver/API (T5); concrete provider (T6); docs/hygiene (T7); index reconcile explicitly deferred. All four "full clean separation" kernel pieces + embedding package + concrete provider covered.
- **Dependency direction:** kernel imports no domain ABC (T3 resolver is generic via `base=`); `embedding/` and `generative/` import only kernel + SDK, never each other (T5 removes the last generative→embedding coupling). ✓
- **Type consistency:** `resolve_provider(name, *, base, models, models_key, expect=None)` defined T3, consumed identically in generative (T3) and embedding (T5) wrappers. `translate_errors`/`fill_usage`/`translate_error` named T4, consumed T4/T6. `UsageCapture` fields consistent T1/T4. `AnyLLMEmbeddingProvider(*, provider, model, **client_kwargs)` defined + tested T6. ✓
- **Placeholders:** none — every code step shows full content.

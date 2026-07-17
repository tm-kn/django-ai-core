"""Resolve generative role names to instantiated provider objects."""

from typing import TypeVar

from django_ai_core.resolve import resolve_provider
from django_ai_core.settings import get_embedding_models, get_generative_models

from .providers import EmbeddingProvider, GenerativeProvider

G = TypeVar("G", bound=GenerativeProvider)
E = TypeVar("E", bound=EmbeddingProvider)


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


def resolve_embedding_provider(
    name: str, *, expect: type[E] | None = None
) -> EmbeddingProvider:
    return resolve_provider(
        name,
        base=EmbeddingProvider,
        models=get_embedding_models(),
        models_key="EMBEDDING_MODELS",
        expect=expect,
    )

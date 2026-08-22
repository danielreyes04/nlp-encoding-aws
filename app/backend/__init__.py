"""
backend
-------
Módulo de backend con la lógica de negocio, procesamiento NLP,
configuración y validación de esquemas Pydantic.
"""

from app.backend.config import (
    SPACY_MODEL,
    SPACY_FALLBACK_MODEL,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
)
from app.backend.nlp_pipeline import (
    clean_and_transform,
    lemmas_only,
    dependency_parse,
    named_entities,
    full_pipeline,
    encode_corpus,
    corpus_pipeline,
)
from app.backend.schemas import TextRequest, EncodingRequest

__all__ = [
    "SPACY_MODEL",
    "SPACY_FALLBACK_MODEL",
    "API_TITLE",
    "API_DESCRIPTION",
    "API_VERSION",
    "CORS_ORIGINS",
    "clean_and_transform",
    "lemmas_only",
    "dependency_parse",
    "named_entities",
    "full_pipeline",
    "encode_corpus",
    "corpus_pipeline",
    "TextRequest",
    "EncodingRequest",
]

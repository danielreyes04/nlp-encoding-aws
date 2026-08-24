"""
schemas.py
----------
Esquemas Pydantic compartidos por ambas APIs (EC2 y Lambda).

Cubre todas las validaciones de la sección 9 de la guía:
  - Ausencia de campos obligatorios            → 422
  - Valores null                               → 422
  - Tipos distintos a los declarados           → 422
  - Listas vacías                              → 422
  - Elementos no string                        → 422
  - Textos vacíos o sólo espacios             → 422
  - Batch en /visualize/dep                   → 422  (sólo acepta str)
  - Menos de 2 documentos en /vectorize       → 422
"""

from __future__ import annotations
from typing import Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_non_blank(value: str, field_name: str = "text") -> str:
    """Lanza ValueError si el string está vacío o es sólo espacios."""
    if not value or not value.strip():
        raise ValueError(f"'{field_name}' no puede estar vacío ni ser sólo espacios.")
    return value


def _require_non_blank_list(values: list[str], field_name: str = "text") -> list[str]:
    """Valida cada elemento de la lista: no vacío, no sólo espacios."""
    for i, v in enumerate(values):
        if not isinstance(v, str):
            raise ValueError(
                f"'{field_name}[{i}]' debe ser string, se recibió {type(v).__name__}."
            )
        if not v or not v.strip():
            raise ValueError(
                f"'{field_name}[{i}]' no puede estar vacío ni ser sólo espacios."
            )
    return values


# ---------------------------------------------------------------------------
# Endpoints /clean, /pos, /ner  — acepta string único O lista de strings
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    """
    Cuerpo de entrada para /api/v1/clean, /api/v1/pos y /api/v1/ner.
    'text' puede ser un string único o una lista de strings no vacíos.
    """
    text: Union[str, list[str]] = Field(
        ...,
        description="Texto único o lista de textos a procesar.",
        examples=["Mi gato come pescado", ["Texto uno", "Texto dos"]],
    )

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            raise ValueError("'text' no puede ser null.")
        if isinstance(v, str):
            return _require_non_blank(v, "text")
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("'text' no puede ser una lista vacía.")
            return _require_non_blank_list(v, "text")
        raise ValueError(
            f"'text' debe ser string o lista de strings, se recibió {type(v).__name__}."
        )


# ---------------------------------------------------------------------------
# Endpoint /visualize/dep  — sólo acepta string único (NO batch)
# ---------------------------------------------------------------------------

class DepRequest(BaseModel):
    """
    Cuerpo de entrada para /api/v1/visualize/dep.
    Sólo acepta un único string; el uso de lista es inválido (sección 9).
    """
    text: str = Field(
        ...,
        description="Texto único para visualizar el árbol de dependencias.",
        examples=["El caballo corre muy rápido"],
    )

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            raise ValueError("'text' no puede ser null.")
        if not isinstance(v, str):
            raise ValueError(
                f"'text' debe ser string (no se admite batch en /visualize/dep), "
                f"se recibió {type(v).__name__}."
            )
        return _require_non_blank(v, "text")


# ---------------------------------------------------------------------------
# Endpoint /vectorize  — lista de al menos 2 documentos
# ---------------------------------------------------------------------------

class VectorizeRequest(BaseModel):
    """
    Cuerpo de entrada para /api/v1/vectorize.
    'documents' debe ser una lista de al menos 2 strings no vacíos.
    """
    documents: list[str] = Field(
        ...,
        description="Colección de al menos 2 documentos a vectorizar.",
        examples=[["El gato come", "El perro corre", "Juan vive en Bogotá"]],
    )

    @field_validator("documents", mode="before")
    @classmethod
    def validate_documents(cls, v):
        if v is None:
            raise ValueError("'documents' no puede ser null.")
        if not isinstance(v, list):
            raise ValueError(
                f"'documents' debe ser lista de strings, se recibió {type(v).__name__}."
            )
        if len(v) == 0:
            raise ValueError("'documents' no puede ser una lista vacía.")
        if len(v) < 2:
            raise ValueError(
                "'documents' debe contener al menos 2 documentos para vectorizar."
            )
        return _require_non_blank_list(v, "documents")

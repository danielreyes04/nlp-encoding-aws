"""Esquemas Pydantic compartidos por ambas APIs (EC2 y Lambda),
asi los request/response bodies quedan identicos en las dos."""

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Mi gato, su gato y nuestro gato comen pescado"])


class EncodingRequest(BaseModel):
    corpus: list[str] = Field(
        ...,
        min_length=1,
        examples=[[
            "Mi gato, su gato y nuestro gato comen pescado",
            "Juan comio en Bogota",
            "El caballo corre muy rapido",
        ]],
    )
    method: str = Field(default="tfidf", description="'onehot' | 'bow' | 'tfidf'")

"""Esquemas Pydantic compartidos por ambas APIs (EC2 y Lambda),
asi los request/response bodies quedan identicos en las dos."""

from pydantic import BaseModel, Field

# pydantic con base model ayuda a verificar el json que llega este definido de la forma correcta, de lo contrario fastpai lo rechaza
class TextRequest(BaseModel):
    # flield hace que sea obligatorio que reciba ese parametro text, que sea str y que no este vacio
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

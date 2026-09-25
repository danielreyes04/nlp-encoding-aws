"""
nlp_pipeline.py
----------------
Núcleo de lógica NLP compartido por ambas APIs (EC2 y Lambda).
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from functools import lru_cache

import spacy
from spacy import displacy

from app.backend.config import SPACY_MODEL, SPACY_FALLBACK_MODEL

import logging
logger = logging.getLogger(__name__)

# para que se cargue solo una vez el modelo
@lru_cache(maxsize=1)
# carga del modelo
def get_nlp() -> "spacy.language.Language":
    try:
        return spacy.load(SPACY_MODEL)# modelo ya configurado en congig.py
    except OSError:
        logger.warning(
            "Modelo '%s' no disponible. Cargando fallback '%s'.",
            SPACY_MODEL,
            SPACY_FALLBACK_MODEL,
        )
        return spacy.load(SPACY_FALLBACK_MODEL)


# ---------------------------------------------------------------------------
# Tokenización compartida: usada tanto por clean_text() como por vectorize().
# Este es el ÚNICO lugar donde se decide qué es un "término resultante de
# la limpieza" (sección 4 del PDF) — antes había dos implementaciones
# distintas (una con texto, otra con lemas) que divergían entre sí.
# ---------------------------------------------------------------------------


# se centra la tokenizacion
def _clean_tokens(text: str) -> list[str]:
    """
    Tokeniza y limpia un documento, devolviendo la lista de términos
    (texto en minúsculas, sin puntuación, sin stopwords). Es el mismo
    proceso para /clean y para /vectorize, por contrato del PDF.
    """
    nlp = get_nlp()
    text_pre = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )
    doc = nlp(text_pre.lower())

    tokens = []
    for tok in doc:
        #Si es un stopword(el, la ,los,que, en, un, etc ...)
        #si es un espacio en blanco 
        # quita dobles espacios en blanco( mas como seguridad que no se colen espacios en blanco)
        # esos caracteres no los guarda
        if tok.is_stop or tok.is_space or not tok.text.strip():
            continue
        tokens.append(tok.text)
    return tokens


#elimina la lista y devuelve un texto plano(str)
def clean_text(text: str) -> str:
    """Limpia un único documento y devuelve el texto limpio como string."""
    return ' '.join(_clean_tokens(text))

#retorna una lista de strings donde cada elemento es un documento
def clean_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]


def pos_analysis(text: str) -> dict:
    nlp = get_nlp()
    doc = nlp(text)
    return {
        "tokens": [
            {"text": tok.text, "pos": tok.pos_, "lemma": tok.lemma_}
            for tok in doc
        ]
    }

#retorna un corpus con con cada documento como dict
def pos_analysis_batch(texts: list[str]) -> list[dict]:
    return [pos_analysis(t) for t in texts]

def ner_analysis(text: str) -> dict:
    nlp = get_nlp()
    doc = nlp(text)
    return {
        "entities": [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    }

#retorna un corpus con con cada documento como dict
def ner_analysis_batch(texts: list[str]) -> list[dict]:
    return [ner_analysis(t) for t in texts]

# genera una representacion grafica de  del analisisi de dependecias
def dependency_html(text: str) -> str:
    nlp = get_nlp()
    doc = nlp(text)
    # displacy es el modulo de visualizacion de spacy
    svg = displacy.render(doc, style="dep", jupyter=False, page=False)
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<title>Dependency Parse</title></head>"
        f"<body>{svg}</body></html>"
    )


def vectorize(documents: list[str]) -> dict:
    """
    Construye vocabulario y calcula One-Hot, Bag of Words y TF-IDF.
    """
    size_documents = len(documents)

    docs_tokens: list[list[str]] = [_clean_tokens(doc) for doc in documents]

    vocab_set: set[str] = set()
    for toks in docs_tokens:
        vocab_set.update(toks)
    vocabulary: list[str] = sorted(vocab_set)
    size_vocabulary = len(vocabulary)
    term_index: dict[str, int] = {term: i for i, term in enumerate(vocabulary)}

    doc_freqs: list[Counter] = [Counter(toks) for toks in docs_tokens]

    nt: dict[str, int] = {
        term: sum(1 for freq in doc_freqs if freq[term] > 0)
        for term in vocabulary
    }

    idf: dict[str, float] = {
        term: math.log((size_documents + 1) / (nt[term] + 1)) + 1
        for term in vocabulary
    }

    bag_of_words: list[list[int]] = [
        [freq[term] for term in vocabulary] for freq in doc_freqs
    ]

    one_hot: list[list[list[int]]] = []
    for toks in docs_tokens:
        matrix: list[list[int]] = []
        for tok in toks:
            if tok in term_index:
                vec = [0] * size_vocabulary
                vec[term_index[tok]] = 1
                matrix.append(vec)
        one_hot.append(matrix)

    tf_idf: list[list[float]] = []
    for freq in doc_freqs:
        row = [round(freq[term] * idf[term], 4) for term in vocabulary]
        tf_idf.append(row)

    return {
        "vocabulary": vocabulary,
        "bag_of_words": bag_of_words,
        "one_hot": one_hot,
        "tf_idf": tf_idf,
    }
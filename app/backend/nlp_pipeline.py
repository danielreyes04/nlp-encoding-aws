"""
nlp_pipeline.py
----------------
Núcleo de lógica NLP compartido por ambas APIs (EC2 y Lambda).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import spacy
from spacy import displacy

from app.backend.config import SPACY_MODEL, SPACY_FALLBACK_MODEL

import logging
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nlp() -> "spacy.language.Language":
    try:
        return spacy.load(SPACY_MODEL)
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

def _clean_tokens(text: str) -> list[str]:
    """
    Tokeniza y limpia un documento, devolviendo la lista de términos
    (texto en minúsculas, sin puntuación, sin stopwords). Es el mismo
    proceso para /clean y para /vectorize, por contrato del PDF.
    """
    nlp = get_nlp()
    text_pre = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    doc = nlp(text_pre.lower())

    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_space or not tok.text.strip():
            continue
        tokens.append(tok.text)
    return tokens


def clean_text(text: str) -> str:
    """Limpia un único documento y devuelve el texto limpio como string."""
    return ' '.join(_clean_tokens(text))


def clean_texts(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]


def pos_analysis(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {"text": tok.text, "pos": tok.pos_, "lemma": tok.lemma_.lower()}
        for tok in doc
        if not tok.is_space
    ]


def pos_analysis_batch(texts: list[str]) -> list[list[dict]]:
    return [pos_analysis(t) for t in texts]


def ner_analysis(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


def ner_analysis_batch(texts: list[str]) -> list[list[dict]]:
    return [ner_analysis(t) for t in texts]


def dependency_html(text: str) -> str:
    nlp = get_nlp()
    doc = nlp(text)
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
    CORRECCIÓN: ahora usa _clean_tokens (texto, no lemas) para que el
    vocabulario coincida exactamente con lo que produce /clean.
    """
    D = len(documents)

    docs_tokens: list[list[str]] = [_clean_tokens(doc) for doc in documents]

    vocab_set: set[str] = set()
    for toks in docs_tokens:
        vocab_set.update(toks)
    vocabulary: list[str] = sorted(vocab_set)
    V = len(vocabulary)
    term_index: dict[str, int] = {term: i for i, term in enumerate(vocabulary)}

    doc_freqs: list[Counter] = [Counter(toks) for toks in docs_tokens]

    nt: dict[str, int] = {
        term: sum(1 for freq in doc_freqs if freq[term] > 0)
        for term in vocabulary
    }

    idf: dict[str, float] = {
        term: math.log((D + 1) / (nt[term] + 1)) + 1
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
                vec = [0] * V
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
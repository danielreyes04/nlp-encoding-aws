"""
nlp_pipeline.py
----------------
Núcleo de lógica NLP compartido por ambas APIs (EC2 y Lambda).

Flujo de procesamiento:
    Corpus -> Limpieza -> POS/Lematización -> NER -> Dependencias -> Vectorización

Todos los resultados cumplen el contrato definido en la guía del laboratorio
(sección 8 – Perfil mínimo de interoperabilidad del evaluador).
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


# ---------------------------------------------------------------------------
# 1. Carga del modelo (una sola vez, cacheado)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_nlp() -> spacy.language.Language:
    """Carga el modelo spaCy con caché LRU y fallback automático."""
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
# 2. Limpieza de texto  →  POST /api/v1/clean
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Limpia un único documento:
    - Convierte a minúsculas.
    - Reemplaza signos de puntuación por espacios (actúan como separadores,
      no concatenan términos).
    - Elimina stopwords (Token.is_stop según es_core_news_sm/md).
    - Conserva letras acentuadas, ñ y dígitos.
    - Normaliza espacios en blanco.

    Retorna el texto limpio como string.
    """
    nlp = get_nlp()
    # Reemplazar signos de puntuación por espacios antes de procesar,
    # para que no concatenen términos adyacentes.
    # Se conservan mayúsculas para que spaCy lematice bien,
    # y se aplica .lower() solo al texto del token resultante.
    text_pre = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    doc = nlp(text_pre)

    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_space or not tok.text.strip():
            continue
        tokens.append(tok.text.lower())

    return ' '.join(tokens)


def clean_texts(texts: list[str]) -> list[str]:
    """Aplica clean_text a una lista de documentos y conserva el orden."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# 3. Análisis POS  →  POST /api/v1/pos
# ---------------------------------------------------------------------------

def pos_analysis(text: str) -> list[dict]:
    """
    Retorna, en orden, la lista de tokens del documento con:
      - text : texto original del token
      - pos  : categoría gramatical universal (UPOS)
      - lemma: lema en minúsculas
    Incluye todos los tokens (sin filtrar stopwords ni puntuación)
    para conservar el orden y la correspondencia con el texto original.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {
            "text": tok.text,
            "pos": tok.pos_,
            "lemma": tok.lemma_.lower(),
        }
        for tok in doc
        if not tok.is_space
    ]


def pos_analysis_batch(texts: list[str]) -> list[list[dict]]:
    """Aplica pos_analysis a una lista de documentos y conserva el orden."""
    return [pos_analysis(t) for t in texts]


# ---------------------------------------------------------------------------
# 4. Reconocimiento de entidades (NER)  →  POST /api/v1/ner
# ---------------------------------------------------------------------------

def ner_analysis(text: str) -> list[dict]:
    """
    Detecta entidades nombradas en un documento.
    Retorna lista de dicts con:
      - text : texto de la entidad
      - label: tipo de entidad (PER, LOC, ORG, …)
      - start: índice de carácter inicial (inclusivo) en el texto original
      - end  : índice de carácter final (exclusivo) en el texto original
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


def ner_analysis_batch(texts: list[str]) -> list[list[dict]]:
    """Aplica ner_analysis a una lista de documentos y conserva el orden."""
    return [ner_analysis(t) for t in texts]


# ---------------------------------------------------------------------------
# 5. Visualización de dependencias  →  POST /api/v1/visualize/dep
# ---------------------------------------------------------------------------

def dependency_html(text: str) -> str:
    """
    Genera un documento HTML completo que contiene el SVG de displaCy
    con el árbol de dependencias sintácticas del texto.
    Solo procesa un único documento por llamada.
    """
    nlp = get_nlp()
    doc = nlp(text)
    svg = displacy.render(doc, style="dep", jupyter=False, page=False)
    html = (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<title>Dependency Parse</title></head>"
        f"<body>{svg}</body></html>"
    )
    return html


# ---------------------------------------------------------------------------
# 6. Vectorización  →  POST /api/v1/vectorize
# ---------------------------------------------------------------------------

def _lemmatize_for_vocab(text: str) -> list[str]:
    """
    Aplica la misma limpieza que clean_text pero retorna la lista de
    lemas (en minúsculas) que se usarán para construir el vocabulario.

    Se procesa el texto ORIGINAL (sin bajar a minúsculas antes) para que
    spaCy tenga el contexto morfológico completo y lematice correctamente.
    El .lower() se aplica solo al lema resultante.
    """
    nlp = get_nlp()
    # Reemplazar puntuación por espacios (igual que clean_text) pero
    # conservar las mayúsculas para que spaCy lematice bien
    text_pre = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    doc = nlp(text_pre)
    return [
        tok.lemma_.lower()
        for tok in doc
        if not tok.is_stop and not tok.is_space and tok.text.strip()
    ]


def vectorize(documents: list[str]) -> dict:
    """
    Construye vocabulario y calcula One-Hot, Bag of Words y TF-IDF
    para una colección de al menos 2 documentos.

    Reglas (sección 4 de la guía):
    ─────────────────────────────
    • Vocabulario: lemas de la limpieza, en orden lexicográfico ascendente.
    • BoW: frecuencia absoluta del término en el documento.
    • One-Hot: por cada *ocurrencia* de un término se genera un vector
      binario de longitud |V| con un único 1 en la posición del término.
      → cada documento produce una matriz de shape (freq_total × |V|).
    • TF-IDF:
        tf(t, d)  = frecuencia absoluta de t en d
        idf(t)    = ln( (|D| + 1) / (nt + 1) ) + 1
        tfidf     = tf * idf, redondeado a 4 decimales.
        Sin normalización posterior.

    Retorna:
    {
      "vocabulary": [...],
      "bag_of_words": [[...], ...],          # N × |V|
      "one_hot": [[[...], ...], ...],        # lista de N matrices
      "tf_idf": [[...], ...]                 # N × |V|
    }
    """
    D = len(documents)  # número de documentos

    # 1. Lematizar cada documento → lista de lemas limpios por doc
    docs_lemmas: list[list[str]] = [_lemmatize_for_vocab(doc) for doc in documents]

    # 2. Construir vocabulario en orden lexicográfico
    vocab_set: set[str] = set()
    for lemmas in docs_lemmas:
        vocab_set.update(lemmas)
    vocabulary: list[str] = sorted(vocab_set)
    V = len(vocabulary)
    term_index: dict[str, int] = {term: i for i, term in enumerate(vocabulary)}

    # 3. Frecuencias absolutas por documento (para BoW y TF-IDF)
    doc_freqs: list[Counter] = [Counter(lemmas) for lemmas in docs_lemmas]

    # 4. nt: número de documentos que contienen cada término
    nt: dict[str, int] = {}
    for term in vocabulary:
        nt[term] = sum(1 for freq in doc_freqs if freq[term] > 0)

    # 5. idf por término: ln((|D|+1)/(nt+1)) + 1
    idf: dict[str, float] = {
        term: math.log((D + 1) / (nt[term] + 1)) + 1
        for term in vocabulary
    }

    # 6. Bag of Words  →  N × |V|
    bag_of_words: list[list[int]] = []
    for freq in doc_freqs:
        row = [freq[term] for term in vocabulary]
        bag_of_words.append(row)

    # 7. One-Hot  →  lista de N matrices (freq_total_i × |V|)
    #    Cada fila representa UNA ocurrencia (aparición) del token en el doc.
    one_hot: list[list[list[int]]] = []
    for lemmas in docs_lemmas:
        matrix: list[list[int]] = []
        for lemma in lemmas:
            if lemma in term_index:
                vec = [0] * V
                vec[term_index[lemma]] = 1
                matrix.append(vec)
        one_hot.append(matrix)

    # 8. TF-IDF  →  N × |V|
    tf_idf: list[list[float]] = []
    for freq in doc_freqs:
        row = [
            round(freq[term] * idf[term], 4)
            for term in vocabulary
        ]
        tf_idf.append(row)

    return {
        "vocabulary": vocabulary,
        "bag_of_words": bag_of_words,
        "one_hot": one_hot,
        "tf_idf": tf_idf,
    }

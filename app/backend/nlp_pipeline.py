"""
nlp_pipeline.py
----------------
Este modulo contiene TODA la logica de negocio (el "core") que sigue
el flujo visto en clase:

    Corpus -> Limpieza -> Transformacion -> Etiquetado -> Codificacion

Las DOS apis (la que corre en EC2/Cloud9 y la que corre en Lambda)
importan este mismo modulo. Asi garantizamos que ambas sigan
exactamente el mismo flujo de datos: el modulo es la "fuente de
verdad" y las apis solo son capas delgadas (endpoints HTTP) encima.

Requiere: spacy + el modelo es_core_news_sm (o es_core_news_md/lg
si quieres mas precision, a costa de mas peso).
"""

from __future__ import annotations

import spacy
from spacy import displacy
import logging
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.backend.config import SPACY_MODEL, SPACY_FALLBACK_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. Carga del modelo (una sola vez, cacheado)
# ---------------------------------------------------------------------
# lru_cache evita recargar el modelo en cada request. En Lambda esto
# tambien ayuda: si el contenedor se reutiliza ("warm start"), el
# modelo ya esta en memoria y la siguiente invocacion es rapida.

@lru_cache(maxsize=1)
def get_nlp(model_name: str | None = None) -> spacy.language.Language:
    """Carga el modelo de spaCy con cache LRU y fallback automático."""
    selected = model_name or SPACY_MODEL
    try:
        return spacy.load(selected)
    except OSError:
        logger.warning(
            "Modelo '%s' no disponible localmente. Cargando fallback '%s'.",
            selected,
            SPACY_FALLBACK_MODEL,
        )
        return spacy.load(SPACY_FALLBACK_MODEL)


# ---------------------------------------------------------------------
# 2. Limpieza + Transformacion + Etiquetado (pasos 1, 2 y 3 del apunte)
# ---------------------------------------------------------------------
def clean_and_transform(text: str) -> list[dict]:
    """
    Paso 1 (Limpieza): elimina stopwords y puntuacion.
    Paso 2 (Transformacion): pasa todo a minuscula y lematiza
        (convierte verbo conjugado -> verbo en infinitivo/normal).
    Paso 3 (Etiquetado): le pone a cada token su etiqueta POS
        (Noun, Verb, Adv, etc.), igual que en tus apuntes.

    Devuelve una lista de tokens "limpios" con su lema y su POS,
    en el mismo orden en que aparecen en el texto.
    """
    nlp = get_nlp()
    
    # nlp es un objeto que viene de la libreria de Spacy
    # se encarga de recibir el texto en plano para realizar un analisisi de nlp
    doc = nlp(text)

    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            # Elimina el ruido al momento de tokenizar
            # elimina los stopwords (el, lo, las, etc..) conectores
            # elimina signis de puntuacion
            # elimina espacios en blanco, saltos de linea, etc..
            continue
    
        tokens.append(
            {
                "texto_original": tok.text,
                "lema": tok.lemma_.lower(), # corriendo --> correr
                "pos": tok.pos_,       # NOUN, VERB, ADJ, ADV, PROPN...
                "pos_detalle": tok.tag_, # Guarda la etiqueta gramatical detallada dependiente del idioma (codigo tecnico)
            }
            #hace el append del dic a la lista tokens
        )
    return tokens


def lemmas_only(text: str) -> list[str]:
    """Devuelve solo la lista de lemas limpios (lo que usamos para
    construir el vocabulario en la etapa de codificacion)."""
    
    # Recorre el diccionario de la funcion clean_and_transform pero solo de la clave [lema] y crea una una lista extrayendo esos valores
    # t trae un diccionario
    return [t["lema"] for t in clean_and_transform(text)]
    


# ---------------------------------------------------------------------
# 3. Dependencias sintacticas (endpoint /dependency)
# ---------------------------------------------------------------------
def dependency_parse(text: str) -> dict:
    """
    Analisis de dependencias: para cada token dice de que palabra
    depende sintacticamente y con que relacion (sujeto, objeto, etc).
    Ademas, utiliza el modulo displacy de spaCy para generar el arbol visual
    de dependencias (en formato SVG) para cada oracion.
    """
    nlp = get_nlp()
    doc = nlp(text)

    tokens = [
        {
            "texto": tok.text,
            "lema": tok.lemma_.lower(),
            "dependencia": tok.dep_,      # nsubj, obj, root, etc.
            "cabeza": tok.head.text,      # palabra de la que depende
            "pos": tok.pos_,
        }
        for tok in doc
        if not tok.is_space
    ]

    arboles = [
        {
            "oracion": sent.text.strip(),
            "svg": displacy.render(sent, style="dep", jupyter=False),
        }
        for sent in doc.sents
        if sent.text.strip()
    ]

    return {
        "dependencias": tokens,
        "arboles": arboles,
    }


# ---------------------------------------------------------------------
# 4. Entidades nombradas (endpoint /ner)
# ---------------------------------------------------------------------
def named_entities(text: str) -> list[dict]:
    """
    Reconocimiento de entidades nombradas: personas, lugares,
    organizaciones, fechas, etc. (p.ej. "Juan" -> PER, "Bogota" -> LOC,
    como en el ejemplo de tus apuntes).
    """
    nlp = get_nlp()
    doc = nlp(text)

    return [
        {
            "texto": ent.text,
            "etiqueta": ent.label_,
            "inicio_caracter": ent.start_char,
            "fin_caracter": ent.end_char,
        }
        for ent in doc.ents
    ]


# ---------------------------------------------------------------------
# 5. Pipeline completo (endpoint /full)
# ---------------------------------------------------------------------
def full_pipeline(text: str) -> dict:
    """Corre limpieza+transformacion+etiquetado, dependencias y NER
    sobre el mismo texto y devuelve todo junto."""
    return {
        "texto_original": text,
        "tokens_procesados": clean_and_transform(text),
        "dependencias": dependency_parse(text),
        "entidades": named_entities(text),
    }


# ---------------------------------------------------------------------
# 6. Codificacion: One-Hot / Bag of Words / TF-IDF (endpoint /encoding)
# ---------------------------------------------------------------------
def encode_corpus(corpus: list[str], method: str = "tfidf") -> dict:
    """
    Replica exactamente el ejemplo de tus apuntes:

        D = {d1, d2, d3}  -> corpus
        1) limpieza + transformacion + etiquetado  (clean_and_transform)
        2) se define el vocabulario a partir de los lemas limpios
        3) se codifica cada documento contra ese vocabulario con:
             - one-hot   -> presencia/ausencia (0/1)
             - bow       -> frecuencia absoluta
             - tfidf     -> importancia relativa (formula de sklearn,
                             la misma que anotaste: log((1+n)/(1+df))+1)

    method: "onehot" | "bow" | "tfidf"
    """
    method = method.lower()
    if method not in {"onehot", "bow", "tfidf"}:
        raise ValueError("method debe ser 'onehot', 'bow' o 'tfidf'")

    # Paso 1-3: limpiamos y lematizamos cada documento del corpus,
    # y lo volvemos a unir en texto para que el Vectorizer de sklearn
    # tokenice sobre nuestros LEMAS limpios (no sobre el texto crudo).
    docs_lemmatizados = [" ".join(lemmas_only(doc)) for doc in corpus]

    if method == "bow":
        vectorizer = CountVectorizer()
    elif method == "onehot":
        vectorizer = CountVectorizer(binary=True)
    else:  # tfidf
        # norm=None para que el valor se vea igual de "crudo" que en
        # el calculo manual de tus apuntes (tf * idf), sin normalizar
        # el vector completo a longitud 1.
        vectorizer = TfidfVectorizer(norm=None, smooth_idf=True)

    matrix = vectorizer.fit_transform(docs_lemmatizados)
    vocabulario = vectorizer.get_feature_names_out().tolist()

    return {
        "metodo": method,
        "vocabulario": vocabulario,
        "documentos": [
            {
                "documento_original": corpus[i],
                "tokens_usados": docs_lemmatizados[i].split(),
                "vector": matrix[i].toarray()[0].round(4).tolist(),
            }
            for i in range(len(corpus))
        ],
    }


# ---------------------------------------------------------------------
# 7. Pipeline completo paso a paso sobre un corpus (endpoint /pipeline)
# ---------------------------------------------------------------------
def corpus_pipeline(corpus: list[str], method: str = "tfidf") -> dict:
    """
    Ejecuta todo el flujo NLP paso a paso sobre un corpus de documentos:
      Paso 1: Limpieza, transformacion y etiquetado POS (processed).
      Paso 2: Analisis de dependencias sintacticas con arboles visuales spaCy displaCy (dependency).
      Paso 3: Reconocimiento de entidades nombradas (ner).
      Paso 4: Resumen integrado full por cada documento (full).
      Paso 5: Codificacion global del corpus: construccion de vocabulario
              y vectorizacion segun el metodo (onehot, bow, tfidf) (encoding).
    """
    method = method.lower()
    if method not in {"onehot", "bow", "tfidf"}:
        raise ValueError("method debe ser 'onehot', 'bow' o 'tfidf'")

    documentos_paso_a_paso = []
    for i, doc_text in enumerate(corpus):
        tokens_proc = clean_and_transform(doc_text)
        lemas = lemmas_only(doc_text)
        dep_data = dependency_parse(doc_text)
        ner_data = named_entities(doc_text)

        documentos_paso_a_paso.append({
            "id": i + 1,
            "documento_original": doc_text,
            "paso_1_processed": {
                "tokens_limpios": tokens_proc,
                "lemas": lemas,
            },
            "paso_2_dependency": dep_data,
            "paso_3_ner": {
                "entidades": ner_data,
            },
            "paso_4_full": {
                "texto_original": doc_text,
                "tokens_procesados": tokens_proc,
                "dependencias": dep_data,
                "entidades": ner_data,
            },
        })

    encoding_data = encode_corpus(corpus, method)

    return {
        "resumen": {
            "total_documentos": len(corpus),
            "metodo_codificacion": method,
            "tamano_vocabulario": len(encoding_data["vocabulario"]),
        },
        "paso_a_paso_documentos": documentos_paso_a_paso,
        "paso_5_encoding_corpus": encoding_data,
    }

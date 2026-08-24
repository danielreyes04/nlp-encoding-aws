"""
api_lambda/main.py
------------------
API FastAPI para despliegue serverless en AWS Lambda (Mangum + Docker).

El handler de Lambda se configura como:  main.handler

La lógica es idéntica a api_ec2/main.py; sólo cambia el adaptador
de transporte (Mangum en lugar de Uvicorn) y se agrega soporte CORS
para Lambda Function URL.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from mangum import Mangum  # type: ignore

from app.backend.config import API_TITLE, API_DESCRIPTION, API_VERSION, CORS_ORIGINS
from app.backend.nlp_pipeline import (
    clean_texts,
    pos_analysis_batch,
    ner_analysis_batch,
    dependency_html,
    vectorize,
)
from app.backend.schemas import TextRequest, DepRequest, VectorizeRequest

app = FastAPI(
    title=f"{API_TITLE} (Lambda)",
    description=API_DESCRIPTION,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "deployment": "Lambda"}


# ---------------------------------------------------------------------------
# POST /api/v1/clean
# ---------------------------------------------------------------------------

@app.post("/api/v1/clean", tags=["nlp"])
def clean(req: TextRequest):
    """
    Limpieza de texto.

    Convierte a minúsculas, elimina signos de puntuación (como separadores),
    elimina stopwords (Token.is_stop de es_core_news_sm) y normaliza espacios.
    Conserva letras acentuadas, ñ y dígitos.

    Acepta texto único o lote. Siempre retorna lista de strings.
    """
    texts = [req.text] if isinstance(req.text, str) else req.text
    return {"cleaned_text": clean_texts(texts)}


# ---------------------------------------------------------------------------
# POST /api/v1/pos
# ---------------------------------------------------------------------------

@app.post("/api/v1/pos", tags=["nlp"])
def pos(req: TextRequest):
    """
    Análisis POS (Part-of-Speech).

    Retorna tokens con text, pos (UPOS) y lemma para cada documento.
    Acepta texto único o lote. results[i] corresponde a texts[i].
    """
    texts = [req.text] if isinstance(req.text, str) else req.text
    return {"results": pos_analysis_batch(texts)}


# ---------------------------------------------------------------------------
# POST /api/v1/ner
# ---------------------------------------------------------------------------

@app.post("/api/v1/ner", tags=["nlp"])
def ner(req: TextRequest):
    """
    Reconocimiento de entidades nombradas (NER).

    Detecta entidades con text, label, start (inclusivo) y end (exclusivo).
    Acepta texto único o lote. results[i] corresponde a texts[i].
    """
    texts = [req.text] if isinstance(req.text, str) else req.text
    return {"results": ner_analysis_batch(texts)}


# ---------------------------------------------------------------------------
# POST /api/v1/visualize/dep
# ---------------------------------------------------------------------------

@app.post("/api/v1/visualize/dep", response_class=HTMLResponse, tags=["nlp"])
def visualize_dep(req: DepRequest):
    """
    Visualización de dependencias sintácticas.

    Genera un documento HTML con el SVG de displaCy.
    Solo acepta un único string (no batch).
    """
    html = dependency_html(req.text)
    return HTMLResponse(content=html, status_code=200)


# ---------------------------------------------------------------------------
# POST /api/v1/vectorize
# ---------------------------------------------------------------------------

@app.post("/api/v1/vectorize", tags=["nlp"])
def vectorize_endpoint(req: VectorizeRequest):
    """
    Vectorización de corpus.

    Construye vocabulario en orden lexicográfico y calcula:
    - one_hot    : lista de N matrices (una por documento)
    - bag_of_words: matriz N × |V|
    - tf_idf     : matriz N × |V| (TF-IDF sin normalización, 4 decimales)

    Requiere al menos 2 documentos.
    """
    return vectorize(req.documents)


# ---------------------------------------------------------------------------
# Handler para AWS Lambda
# ---------------------------------------------------------------------------

handler = Mangum(app, lifespan="off")

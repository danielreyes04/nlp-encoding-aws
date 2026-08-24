"""
api_ec2/main.py
---------------
API FastAPI para despliegue persistente en EC2/Cloud9 con Uvicorn.

Ejecutar:
    uvicorn api_ec2.main:app --host 0.0.0.0 --port 8000 --reload

Documentación interactiva:
    http://<IP-pública>:8000/docs
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

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
    title=f"{API_TITLE} (EC2)",
    description=API_DESCRIPTION,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Archivos estáticos del frontend
_static_dir = Path(__file__).resolve().parent.parent / "app" / "frontend"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "deployment": "EC2"}


@app.get("/client", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def serve_client():
    client_path = _static_dir / "client.html"
    if client_path.exists():
        return FileResponse(str(client_path))
    raise HTTPException(status_code=404, detail="Cliente web no encontrado")


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

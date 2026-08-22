"""
API #1 - Corre en una instancia EC2 (o dentro de Cloud9, que por debajo
es una EC2) con uvicorn como servidor persistente.

Ejecutar localmente / en la instancia:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Documentacion interactiva automatica en:
    http://<IP-publica-de-tu-instancia>:8000/docs
"""

import sys
from pathlib import Path

# Permite importar el paquete "app" (nlp_pipeline, schemas) que vive
# un nivel arriba, compartido con api_lambda.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.config import API_TITLE, API_DESCRIPTION, API_VERSION, CORS_ORIGINS
from app.backend.nlp_pipeline import (
    clean_and_transform,
    dependency_parse,
    named_entities,
    full_pipeline,
    encode_corpus,
    corpus_pipeline,
)
from app.backend.schemas import TextRequest, EncodingRequest

app = FastAPI(
    title=f"{API_TITLE} (EC2)",
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos para la interfaz de usuario
static_dir = Path(__file__).resolve().parent.parent / "app" / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def health():
    return {"status": "ok", "servicio": "EC2/Cloud9"}


@app.get("/ui", include_in_schema=False)
@app.get("/client", include_in_schema=False)
def serve_client():
    """Sirve la interfaz web del cliente."""
    client_path = static_dir / "client.html"
    if client_path.exists():
        return FileResponse(str(client_path))
    raise HTTPException(status_code=404, detail="Cliente web no encontrado")


@app.post("/processed")
def processed(req: TextRequest):
    """Limpieza + transformacion + etiquetado (pasos 1-3 del flujo)."""
    return {"tokens": clean_and_transform(req.text)}


@app.post("/dependency")
def dependency(req: TextRequest):
    """Analisis de dependencias sintacticas."""
    return dependency_parse(req.text)


@app.post("/ner")
def ner(req: TextRequest):
    """Reconocimiento de entidades nombradas."""
    return {"entidades": named_entities(req.text)}


@app.post("/full")
def full(req: TextRequest):
    """Pipeline completo: processed + dependency + ner en un solo llamado."""
    return full_pipeline(req.text)


@app.post("/encoding")
def encoding(req: EncodingRequest):
    """Codificacion del corpus: one-hot | bow | tfidf."""
    try:
        return encode_corpus(req.corpus, req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/pipeline")
def pipeline(req: EncodingRequest):
    """Pipeline integral paso a paso (processed + dependency + ner + full + encoding) sobre un corpus."""
    try:
        return corpus_pipeline(req.corpus, req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
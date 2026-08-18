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

from app.nlp_pipeline import (
    clean_and_transform,
    dependency_parse,
    named_entities,
    full_pipeline,
    encode_corpus,
)
from app.schemas import TextRequest, EncodingRequest

app = FastAPI(
    title="NLP Encoding API (EC2)",
    description="Preprocesamiento y codificacion de texto con spaCy",
    version="1.0.0",
)

# Permite que client.html (abierto desde file:// o cualquier origen)
# pueda llamar a esta API sin que el navegador lo bloquee por CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "servicio": "EC2/Cloud9"}


@app.post("/processed")
def processed(req: TextRequest):
    """Limpieza + transformacion + etiquetado (pasos 1-3 del flujo)."""
    return {"tokens": clean_and_transform(req.text)}


@app.post("/dependency")
def dependency(req: TextRequest):
    """Analisis de dependencias sintacticas."""
    return {"dependencias": dependency_parse(req.text)}


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
"""
API #2 - La MISMA logica (mismo app/nlp_pipeline.py) pero empaquetada
para correr dentro de AWS Lambda, detras de API Gateway.

Mangum convierte una app FastAPI (ASGI) en un "handler" que Lambda
sabe invocar. Los endpoints y el flujo de datos son IDENTICOS a
api_ec2/main.py -> por eso ambas apis siguen el mismo comportamiento,
solo cambia como se despliegan.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from app.nlp_pipeline import (
    clean_and_transform,
    dependency_parse,
    named_entities,
    full_pipeline,
    encode_corpus,
)
from app.schemas import TextRequest, EncodingRequest

app = FastAPI(
    title="NLP Encoding API (Lambda)",
    description="Preprocesamiento y codificacion de texto con spaCy",
    version="1.0.0",
)


@app.get("/")
def health():
    return {"status": "ok", "servicio": "Lambda"}


@app.post("/processed")
def processed(req: TextRequest):
    return {"tokens": clean_and_transform(req.text)}


@app.post("/dependency")
def dependency(req: TextRequest):
    return {"dependencias": dependency_parse(req.text)}


@app.post("/ner")
def ner(req: TextRequest):
    return {"entidades": named_entities(req.text)}


@app.post("/full")
def full(req: TextRequest):
    return full_pipeline(req.text)


@app.post("/encoding")
def encoding(req: EncodingRequest):
    try:
        return encode_corpus(req.corpus, req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Esto es lo que AWS Lambda va a invocar en cada request
# (se configura como "Handler" = main.handler)
handler = Mangum(app)

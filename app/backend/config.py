"""
config.py
---------
Configuración centralizada del proyecto NLP Pipeline.
Permite gestionar parámetros mediante variables de entorno con valores por defecto.
"""

import os
from typing import List

# Modelo de spaCy preferido y modelo alternativo liviano (fallback)
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "es_core_news_sm")
SPACY_FALLBACK_MODEL: str = os.getenv("SPACY_FALLBACK_MODEL", "es_core_news_sm")

# Metadatos generales de la API
API_TITLE: str = "NLP Pipeline & Encoding API"
API_DESCRIPTION: str = (
    "API para procesamiento de lenguaje natural y codificación vectorial "
    "de textos en español mediante spaCy y scikit-learn."
)
API_VERSION: str = "1.0.0"

# CORS
# allow_credentials=True es incompatible con allow_origins=["*"] en los navegadores,
# por eso se usa allow_origins=["*"] + allow_credentials=False (ver api_ec2/main.py).
CORS_ORIGINS_RAW: str = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS: List[str] = (
    [origin.strip() for origin in CORS_ORIGINS_RAW.split(",")]
    if CORS_ORIGINS_RAW != "*"
    else ["*"]
)

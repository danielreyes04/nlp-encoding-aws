"""
test_api.py
-----------
Pruebas de integración de los endpoints HTTP en FastAPI usando TestClient.
"""

from fastapi.testclient import TestClient
from api_ec2.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_processed_endpoint():
    payload = {"text": "Mi gato come pescado"}
    response = client.post("/processed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert len(data["tokens"]) > 0


def test_dependency_endpoint():
    payload = {"text": "El caballo corre muy rapido"}
    response = client.post("/dependency", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "dependencias" in data
    assert "arboles" in data


def test_ner_endpoint():
    payload = {"text": "Juan comio en Bogota"}
    response = client.post("/ner", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "entidades" in data


def test_full_endpoint():
    payload = {"text": "Juan comio pescado en Bogota"}
    response = client.post("/full", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tokens_procesados" in data
    assert "dependencias" in data
    assert "entidades" in data


def test_encoding_endpoint():
    payload = {
        "corpus": [
            "Mi gato, su gato y nuestro gato comen pescado",
            "Juan comio en Bogota",
            "El caballo corre muy rapido",
        ],
        "method": "tfidf",
    }
    response = client.post("/encoding", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["metodo"] == "tfidf"
    assert "vocabulario" in data
    assert len(data["documentos"]) == 3


def test_pipeline_endpoint():
    payload = {
        "corpus": [
            "Mi gato come pescado",
            "Juan vive en Bogota",
        ],
        "method": "bow",
    }
    response = client.post("/pipeline", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "resumen" in data
    assert "paso_a_paso_documentos" in data
    assert "paso_5_encoding_corpus" in data


def test_validation_empty_text():
    response = client.post("/processed", json={"text": ""})
    assert response.status_code == 422

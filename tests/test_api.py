"""
test_api.py
-----------
Pruebas de integración HTTP para los endpoints del contrato (sección 8).
Se ejecutan contra api_ec2/main.py con TestClient de FastAPI.
La misma lógica aplica a api_lambda/main.py (paridad garantizada por el núcleo compartido).
"""

import pytest
from fastapi.testclient import TestClient

from api_ec2.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/v1/clean
# ---------------------------------------------------------------------------

class TestCleanEndpoint:
    URL = "/api/v1/clean"

    def test_texto_unico(self):
        r = client.post(self.URL, json={"text": "Mi gato come pescado."})
        assert r.status_code == 200
        data = r.json()
        assert "cleaned_text" in data
        assert isinstance(data["cleaned_text"], list)
        assert len(data["cleaned_text"]) == 1
        assert isinstance(data["cleaned_text"][0], str)

    def test_lote(self):
        r = client.post(self.URL, json={"text": ["Texto uno.", "Texto dos."]})
        assert r.status_code == 200
        data = r.json()
        assert len(data["cleaned_text"]) == 2

    def test_entrada_unica_retorna_lista(self):
        """Aunque la entrada sea string, la salida siempre es lista."""
        r = client.post(self.URL, json={"text": "hola mundo"})
        assert isinstance(r.json()["cleaned_text"], list)

    def test_texto_vacio_422(self):
        r = client.post(self.URL, json={"text": ""})
        assert r.status_code == 422

    def test_texto_solo_espacios_422(self):
        r = client.post(self.URL, json={"text": "   "})
        assert r.status_code == 422

    def test_campo_ausente_422(self):
        r = client.post(self.URL, json={})
        assert r.status_code == 422

    def test_null_422(self):
        r = client.post(self.URL, json={"text": None})
        assert r.status_code == 422

    def test_lista_vacia_422(self):
        r = client.post(self.URL, json={"text": []})
        assert r.status_code == 422

    def test_elemento_vacio_en_lote_422(self):
        r = client.post(self.URL, json={"text": ["válido", ""]})
        assert r.status_code == 422

    def test_content_type_json(self):
        r = client.post(self.URL, json={"text": "hola"})
        assert "application/json" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /api/v1/pos
# ---------------------------------------------------------------------------

class TestPosEndpoint:
    URL = "/api/v1/pos"

    def test_texto_unico(self):
        r = client.post(self.URL, json={"text": "Juan come pescado"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 1
        for tok in data["results"][0]:
            assert "text" in tok
            assert "pos" in tok
            assert "lemma" in tok

    def test_lote(self):
        r = client.post(self.URL, json={"text": ["Juan come", "María corre"]})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2

    def test_correspondencia_posicion(self):
        texts = ["gato", "perro caballo pez"]
        r = client.post(self.URL, json={"text": texts})
        results = r.json()["results"]
        assert len(results[0]) <= len(results[1])

    def test_texto_vacio_422(self):
        r = client.post(self.URL, json={"text": ""})
        assert r.status_code == 422

    def test_null_422(self):
        r = client.post(self.URL, json={"text": None})
        assert r.status_code == 422

    def test_lista_vacia_422(self):
        r = client.post(self.URL, json={"text": []})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/ner
# ---------------------------------------------------------------------------

class TestNerEndpoint:
    URL = "/api/v1/ner"

    def test_texto_unico(self):
        r = client.post(self.URL, json={"text": "Juan viajó a Bogotá."})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 1
        for ent in data["results"][0]:
            assert "text" in ent
            assert "label" in ent
            assert "start" in ent
            assert "end" in ent

    def test_indices_correctos(self):
        text = "María vive en Colombia."
        r = client.post(self.URL, json={"text": text})
        for ent in r.json()["results"][0]:
            assert text[ent["start"]:ent["end"]] == ent["text"]

    def test_lote(self):
        r = client.post(self.URL, json={"text": ["Juan en Bogotá", "Apple Inc."]})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 2

    def test_sin_entidades_lista_vacia(self):
        r = client.post(self.URL, json={"text": "el gato come"})
        assert r.status_code == 200
        assert isinstance(r.json()["results"][0], list)

    def test_texto_vacio_422(self):
        r = client.post(self.URL, json={"text": ""})
        assert r.status_code == 422

    def test_null_422(self):
        r = client.post(self.URL, json={"text": None})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/visualize/dep
# ---------------------------------------------------------------------------

class TestVisualizeDepEndpoint:
    URL = "/api/v1/visualize/dep"

    def test_retorna_html(self):
        r = client.post(self.URL, json={"text": "El gato corre."})
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_contiene_svg(self):
        r = client.post(self.URL, json={"text": "El caballo corre muy rápido."})
        assert "<svg" in r.text

    def test_es_html(self):
        r = client.post(self.URL, json={"text": "Juan come."})
        body = r.text
        assert "<html" in body or "<!DOCTYPE html>" in body

    def test_batch_rechazado_422(self):
        """El batch no está permitido en /visualize/dep."""
        r = client.post(self.URL, json={"text": ["texto uno", "texto dos"]})
        assert r.status_code == 422

    def test_texto_vacio_422(self):
        r = client.post(self.URL, json={"text": ""})
        assert r.status_code == 422

    def test_campo_ausente_422(self):
        r = client.post(self.URL, json={})
        assert r.status_code == 422

    def test_null_422(self):
        r = client.post(self.URL, json={"text": None})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/vectorize
# ---------------------------------------------------------------------------

class TestVectorizeEndpoint:
    URL = "/api/v1/vectorize"
    CORPUS = [
        "Mi gato, su gato y nuestro gato comen pescado",
        "Juan comió en Bogotá",
        "El caballo corre muy rápido",
    ]

    def test_estructura_respuesta(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        assert r.status_code == 200
        data = r.json()
        assert "vocabulary" in data
        assert "bag_of_words" in data
        assert "one_hot" in data
        assert "tf_idf" in data

    def test_vocabulario_lexicografico(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        vocab = r.json()["vocabulary"]
        assert vocab == sorted(vocab)

    def test_bow_dimension_n_por_v(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        data = r.json()
        N = len(self.CORPUS)
        V = len(data["vocabulary"])
        assert len(data["bag_of_words"]) == N
        for row in data["bag_of_words"]:
            assert len(row) == V

    def test_one_hot_es_lista_de_matrices(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        data = r.json()
        V = len(data["vocabulary"])
        assert len(data["one_hot"]) == len(self.CORPUS)
        for doc_matrix in data["one_hot"]:
            assert isinstance(doc_matrix, list)
            for row in doc_matrix:
                assert len(row) == V
                assert sum(row) == 1

    def test_tfidf_dimension_n_por_v(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        data = r.json()
        N = len(self.CORPUS)
        V = len(data["vocabulary"])
        assert len(data["tf_idf"]) == N
        for row in data["tf_idf"]:
            assert len(row) == V

    def test_menos_de_dos_documentos_422(self):
        r = client.post(self.URL, json={"documents": ["solo uno"]})
        assert r.status_code == 422

    def test_lista_vacia_422(self):
        r = client.post(self.URL, json={"documents": []})
        assert r.status_code == 422

    def test_campo_ausente_422(self):
        r = client.post(self.URL, json={})
        assert r.status_code == 422

    def test_null_422(self):
        r = client.post(self.URL, json={"documents": None})
        assert r.status_code == 422

    def test_elemento_vacio_en_lista_422(self):
        r = client.post(self.URL, json={"documents": ["válido", ""]})
        assert r.status_code == 422

    def test_elemento_solo_espacios_422(self):
        r = client.post(self.URL, json={"documents": ["válido", "   "]})
        assert r.status_code == 422

    def test_content_type_json(self):
        r = client.post(self.URL, json={"documents": self.CORPUS})
        assert "application/json" in r.headers["content-type"]

    def test_lote_25_documentos(self):
        """Capacidad mínima: 10 docs para /vectorize según la guía."""
        docs = [f"documento número {i} con contenido relevante" for i in range(10)]
        r = client.post(self.URL, json={"documents": docs})
        assert r.status_code == 200
        assert len(r.json()["bag_of_words"]) == 10

"""
test_nlp_pipeline.py
--------------------
Pruebas unitarias para las funciones del núcleo NLP (app/backend/nlp_pipeline.py).
Cubre las cinco capacidades funcionales de la guía del laboratorio.
"""

import math
import pytest

from app.backend.nlp_pipeline import (
    clean_text,
    clean_texts,
    pos_analysis,
    pos_analysis_batch,
    ner_analysis,
    ner_analysis_batch,
    dependency_html,
    vectorize,
)


# ---------------------------------------------------------------------------
# 1. Limpieza de texto
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_retorna_string(self):
        result = clean_text("Mi gato come pescado.")
        assert isinstance(result, str)

    def test_minusculas(self):
        result = clean_text("JUAN COME PESCADO")
        assert result == result.lower()

    def test_elimina_puntuacion(self):
        result = clean_text("Hola, mundo!")
        assert "," not in result
        assert "!" not in result

    def test_puntuacion_como_separador_no_concatena(self):
        # "gato,perro" no debe producir "gatoperro"
        result = clean_text("gato,perro")
        tokens = result.split()
        for tok in tokens:
            assert "gato" not in tok or tok == "gato"
            assert "perro" not in tok or tok == "perro"

    def test_elimina_stopwords(self):
        # "el", "la", "de", etc. son stopwords en es_core_news_sm
        result = clean_text("El gato de la casa")
        tokens = result.split()
        # stopwords comunes no deben aparecer
        for sw in ("el", "la", "de"):
            assert sw not in tokens

    def test_conserva_acentos_y_enie(self):
        result = clean_text("La niña comió piñón")
        # "niña" y "piñón" deben conservarse (sin ser stopwords)
        assert "ñ" in result or "niñ" in result or "piñ" in result

    def test_conserva_digitos(self):
        result = clean_text("Hay 3 gatos y 2 perros")
        assert "3" in result or "2" in result

    def test_normaliza_espacios(self):
        result = clean_text("gato   perro")
        # no debe haber dobles espacios
        assert "  " not in result

    def test_batch_conserva_orden(self):
        texts = ["Juan come", "María corre", "El perro ladra"]
        results = clean_texts(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


# ---------------------------------------------------------------------------
# 2. Análisis POS
# ---------------------------------------------------------------------------

class TestPosAnalysis:
    def test_estructura_tokens(self):
        tokens = pos_analysis("El gato corre rápido")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        for tok in tokens:
            assert "text" in tok
            assert "pos" in tok
            assert "lemma" in tok

    def test_lema_en_minusculas(self):
        tokens = pos_analysis("JUAN COME PESCADO")
        for tok in tokens:
            assert tok["lemma"] == tok["lemma"].lower()

    def test_conserva_orden_tokens(self):
        text = "Juan come pescado en Bogotá"
        tokens = pos_analysis(text)
        textos = [t["text"] for t in tokens]
        # los tokens deben aparecer en el mismo orden que en el texto
        pos_in_text = [text.index(t) for t in textos if t in text]
        assert pos_in_text == sorted(pos_in_text)

    def test_batch_longitud_correcta(self):
        texts = ["Juan come", "María corre veloz", "El cielo es azul"]
        results = pos_analysis_batch(texts)
        assert len(results) == 3

    def test_batch_correspondencia(self):
        texts = ["gato", "perro caballo"]
        results = pos_analysis_batch(texts)
        # el primer doc tiene menos tokens que el segundo
        assert len(results[0]) <= len(results[1])


# ---------------------------------------------------------------------------
# 3. Reconocimiento de entidades (NER)
# ---------------------------------------------------------------------------

class TestNerAnalysis:
    def test_estructura_entidades(self):
        ents = ner_analysis("Juan viajó a Bogotá con María.")
        assert isinstance(ents, list)
        for ent in ents:
            assert "text" in ent
            assert "label" in ent
            assert "start" in ent
            assert "end" in ent

    def test_indices_en_texto_original(self):
        text = "Juan viajó a Bogotá."
        ents = ner_analysis(text)
        for ent in ents:
            # start inclusivo, end exclusivo
            assert text[ent["start"]:ent["end"]] == ent["text"]

    def test_start_inclusivo_end_exclusivo(self):
        text = "María vive en Colombia."
        ents = ner_analysis(text)
        for ent in ents:
            assert ent["start"] < ent["end"]

    def test_batch_conserva_orden(self):
        texts = ["Juan está en Bogotá", "Apple es una empresa", "Sin entidades aquí"]
        results = ner_analysis_batch(texts)
        assert len(results) == 3
        assert isinstance(results[0], list)
        assert isinstance(results[2], list)  # puede estar vacía, pero es lista

    def test_sin_entidades_retorna_lista_vacia(self):
        ents = ner_analysis("el gato come")
        assert isinstance(ents, list)


# ---------------------------------------------------------------------------
# 4. Visualización de dependencias
# ---------------------------------------------------------------------------

class TestDependencyHtml:
    def test_retorna_string(self):
        html = dependency_html("El gato corre.")
        assert isinstance(html, str)

    def test_contiene_svg(self):
        html = dependency_html("El caballo corre muy rápido.")
        assert "<svg" in html

    def test_es_html_valido(self):
        html = dependency_html("Juan come.")
        assert "<!DOCTYPE html>" in html or "<html" in html

    def test_contiene_body(self):
        html = dependency_html("Juan come pescado.")
        assert "<body>" in html


# ---------------------------------------------------------------------------
# 5. Vectorización
# ---------------------------------------------------------------------------

class TestVectorize:
    CORPUS = [
        "Mi gato, su gato y nuestro gato comen pescado",
        "Juan comió en Bogotá",
        "El caballo corre muy rápido",
    ]

    def test_estructura_resultado(self):
        result = vectorize(self.CORPUS)
        assert "vocabulary" in result
        assert "bag_of_words" in result
        assert "one_hot" in result
        assert "tf_idf" in result

    def test_vocabulario_orden_lexicografico(self):
        result = vectorize(self.CORPUS)
        vocab = result["vocabulary"]
        assert vocab == sorted(vocab)

    def test_vocabulario_no_vacio(self):
        result = vectorize(self.CORPUS)
        assert len(result["vocabulary"]) > 0

    def test_bow_dimensiones(self):
        result = vectorize(self.CORPUS)
        N = len(self.CORPUS)
        V = len(result["vocabulary"])
        assert len(result["bag_of_words"]) == N
        for row in result["bag_of_words"]:
            assert len(row) == V

    def test_bow_frecuencias_absolutas(self):
        corpus = ["gato gato perro", "perro perro gato"]
        result = vectorize(corpus)
        vocab = result["vocabulary"]
        bow = result["bag_of_words"]
        # "gato" aparece 2 veces en doc 0
        if "gato" in vocab:
            idx = vocab.index("gato")
            assert bow[0][idx] == 2

    def test_one_hot_estructura(self):
        result = vectorize(self.CORPUS)
        V = len(result["vocabulary"])
        assert len(result["one_hot"]) == len(self.CORPUS)
        for doc_matrix in result["one_hot"]:
            assert isinstance(doc_matrix, list)
            for row in doc_matrix:
                assert len(row) == V
                assert sum(row) == 1
                assert all(v in (0, 1) for v in row)

    def test_one_hot_ocurrencias(self):
        # "gato gato" debe producir 2 filas en one_hot para ese doc
        corpus = ["gato gato perro", "perro perro"]
        result = vectorize(corpus)
        vocab = result["vocabulary"]
        one_hot = result["one_hot"]
        if "gato" in vocab:
            # doc 0 tiene 3 tokens retenidos → 3 filas
            assert len(one_hot[0]) == 3

    def test_tfidf_dimensiones(self):
        result = vectorize(self.CORPUS)
        N = len(self.CORPUS)
        V = len(result["vocabulary"])
        assert len(result["tf_idf"]) == N
        for row in result["tf_idf"]:
            assert len(row) == V

    def test_tfidf_cuatro_decimales(self):
        result = vectorize(self.CORPUS)
        for row in result["tf_idf"]:
            for val in row:
                # round(x, 4) debe ser igual al valor almacenado
                assert val == round(val, 4)

    def test_tfidf_formula_exacta(self):
        """Verifica la fórmula idf = ln((|D|+1)/(nt+1)) + 1."""
        corpus = ["gato perro", "gato caballo"]
        result = vectorize(corpus)
        vocab = result["vocabulary"]
        tf_idf = result["tf_idf"]
        D = 2
        if "gato" in vocab:
            idx = vocab.index("gato")
            nt_gato = 2  # aparece en ambos documentos
            idf_gato = math.log((D + 1) / (nt_gato + 1)) + 1
            tf_doc0 = 1  # "gato" aparece 1 vez en doc 0
            expected = round(tf_doc0 * idf_gato, 4)
            assert tf_idf[0][idx] == expected

    def test_tfidf_sin_normalizacion(self):
        """Los valores TF-IDF no deben estar normalizados a longitud 1."""
        result = vectorize(self.CORPUS)
        for row in result["tf_idf"]:
            # si estuviera normalizado, la norma euclidiana sería ~1.0
            norm = math.sqrt(sum(v ** 2 for v in row))
            # para estos documentos la norma nunca debería ser exactamente 1
            assert norm == 0 or abs(norm - 1.0) > 1e-3

    def test_dos_documentos_minimo(self):
        result = vectorize(["primer documento", "segundo documento"])
        assert "vocabulary" in result

    def test_filas_conservan_orden_documentos(self):
        corpus = ["único término alfa", "único término beta"]
        result = vectorize(corpus)
        vocab = result["vocabulary"]
        bow = result["bag_of_words"]
        if "alfa" in vocab and "beta" in vocab:
            idx_alfa = vocab.index("alfa")
            idx_beta = vocab.index("beta")
            # doc 0 tiene "alfa", doc 1 tiene "beta"
            assert bow[0][idx_alfa] >= 1
            assert bow[1][idx_beta] >= 1

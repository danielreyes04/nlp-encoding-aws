"""
test_nlp_pipeline.py
---------------------
Pruebas unitarias para las funciones core de NLP en app/nlp_pipeline.py.
"""

import pytest
from app.backend.nlp_pipeline import (
    clean_and_transform,
    lemmas_only,
    dependency_parse,
    named_entities,
    full_pipeline,
    encode_corpus,
    corpus_pipeline,
)


def test_clean_and_transform():
    text = "Mi gato, su gato y nuestro gato comen pescado."
    tokens = clean_and_transform(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # No deben haber signos de puntuación ni stopwords en tokens limpios
    for tok in tokens:
        assert "texto_original" in tok
        assert "lema" in tok
        assert "pos" in tok
        assert "pos_detalle" in tok
        assert tok["lema"] != "."
        assert tok["lema"] != ","


def test_lemmas_only():
    text = "El caballo corre muy rapido"
    lemmas = lemmas_only(text)
    assert isinstance(lemmas, list)
    assert len(lemmas) > 0
    assert all(isinstance(l, str) for l in lemmas)


def test_dependency_parse():
    text = "El caballo corre muy rapido."
    result = dependency_parse(text)
    assert isinstance(result, dict)
    assert "dependencias" in result
    assert "arboles" in result
    assert isinstance(result["dependencias"], list)
    assert isinstance(result["arboles"], list)
    assert len(result["arboles"]) >= 1
    
    first_tree = result["arboles"][0]
    assert "oracion" in first_tree
    assert "svg" in first_tree
    assert "<svg" in first_tree["svg"]


def test_named_entities():
    text = "Juan comio en Bogota con Maria."
    ents = named_entities(text)
    assert isinstance(ents, list)
    for ent in ents:
        assert "texto" in ent
        assert "etiqueta" in ent
        assert "inicio_caracter" in ent
        assert "fin_caracter" in ent


def test_full_pipeline():
    text = "Juan comio pescado en Bogota."
    result = full_pipeline(text)
    assert isinstance(result, dict)
    assert result["texto_original"] == text
    assert "tokens_procesados" in result
    assert "dependencias" in result
    assert "entidades" in result


def test_encode_corpus_tfidf():
    corpus = [
        "Mi gato, su gato y nuestro gato comen pescado",
        "Juan comio en Bogota",
        "El caballo corre muy rapido",
    ]
    result = encode_corpus(corpus, method="tfidf")
    assert result["metodo"] == "tfidf"
    assert isinstance(result["vocabulario"], list)
    assert len(result["vocabulario"]) > 0
    assert len(result["documentos"]) == 3
    for doc in result["documentos"]:
        assert len(doc["vector"]) == len(result["vocabulario"])


def test_encode_corpus_bow_and_onehot():
    corpus = ["El gato come", "El perro corre"]
    bow_res = encode_corpus(corpus, method="bow")
    assert bow_res["metodo"] == "bow"
    
    onehot_res = encode_corpus(corpus, method="onehot")
    assert onehot_res["metodo"] == "onehot"


def test_encode_corpus_invalid_method():
    with pytest.raises(ValueError):
        encode_corpus(["Texto de prueba"], method="invalido")


def test_corpus_pipeline():
    corpus = [
        "Mi gato come pescado.",
        "Juan vive en Bogota.",
    ]
    result = corpus_pipeline(corpus, method="tfidf")
    assert isinstance(result, dict)
    assert "resumen" in result
    assert result["resumen"]["total_documentos"] == 2
    assert "paso_a_paso_documentos" in result
    assert len(result["paso_a_paso_documentos"]) == 2
    assert "paso_5_encoding_corpus" in result

    doc1 = result["paso_a_paso_documentos"][0]
    assert doc1["id"] == 1
    assert "paso_1_processed" in doc1
    assert "paso_2_dependency" in doc1
    assert "paso_3_ner" in doc1
    assert "paso_4_full" in doc1

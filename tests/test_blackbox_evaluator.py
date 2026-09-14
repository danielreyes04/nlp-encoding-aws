import os
import time
import httpx
import pytest

EC2_BASE_URL = os.getenv("EC2_BASE_URL", "http://localhost:8000")
LAMBDA_BASE_URL = os.getenv("LAMBDA_BASE_URL", "http://localhost:8000")

def test_blackbox_clean_parity():
    payload = {"text": ["Hola, mundo_feliz!", "El gato corre."]}
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/clean", json=payload, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/clean", json=payload, timeout=10.0)
    
    assert r_ec2.status_code == 200
    assert r_lambda.status_code == 200
    assert r_ec2.json() == r_lambda.json()

def test_blackbox_pos_parity():
    payload = {"text": ["Hola mundo", "gato"]}
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/pos", json=payload, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/pos", json=payload, timeout=10.0)
    
    assert r_ec2.status_code == 200
    assert r_lambda.status_code == 200
    assert r_ec2.json() == r_lambda.json()

def test_blackbox_ner_parity():
    payload = {"text": ["Juan viajó a Bogotá"]}
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/ner", json=payload, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/ner", json=payload, timeout=10.0)
    
    assert r_ec2.status_code == 200
    assert r_lambda.status_code == 200
    assert r_ec2.json() == r_lambda.json()

def test_blackbox_vectorize_parity():
    payload = {"documents": ["El perro corre", "El gato come", "El perro y el gato"]}
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/vectorize", json=payload, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/vectorize", json=payload, timeout=10.0)
    
    assert r_ec2.status_code == 200
    assert r_lambda.status_code == 200
    assert r_ec2.json() == r_lambda.json()

def test_blackbox_dep_parity():
    payload = {"text": "El caballo corre muy rápido"}
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/visualize/dep", json=payload, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/visualize/dep", json=payload, timeout=10.0)
    
    assert r_ec2.status_code == 200
    assert r_lambda.status_code == 200
    assert "<svg" in r_ec2.text
    assert "<svg" in r_lambda.text
    # Dependiendo de versiones, podrían no ser byte a byte idénticos, pero funcionalmente lo son.

def test_blackbox_invalid_inputs():
    # Empty payload
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/clean", json={}, timeout=10.0)
    r_lambda = httpx.post(f"{LAMBDA_BASE_URL}/api/v1/clean", json={}, timeout=10.0)
    assert r_ec2.status_code == 422
    assert r_lambda.status_code == 422

    # Dep batch
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/visualize/dep", json={"text": ["a", "b"]}, timeout=10.0)
    assert r_ec2.status_code == 422

    # Vectorize missing documents
    r_ec2 = httpx.post(f"{EC2_BASE_URL}/api/v1/vectorize", json={"documents": ["solo uno"]}, timeout=10.0)
    assert r_ec2.status_code == 422

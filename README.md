# NLP Encoding & Pipeline AWS

Pipeline modular de **Procesamiento de Lenguaje Natural (NLP)** y **Codificación Vectorial de Corpus** en español, diseñado con arquitectura desacoplada para despliegue dual en **AWS EC2 / Cloud9** y **AWS Lambda** (mediante contenedores Docker y Mangum).

---

## 🏛️ Estructura del Proyecto

El proyecto sigue una arquitectura modular donde el paquete `app/` contiene la lógica central ("fuente de verdad"), mientras que `api_ec2/` y `api_lambda/` son adaptadores de despliegue:

```text
nlp-encoding-aws/
├── api_ec2/                     # Despliegue en Instancia EC2 / Cloud9
│   ├── __init__.py
│   ├── main.py                  # Servidor FastAPI persistente con CORS y estáticos
│   └── requirements.txt
├── api_lambda/                  # Despliegue Serverless en AWS Lambda
│   ├── __init__.py
│   ├── Dockerfile               # Imagen base AWS Lambda Python 3.12
│   ├── main.py                  # Adaptador Mangum ASGI para Lambda
│   └── requirements.txt
├── app/                         # Núcleo de Lógica de Negocio y Frontend
│   ├── __init__.py              # Re-exporta componentes de backend
│   ├── backend/                 # Lógica de negocio (Python)
│   │   ├── __init__.py          # Exportaciones del subpaquete backend
│   │   ├── config.py            # Configuración central y variables de entorno
│   │   ├── nlp_pipeline.py      # Lógica NLP: Limpieza, POS, Dependencias, NER, Codificación
│   │   └── schemas.py           # Modelos de validación Pydantic
│   └── frontend/                # Interfaz web del cliente
│       ├── client.html          # Interfaz web cliente semántica
│       ├── index.html           # Página de entrada
│       ├── style.css            # Estilos visuales y diseño del cliente
│       └── app.js               # Lógica JavaScript y renderizado interactivo
├── tests/                       # Suite de pruebas automatizadas
│   ├── __init__.py
│   ├── test_nlp_pipeline.py     # Pruebas unitarias de NLP
│   └── test_api.py              # Pruebas de integración de endpoints FastAPI
├── .dockerignore
├── .env.example                 # Plantilla de variables de entorno
├── .gitignore                   # Exclusiones de Git
├── README.md                    # Documentación del proyecto
├── requirements.txt             # Dependencias principales del proyecto
└── requirements-dev.txt         # Dependencias de desarrollo y testing
```

---

## 🔄 Flujo de Procesamiento NLP

El flujo implementa las etapas estándar de NLP sobre texto en español:

1. **Paso 1 (Limpieza + Transformación + Etiquetado POS)**:
   - Eliminación de signos de puntuación, espacios y palabras vacías (*stopwords*).
   - Lematización (reducción a forma canónica o infinitivo).
   - Asignación de etiquetas gramaticales POS (*Part of Speech*).
2. **Paso 2 (Dependencias Sintácticas)**:
   - Identificación de relaciones gramaticales (sujeto, objeto directo, cabeza sintáctica).
   - Generación del **árbol visual de dependencias** en formato SVG usando **spaCy `displacy`** por cada oración.
3. **Paso 3 (Entidades Nombradas - NER)**:
   - Detección de personas, lugares, organizaciones y fechas.
4. **Paso 4 (Full Pipeline)**:
   - Ejecución integrada de los pasos 1 a 3 para un texto individual.
5. **Paso 5 (Codificación Vectorial del Corpus)**:
   - Extracción del vocabulario global a partir de los lemas limpios.
   - Vectorización matricial mediante:
     - `onehot`: Presencia/ausencia binaria (0/1).
     - `bow`: Frecuencia absoluta de términos (*Bag of Words*).
     - `tfidf`: Ponderación de frecuencia de término e inversa de documento (*TF-IDF*).

---

## 🚀 Instalación y Configuración Local

### 1. Clonar el repositorio y crear entorno virtual

```bash
git clone <URL_DEL_REPOSITORIO>
cd nlp-encoding-aws

python3 -m venv .venv
source .venv/bin/activate   # En Linux/macOS
# .venv\Scripts\activate    # En Windows
```

### 2. Instalar dependencias y modelo de spaCy

```bash
# Dependencias base
pip install -r requirements.txt

# Modelo de spaCy para español (recomendado: md para mayor precisión)
python -m spacy download es_core_news_md

# (Opcional) Dependencias de pruebas y desarrollo
pip install -r requirements-dev.txt
```

### 3. Variables de entorno (opcional)

Crea un archivo `.env` basado en `.env.example`:

```bash
cp .env.example .env
```

---

## 🌐 Ejecución de los Servicios

### Opción A: API EC2 / Desarrollo Local (Uvicorn)

Para correr la API con recarga automática:

```bash
uvicorn api_ec2.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Documentación Swagger interactiva:** [http://ec2-34-192-45-54.compute-1.amazonaws.com:8000/docs](http://ec2-34-192-45-54.compute-1.amazonaws.com:8000/docs)
- **Cliente Web Integrado:** [https://u6dxtnjdzdxh2kddpvzuxbidni0lfpat.lambda-url.us-east-1.on.aws/docs](https://u6dxtnjdzdxh2kddpvzuxbidni0lfpat.lambda-url.us-east-1.on.aws/docs) o [http://localhost:8000/ui](https://u6dxtnjdzdxh2kddpvzuxbidni0lfpat.lambda-url.us-east-1.on.aws/docs)

### Opción B: AWS Lambda (Docker Container)

Para construir y probar la imagen de Lambda localmente:

```bash
# Construir la imagen
docker build -t nlp-lambda -f api_lambda/Dockerfile .

# Ejecutar el contenedor simulando el runtime de Lambda
docker run -p 9000:8080 nlp-lambda
```

---

## 📡 Referencia de Endpoints HTTP

Todos los endpoints reciben y devuelven `application/json`.

| Método | Endpoint | Entrada | Descripción |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | *Ninguna* | Verificación de estado de la API (*Health check*). |
| `GET` | `/client` | *Ninguna* | Sirve la interfaz gráfica de usuario. |
| `POST` | `/processed` | `{"text": "..."}` | Limpieza, lematización y etiquetas POS. |
| `POST` | `/dependency` | `{"text": "..."}` | Dependencias sintácticas y árboles SVG de spaCy `displacy`. |
| `POST` | `/ner` | `{"text": "..."}` | Reconocimiento de Entidades Nombradas (NER). |
| `POST` | `/full` | `{"text": "..."}` | Ejecución combinada de processed + dependency + ner. |
| `POST` | `/encoding` | `{"corpus": [...], "method": "tfidf"}` | Vectorización del corpus (`tfidf`, `bow` o `onehot`). |
| `POST` | `/pipeline` | `{"corpus": [...], "method": "tfidf"}` | Flujo completo paso a paso sobre todo el corpus. |

### Ejemplos de uso con `curl`

#### 1. Dependencias sintácticas con árbol displaCy (`/dependency`)
```bash
curl -X POST "http://localhost:8000/dependency" \
     -H "Content-Type: application/json" \
     -d '{"text": "El caballo corre muy rapido"}'
```

#### 2. Pipeline integral paso a paso sobre un corpus (`/pipeline`)
```bash
curl -X POST "http://localhost:8000/pipeline" \
     -H "Content-Type: application/json" \
     -d '{
       "corpus": [
         "Mi gato come pescado",
         "Juan vive en Bogota"
       ],
       "method": "tfidf"
     }'
```

---

## 🧪 Pruebas Automatizadas

Para ejecutar las pruebas unitarias y de integración:

```bash
pytest -v
```

Para generar reporte de cobertura de código:

```bash
pytest --cov=app --cov=api_ec2
```
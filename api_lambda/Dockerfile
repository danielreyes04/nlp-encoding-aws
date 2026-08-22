# Imagen base oficial de AWS para funciones Lambda en Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Instalar dependencias
COPY api_lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Descargar modelo liviano en español para optimizar tamaño y arranque en frío
RUN python -m spacy download es_core_news_sm
ENV SPACY_MODEL=es_core_news_sm

# Copiar código fuente
COPY app ${LAMBDA_TASK_ROOT}/app
COPY api_lambda/main.py ${LAMBDA_TASK_ROOT}/main.py

CMD ["main.handler"]

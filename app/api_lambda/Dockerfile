# Imagen base oficial de AWS para funciones Lambda en Python
FROM public.ecr.aws/lambda/python:3.12

COPY api_lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

RUN python -m spacy download es_core_news_sm

COPY app ${LAMBDA_TASK_ROOT}/app
COPY api_lambda/main.py ${LAMBDA_TASK_ROOT}/main.py

CMD ["main.handler"]

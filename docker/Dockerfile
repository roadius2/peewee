# CPU image for the Peewee decision service.
#   docker build -f docker/Dockerfile -t peewee:cpu .
#   docker run -p 8000:8000 -e PEEWEE_MODELS=english,multilingual -v peewee-hub:/root/.cache/huggingface peewee:cpu
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/root/.cache/huggingface
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY peewee_decide ./peewee_decide
RUN pip install --no-cache-dir ".[server,onnx]"
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3).status==200 else 1)"
CMD ["peewee", "serve", "--host", "0.0.0.0", "--port", "8000"]

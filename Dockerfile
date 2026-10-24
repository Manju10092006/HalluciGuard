FROM python:3.11.15-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080 HF_HOME=/opt/hf \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements-linux.lock /app/requirements-linux.lock
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux.lock
COPY models-pinned.json download_models.py /app/
RUN python /app/download_models.py
COPY . /app/
RUN python /app/hydrate_detector.py
ENV ALLOW_MODEL_DOWNLOADS=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HALLUCIGUARD_FREE_ONLY=true HALLUCIGUARD_LLM_PROVIDER_ORDER=openrouter \
    HALLUCIGUARD_LLM_PROVIDER=openrouter OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
    OPENROUTER_MODEL=openrouter/free HALLUCIGUARD_LLM_MODEL=openrouter/free \
    HG_CORRECTOR_PROVIDER=openrouter HG_CORRECTOR_OPENROUTER_MODEL=openrouter/free \
    OPENROUTER_MAX_RETRIES=0 OPENROUTER_MAX_TOKENS=1024 HG_CORRECTOR_MAX_ATTEMPTS=1 \
    HALLUCIGUARD_LLM_TIMEOUT=30 N8N_RETRIEVAL_ENABLED=false \
    ALWAYS_VERIFY=true ALLOW_DETECTOR_FAST_PATH=false MOCK_MODE=false
RUN python /app/container_selftest.py
EXPOSE 8080
CMD exec uvicorn orchestration.api:app --host 0.0.0.0 --port ${PORT:-8080}

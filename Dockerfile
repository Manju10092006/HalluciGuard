# HalluciGuard Backend — Docker image for Hugging Face Spaces
FROM python:3.11.9-slim

# System deps some ML wheels need at build/runtime (faiss, torch, lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker layer caching skips this on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expects the container to listen on 7860 by default.
# Also give the process a writable HF cache dir inside the container
# (Spaces runs containers as a non-root user by default).
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PORT=7860

RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 7860

CMD ["uvicorn", "orchestration.api:app", "--host", "0.0.0.0", "--port", "7860"]

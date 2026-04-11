# Disaster Response Coordination System v2.2
# OpenEnv-compatible — Hugging Face Spaces ready
#
# Build:  docker build -t disaster-response .
# Run:    docker run -p 7860:7860 disaster-response
# Test:   docker run --rm disaster-response \
#           python -c "import sys; sys.path.insert(0,'.'); \
#           from env.environment import DisasterResponseEnv; \
#           e=DisasterResponseEnv('task1_prioritization',seed=42); \
#           obs=e.reset(); print('OK incidents:', len(obs.incidents))"

FROM python:3.11-slim

LABEL maintainer="disaster-response-team"
LABEL description="Disaster Response Coordination System — OpenEnv v2.2"
LABEL version="2.2.0"

# Minimal system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires non-root user with uid=1000
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# ── Install Python deps first (cached layer) ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY env/         ./env/
COPY server.py    .
COPY inference.py .
COPY openenv.yaml .
COPY README.md    .

RUN chown -R appuser:appuser /app
USER appuser

# Port exposed by the FastAPI server
EXPOSE 7860

# Health check — /health is registered BEFORE env imports so it responds
# immediately even during slow startup. Generous start-period for cold starts.
HEALTHCHECK \
    --interval=15s \
    --timeout=10s \
    --start-period=30s \
    --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Start uvicorn — single worker is sufficient and starts faster
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]

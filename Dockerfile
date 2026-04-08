# Disaster Response Coordination System v2.1
# OpenEnv-compatible — Hugging Face Spaces ready
#
# Build:  docker build -t disaster-response .
# Run:    docker run -p 7860:7860 disaster-response
#
# Test environment logic (no HTTP):
#   docker run --rm disaster-response python -c \
#     "import sys; sys.path.insert(0,'.'); \
#      from env.environment import DisasterResponseEnv; \
#      e=DisasterResponseEnv('task1_prioritization',seed=42); \
#      obs=e.reset(); print('Tasks OK, incidents:', len(obs.incidents))"

FROM python:3.11-slim

LABEL maintainer="disaster-response-team"
LABEL description="Disaster Response Coordination System — OpenEnv v2.1"
LABEL version="2.1.0"

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps (cached layer — before copying code)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/        ./env/
COPY server.py   .
COPY inference.py .
COPY openenv.yaml .
COPY README.md   .

# Set ownership and switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port HF Spaces expects
EXPOSE 7860

# Health check so the orchestrator knows the server is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

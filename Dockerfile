# FastAgent framework - lightweight container image
# Single-stage Python 3.12 slim image, ~150MB compressed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (none required today; left as a placeholder for future native extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY fastagent ./fastagent
COPY app.py ./

RUN pip install --upgrade pip && pip install -e .

# Default to the offline mock so the demo runs without any keys.
ENV FASTAGENT_PROVIDER=mock

# Smoke-test the package import at build time so a broken image fails fast.
RUN python -c "import fastagent; print('fastagent', fastagent.__version__, 'OK')"

# Run the validation demo by default; override with `docker run ... <command>`.
CMD ["python", "app.py"]

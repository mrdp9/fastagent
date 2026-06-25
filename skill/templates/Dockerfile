# Minimal Dockerfile for FastAgent
# Build:   docker build -t fastagent .
# Run:     docker run --rm fastagent
FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the framework + demo.
COPY fastagent/ ./fastagent/
COPY app.py .

# Run the demo by default; override with `docker run --rm fastagent python my_app.py`.
CMD ["python", "app.py"]
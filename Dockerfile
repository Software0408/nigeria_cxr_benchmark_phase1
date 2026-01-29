# Stage 1: Builder - Installing dependencies with caching
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for imaging/NLP/ML libs
# Replaced obsolete libgl1-mesa-glx with libgl1 and libglx-mesa0 for Debian Trixie compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copy and install pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Re-install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copy your source code from src/
COPY src/ src/
COPY tests/ tests
# COPY configs/ configs/  # Uncomment when configs/ is added
# COPY run_pipeline.py .           # Uncomment when implemented

ENV PYTHONPATH=/app

EXPOSE 8888

CMD ["bash"]

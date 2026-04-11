# Stage 1: Builder - Install dependencies and build JupyterLab assets
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies (existing + Node.js for JupyterLab build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libfontconfig1 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copy and install Python dependencies
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.lock.txt

# Build JupyterLab assets (requires Node.js/Yarn)
# Full build for all extensions; removes Node.js need at runtime
RUN jupyter lab build --minimize=False  

# Stage 2: Final runtime image (no Node.js - slim and clean)
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and built JupyterLab assets
# Includes built assets
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/jupyter /usr/local/share/jupyter

# Re-install minimal runtime deps (no Node.js)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Copy your source code from src/
COPY src/ src/
COPY tests/ tests/
COPY configs/ configs/
# COPY run_pipeline.py .           # Uncomment when implemented

ENV PYTHONPATH=/app/src:/app

EXPOSE 8888

CMD ["bash"]

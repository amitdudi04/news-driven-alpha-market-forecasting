# Institutional Deterministic Docker Build
# Target Environment: PAPER_TRADING_READY

FROM python:3.10-slim-bullseye

# Set explicit environmental boundaries
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency Lineage
# Ensure requirements are copied and installed securely
# Note: Ideally a requirements.txt with exact hashes is used for pure determinism.
# For this phase, we use standard installation.
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy institutional codebase
COPY . /app/

# Enforce explicit permissions
# The platform should not run as root.
RUN useradd -m algo_user && chown -R algo_user:algo_user /app
USER algo_user

# Define explicit entrypoint for daily pipeline
CMD ["python", "run_daily_pipeline.py"]

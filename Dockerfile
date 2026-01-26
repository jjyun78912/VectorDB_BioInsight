# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Multi-stage Dockerfile
# R + Python environment for RNA-seq analysis
# ═══════════════════════════════════════════════════════════════

# Stage 1: R base with Bioconductor packages
FROM rocker/r-ver:4.3.2 AS r-base

# Install R dependencies for DESeq2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor packages
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org')" && \
    R -e "BiocManager::install(c('DESeq2', 'apeglm', 'EnhancedVolcano'), ask=FALSE, update=FALSE)"

# ═══════════════════════════════════════════════════════════════
# Stage 2: Python + R combined
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS production

# Copy R installation from r-base stage
COPY --from=r-base /usr/local/lib/R /usr/local/lib/R
COPY --from=r-base /usr/lib/R /usr/lib/R
COPY --from=r-base /usr/bin/R /usr/bin/R
COPY --from=r-base /usr/bin/Rscript /usr/bin/Rscript

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # R runtime dependencies
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    # Build tools
    build-essential \
    gcc \
    g++ \
    gfortran \
    # HDF5 for h5ad files
    libhdf5-dev \
    # Git for version info
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user (non-root)
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-rnaseq.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-rnaseq.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/models /app/chroma_db \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

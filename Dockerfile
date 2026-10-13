# ============================================================================
# MOMP Benchmarking Platform - Ubuntu-based Docker Image
# Solves macOS SIGSEGV (-11) by running on Linux
# ============================================================================
FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System dependencies for geospatial + scientific stack
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    # Geospatial libraries (required by GDAL, GEOS, PROJ, Fiona, GeoPandas)
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    # NetCDF / HDF5
    libnetcdf-dev \
    libhdf5-dev \
    netcdf-bin \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    # Misc
    git \
    curl \
    ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    # GDAL environment variables (critical for Fiona/GeoPandas)
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj \
    # Force single-threaded execution in container (safety + performance)
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    MPLBACKEND=Agg \
    HDF5_USE_FILE_LOCKING=FALSE \
    KMP_DUPLICATE_LIB_OK=TRUE

# ----------------------------------------------------------------------------
# Create app directory
# ----------------------------------------------------------------------------
WORKDIR /app

# ----------------------------------------------------------------------------
# Install Python dependencies (cached layer)
# ----------------------------------------------------------------------------
COPY requirements-docker.txt /tmp/requirements-docker.txt

# Install GDAL Python bindings matching system GDAL version
RUN GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir "GDAL==${GDAL_VERSION}" \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# ----------------------------------------------------------------------------
# Copy project code
# ----------------------------------------------------------------------------
COPY . /app

# ----------------------------------------------------------------------------
# Create required directories
# ----------------------------------------------------------------------------
RUN mkdir -p /app/.streamlit \
    /app/.mplcache \
    /app/.romp_runtime \
    /data \
    && chmod -R 777 /app/.mplcache /app/.romp_runtime /data

# ----------------------------------------------------------------------------
# Streamlit config (force light theme, disable telemetry)
# ----------------------------------------------------------------------------
RUN printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 500\n\n[browser]\ngatherUsageStats = false\n\n[theme]\nbase = "light"\n' \
    > /app/.streamlit/config.toml

# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py"]

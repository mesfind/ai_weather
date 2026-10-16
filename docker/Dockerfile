# Demo 4 on a DGX Spark: the Streamlit page and every model in one image.
#
# Build (from the ai_weather/demos/d folder, on the Spark; aarch64 only):
#   docker build -t ai_weather -f docker/Dockerfile .
# Run: see docker/run.sh
#
# Each model family needs conflicting libraries, so each gets its own virtual
# environment under /opt/envs/<env> (the names in demo3/catalog.py). They share
# the NGC container's PyTorch / CUDA builds through --system-site-packages, and
# docker/locks/<env>.txt pins every other package to the versions timed on the
# Spark. Model weights are not baked in: they download on first use into
# /root/.cache, which run.sh keeps on the host.
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Use bash for every RUN so `source` works (the default /bin/sh does not have it)
SHELL ["/bin/bash", "-c"]

# The locks are exact; NGC's pip constraint file would fight themdd.
ENV PIP_CONSTRAINT= \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/locks /tmp/locks

# Atlas CRPS, Aurora 1.5, AIFS v2 single (Earth2Studio 0.18). No build isolation:
# NATTEN compiles from source and must build against the container's PyTorch.
# There is no GPU during a build, so name the Spark's GB10 (sm_121) explicitly.
RUN python -m venv --system-site-packages /opt/envs/e2s018 \
 && NATTEN_CUDA_ARCH="12.1" NATTEN_N_WORKERS=16 /opt/envs/e2s018/bin/pip install --no-deps --no-build-isolation -r /tmp/locks/e2s018.txt \
 && /opt/envs/e2s018/bin/python -c "from earth2studio.models.px import AtlasCRPS, Aurora1p5, AIFS2"

# AIFS v2 ENS (needs a different anemoi-models)
RUN python -m venv --system-site-packages /opt/envs/e2s018ens \
 && /opt/envs/e2s018ens/bin/pip install --no-deps -r /tmp/locks/e2s018ens.txt \
 && /opt/envs/e2s018ens/bin/python -c "from earth2studio.models.px import AIFS2ENS"

# GraphCast and NeuralGCM (JAX)
RUN python -m venv --system-site-packages /opt/envs/graphcast \
 && /opt/envs/graphcast/bin/pip install --no-deps -r /tmp/locks/graphcast.txt \
 && /opt/envs/graphcast/bin/python -c "import graphcast, neuralgcm, jax; from earth2studio.models.px import GraphCastOperational" \
 && ln -s /opt/envs/graphcast /opt/envs/neuralgcm

# FGN (WeatherNext 2, JAX)
RUN python -m venv --system-site-packages /opt/envs/fgn \
 && /opt/envs/fgn/bin/pip install --no-deps -r /tmp/locks/fgn.txt \
 && /opt/envs/fgn/bin/python -c "from weathernext.weathernext2 import fgn; import xarray_jax"

# Earth2Studio 0.16's GraphCast wraps Datasets in xr.Dataset(...), which newer
# xarray rejects; copy them instead (the same fix the summer environment carried).
RUN cd /opt/envs/graphcast/lib/python3.12/site-packages/earth2studio/models/px \
 && sed -i -E 's/^( +)(inputs|targets_template|forcings) = xr\.Dataset\(\2\)$/\1\2 = \2.copy()/' \
      graphcast_operational.py graphcast_small.py \
 && grep -q "forcings = forcings.copy()" graphcast_operational.py \
 && grep -q "forcings = forcings.copy()" graphcast_small.py

# Install uv for fast virtual environment management
RUN pip install uv

# Set workdir and copy app code so .venv is created inside /app
WORKDIR /app
COPY . /app

# Create the UI venv
RUN rm -rf .venv && uv venv .venv

# "Activate" the venv for every later build step, the CMD and docker exec sessions.
# The model venvs above are called by absolute path, so they are not affected.
ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    DEMO_ENV_ROOT=/opt/envs
d
# Install UI requirements and pre-download cartopy maps.
# `source` has the same effect as the ENV above; it is kept to make the intent explicit.
RUN source .venv/bin/activate \
 && uv pip install -r docker/requirements.txt \
 && python -c "import cartopy.feature as cf, cartopy.io.shapereader as sr; [sr.natural_earth(resolution=r, category=c, name=n) for r in ('10m', '50m', '110m') for c, n in [('physical', 'coastline'), ('physical', 'land'), ('physical', 'ocean'), ('cultural', 'admin_0_boundary_lines_land'), ('cultural', 'admin_0_countries')]]"

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]
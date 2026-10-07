---
title: Demo 3
teaching: 1
exercises: 0
questions:
- "How do we set up and run AI weather models locally in a container?"
- "What are the core commands needed to go from environment setup to generating a forecast figure?"
- "How do we tailor the model execution to specific use cases like onset, cessation, or temperature exceedance?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Execute a streamlined 4-command workflow to build a container, run a model, and generate use-case-specific outputs."
keypoints:
- "A streamlined 4-command workflow (build container, run model, get output, generate figure) simplifies local AI forecasting."
- "Use-case flags in the code allow groups to tailor outputs for onset/cessation, temperature exceedance, or precipitation exceedance."
- "Jupyter notebooks provide an interactive environment for executing and visualizing the AI weather models."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Running AI Weather Forecast Models on a DGX Spark

This module provides a comprehensive guide to deploying, executing, and benchmarking modern AI weather forecasting models locally using containerized workflows on a DGX Spark infrastructure. 

*Note: All demonstration applications, including Demo 3, are now organized under the `demos/` directory in the repository.*

---

## 1. Environment and Workflow Setup

Step-by-step setup for the Demo 3 Streamlit platform (AI forecast lab). The entire stack operates within a single Docker container on the Spark, accessed via an SSH tunnel from your laptop's browser. This leverages modern rootless Container Device Interface (CDI) standards for secure GPU passthrough.

**Access Pathways:**
- **Path A (Pre-configured Spark):** Use a Spark where Demo 3 is already deployed (e.g., `hcwfpgx`). Proceed directly to **Step 5**. Requires Tailscale access and a user account.
- **Path B (Custom Spark Setup):** Execute Steps 1–5. Allocate ~1 hour for the initial container build, plus additional time for model weight downloads on first execution (or copy them from an existing Spark as detailed in Step 4).

### 1.1. Verify GPU Visibility via CDI
Confirm that rootless Docker can access the GPU. This command should output your GPU model:
~~~{bash}
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
~~~
*Troubleshooting:* If it fails with `unresolvable CDI devices nvidia.com/gpu=all`, Docker likely started before the GPU registry was populated (common post-reboot). Restart the user-level Docker service:
```bash
systemctl --user restart docker
```
*Preventative Measure:* Configure Docker to wait for the GPU registry automatically:
```bash
mkdir -p ~/.config/systemd/user/docker.service.d
cat > ~/.config/systemd/user/docker.service.d/wait-for-gpu.conf <<'EOF'
[Service]
ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; sleep 1; done; exit 0"
EOF
systemctl --user daemon-reload
```

### 1.2. Acquire the Codebase
```bash
git clone https://github.com/mesfind/ai_weather.git
cd ai_weather/demos/demo3
```

### 1.3. Build the Container Image (One-time, ~1 hour)
```bash
docker build -t demo3 -f docker/Dockerfile .
```
*Note:* The build duration is primarily consumed by compiling advanced AI weather dependencies (e.g., `earth2grid`, `NATTEN`). Subsequent rebuilds leverage Docker's layer cache and complete in seconds unless `docker/locks/` is modified. Model weights are explicitly excluded from the image to maintain portability.

*Optimization:* Transfer a pre-built image from an existing Spark:
```bash
# On the source Spark:
docker save demo3 | gzip > demo3-image.tar.gz
# Transfer via scp/rsync over Tailscale, then on the target Spark:
docker load < demo3-image.tar.gz
```

### 1.4. Initialize the Container
```bash
bash docker/run.sh
```
This launches a self-healing container named `demo3` that persists across reboots. It manages two primary directories:

| Asset Type | Spark Location | Storage Profile |
|---|---|---|
| Saved forecasts, event movies, job logs | `ai_weather/demos/demo3/outputs/` | Grows incrementally with usage |
| Model weights (downloaded on first run) | `~/.cache/demo3/` | Up to ~75 GB total |

**Runtime Configuration Flags** (prepend to `bash docker/run.sh`):
- `DEV=1`: Mounts the local directory into the container, enabling live code updates via `git pull` or direct editing without rebuilding the image. Recommended for active development.
- `DEMO3_CACHE=/path`: Redirects the model weight cache to an alternative location (e.g., a shared high-performance storage volume).
- `DEMO3_IFS_SOURCE=azure`: Routes ECMWF data downloads through the Azure mirror to mitigate AWS throttling.

*Optimization:* Pre-populate weights and outputs from an existing Spark:
```bash
rsync -a --info=progress2 <user>@<spark>:<its weights cache>/ ~/.cache/demo3/
rsync -a --info=progress2 <user>@<spark>:ai_weather/demos/demo3/outputs/ ai_weather/demos/demo3/outputs/
```
*(On `hcwfpgx`, the canonical weights cache is located at `~/e2s-spark/root_cache`)*.

### 1.5. Establish the SSH Tunnel and Access the UI
From your local machine (connected to Tailscale):
```bash
ssh -L 8501:localhost:8501 <user>@<spark>
```
Keep this terminal active and navigate to **http://localhost:8501** in your browser.

### 1.6. Interactive Execution Workflow ("Trying it")
1. **Model Selection:** Choose **Deterministic** or **Probabilistic**, then select a specific model. The UI card displays benchmarked Spark runtimes and known limitations (e.g., NeuralGCM lacks 2m temperature outputs).
2. **Parameterization:** Define the start date (validated against the model's training domain), lead time, ensemble members, geographic region, and specific use case.
3. **Execution:** If a matching forecast exists, **Load saved run** provides instant retrieval. Otherwise, **Run forecast** executes the model on the Spark with a real-time progress indicator.
4. **Analysis:** Explore the result tabs, including the **Event movie** tab for temporal forecast-to-observation comparisons.

*Pedagogical Note:* Computationally intensive models (e.g., Aurora 1.5 at ~26 min, Atlas CRPS at ~19 min for 10-day forecasts) represent advanced high-resolution probabilistic forecasting. Pre-running these is recommended for seamless classroom demonstration.

---

## 2. Use-Case Group Execution

Organize into your designated use-case groups to begin the hands-on exercises. Ensure your team uses the specific flag in the code that tailors the model output to your assigned challenge:

- **Temperature Exceedance:** Led by Docko, supported by Narayana.
- **Precipitation Exceedance (Short-run rainfall):** Led by Koomi, supported by Shruti.
- **Onset/Cessation:** Led by Aryan, supported by Panchali.

---

## 3. Application Interface and Execution Architecture

The Demo 3 Streamlit application serves as the unified interface for orchestrating advanced AI weather models on the DGX Spark. 

**System Status & Capabilities:**
- All 7 integrated models execute live from a unified Docker image, with instant loading for cached runs.
- The event movie pipeline (`demos/demo3/event_movie/forecast_event_movie.py`, developed by Panchali) renders side-by-side forecast and ERA5 observation comparisons.
- Comprehensive result packages include: execution telemetry (load, input, run, save times), geospatial maps (z500, 2m temperature, rainfall), and downloadable NetCDF artifacts.
- Each model card transparently reports hardware requirements: weight size, peak GPU memory, output dimensions, and required input variables.

### 3.1. Core Directory Layout
*(Paths are relative to `ai_weather/demos/demo3/`)*
```text
app.py                 # Main Streamlit application: orchestration and results display
catalog.py             # Model registry (type, environment, lead-time limits, member counts)
regions.py             # Geographic mapping: program countries to padded plotting bounding boxes
contract.py            # Strict data contract: output-file format enforced for all runners
store.py               # Artifact management: storage paths and saved-run deduplication
jobs.py                # Asynchronous job launcher and progress tracker
movie.py               # Background event movie generator for loaded runs
results.py             # Pipeline visualization: model requirements, forecast maps, NetCDF export
event_movie/           # forecast_event_movie.py + Spark adaptation layer
theme.py               # UI styling, aligned with the Demo 5 platform design system
runners/run_model.py   # Universal runner entry point (+ synthetic output generator for testing)
runners/fgn_convert.py # FGN-specific starting condition converter from ECMWF open data
timings.json           # Empirically measured Spark runtimes displayed in the UI
docker/                # Containerization: Dockerfile, run.sh, per-environment dependency locks
outputs/               # Persistent storage: outputs/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
```

### 3.2. Synthetic Testing
To validate the pipeline without consuming GPU resources, enable **Use synthetic output** in the Streamlit sidebar and execute a run. This generates a mock NetCDF file that strictly adheres to the production data contract.

---

## 4. Data Specifications and Model Integration

### 4.1. Output Data Contract
Every model execution produces a single, globally covered NetCDF file. Runners must strictly adhere to this schema:

| Attribute | Specification |
|---|---|
| **dims** | `ensemble, lead_time, lat, lon` (`ensemble` dimension size is 1 for deterministic models) |
| **`lead_time`** | Integer hours post-initialization: 0, 6, 12, … |
| **`lat` / `lon`** | Ascending order; `lon` spans −180 to 180 |
| **`tp`** | Total precipitation accumulated over the preceding 6 hours, in **mm** (NaN at lead 0) |
| **`t2m`** | 2-meter temperature, in **K** |
| **`z500`** | 500 hPa geopotential, in **m² s⁻²** (divide by 9.80665 to derive geopotential height in meters) |
| **coords** | `valid_time` (aligned with `lead_time`), `init_time` |
| **attrs** | `model`, `model_name`, `init_source`, `members`, `synthetic` |

### 4.2. Event Movie Pipeline
The visualization pipeline reads Demo 3 output files and dynamically fetches ERA5 observations from Google's ARCO dataset (cached locally in `outputs/_obs_cache/` if the cluster's primary ERA5 volume is unavailable). The selected use case dictates the movie variable: **Heat** (daily max 2m temperature) or **Precipitation** (daily rainfall accumulation). Movies are rendered on-demand and cached in `outputs/_movies/`.

### 4.3. FGN (WeatherNext 2) Integration Specifics
FGN (Functional Generative Network) is Google DeepMind's advanced approach for probabilistic forecasting, injecting noise directly into the generative process. As it is not yet natively supported in NVIDIA Earth2Studio, and Google publishes FGN-ready inputs for a single date (2024-10-07 00Z), Demo 3 implements a specialized workflow for Google's 1° Mini model (`WeatherNextCyclones_Mini`), optimized for constrained memory environments while maintaining cyclone tracking fidelity.

- **Converter (`runners/fgn_convert.py`):** Executes in the `e2s018` environment. Downloads ECMWF IFS analyses (step 0) for the target time and T-6h at FGN's 13 pressure levels, regrids to FGN's 1° resolution, and constructs Google's expected file layout. Valid for any date covered by ECMWF open data (from 2024-03-01).
  - *Approximations:* Sea-surface temperature uses IFS skin temperature over oceans, floored at seawater freezing (271.46 K) under sea ice. Surface geopotential and land-sea masks are fixed fields copied from Google's sample. Validation against Google's 2024-10-07 file shows a correlation of 1.000 for all fields except SST (correlation 0.997, mean bias -0.08 K).
- **Runner (`runners/runner_fgn.py`):** Utilizes Google's sample file for 2024-10-07 (up to 7.5 days); invokes the converter for all other dates. Downloads weights (1.1 GB) to `DEMO3_FGN_DIR` (default `~/.cache/fgn`) on first run. Caches converted inputs (53 MB/date) in `outputs/_fgn_inputs/`, reducing subsequent run setup time to ~95 seconds. A 10-day, 3-member forecast requires ~2 minutes and 1.8 GB peak GPU memory.

### 4.4. Developer Guide: Adding a New Model
1. Implement `runners/runner_<model_key>.py` containing `run(init, lead_hours, members, report) -> xr.Dataset`, ensuring strict compliance with the output data contract (`tp`, `t2m`, `z500`).
2. Verify the model's Python environment is provisioned at `/opt/envs/<env>/bin/python` within the container.
3. Register the model with `status="ready"` in `catalog.py`.
4. Benchmark the execution and append the measured runtime to `timings.json`.

---

## 5. Performance Reporting and Benchmarks

The following table reports the empirical performance baseline for each model on a DGX Spark (H100-class GPU) for a standard 10-day forecast. These metrics illustrate the architectural trade-offs between spatial resolution, ensemble size, and computational complexity in modern AI weather models.

| Model | Members | Execution Time | Peak GPU Memory |
| --- | --- | --- | --- |
| **AIFS v2 single** | 1 | ~2 min | 14 GB |
| **FGN Mini (1°)** | 3 | ~2 min* | 1.8 GB |
| **NeuralGCM (2.8°)** | 3 | ~2.5 min | 18 GB |
| **GraphCast** | 1 | ~3.5 min | 16 GB |
| **AIFS v2 ENS** | 3 | ~5.5 min | 25 GB |
| **Atlas CRPS** | 3 | ~19 min | 33 GB |
| **Aurora 1.5** | 1 | ~26 min | 27 GB |

*\* **FGN Note:** Time is extrapolated from a measured 106 seconds for Google’s 7.5-day sample case.*

**Benchmarking Context:**
- **Initialization Overhead:** Reported times include the download of starting conditions (from Google’s ERA5 copy or ECMWF), except for FGN, which utilizes the pre-packaged sample file. Model weights are assumed to be pre-cached. First-time executions incur additional latency for weight downloads (e.g., ~31 min for Atlas CRPS, ~7 min for Aurora 1.5, ~5 min for AIFS v2 ENS).
- **Aurora 1.5:** Microsoft's open foundation model for the Earth system. The extended runtime reflects its high-resolution, multi-variable atmospheric and environmental modeling capabilities.
- **Atlas CRPS:** NVIDIA's advanced ensemble prognostic model. The higher memory footprint and runtime are attributable to its noise-conditioned transformer blocks sharing the Atlas autoencoder, designed for robust probabilistic dispersion.
- **AIFS v2:** ECMWF's operational machine-learning ensemble. Represents the current operational standard for medium-range global forecasting, balancing speed and accuracy, with v3 architectures slated to introduce hourly surface forecasts.
- **FGN (WeatherNext 2):** Demonstrates highly efficient, scalable probabilistic forecasting. The Mini variant achieves rapid cyclone tracking with minimal memory overhead, validating the efficacy of the Functional Generative Network architecture.

---
### Everyday Maintenance Commands

```bash
docker logs -f demo3                 # Monitor real-time application logs
docker restart demo3                 # Gracefully restart the application container
git pull                             # Fetch latest code (takes effect immediately if DEV=1; otherwise requires rebuild)
docker build -t demo3 -f docker/Dockerfile . && docker rm -f demo3 && bash docker/run.sh  # Full rebuild and restart cycle
```

### Rapid Troubleshooting Reference

| Symptom | Resolution |
|---|---|
| `docker run` fails: `unresolvable CDI devices` | Execute `systemctl --user restart docker`, then `docker start demo3`. Implement the `wait-for-gpu.conf` fix (Step 1.1) for permanence. |
| `docker: Conflict ... name "/demo3" is already in use` | A stale container exists. Resolve via `docker start demo3`, or forcefully remove and recreate: `docker rm -f demo3` followed by `bash docker/run.sh`. |
| Browser cannot reach `localhost:8501` | Verify the SSH tunnel is active. If port 8501 is occupied locally, remap the tunnel: `ssh -L 8502:localhost:8501 <user>@<spark>` and navigate to `:8502`. |
| Run fails during starting condition download | ECMWF/Google mirrors may be throttled. Retry the execution, or enforce the Azure mirror via `DEMO3_IFS_SOURCE=azure`. |
| First model run is exceptionally slow | This is expected behavior as the model downloads its weight files (e.g., Atlas CRPS ~30 min). Subsequent runs will initialize in seconds. |
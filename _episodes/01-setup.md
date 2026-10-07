---
title: "Demo 1"
teaching: 30
exercises: 15
questions:
- "What are the essential components of an AI weather forecasting environment?"
- "How do we verify GPU availability and configure containerized workflows?"
- "What libraries and dependencies are required for running modern AI weather models?"
objectives:
- "Understand the environmental requirements for AI weather forecasting on DGX Spark infrastructure."
- "Verify GPU visibility and configure Docker with NVIDIA Container Device Interface (CDI)."
- "Identify essential libraries for AI weather model execution and deployment."
- "Troubleshoot common setup issues related to GPU access and container initialization."
keypoints:
- "GPU visibility must be verified before running AI weather models in containers."
- "Docker rootless mode with NVIDIA CDI provides secure GPU passthrough for AI workloads."
- "Essential libraries include earth2grid, NATTEN, and model-specific dependencies."
- "Pre-built container images can significantly reduce setup time for new deployments."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Setting Up AI Weather Forecasting Lab

This lesson documents the complete environmental setup for AI weather forecasting on DGX Spark infrastructure. It covers GPU verification, container configuration, and essential library dependencies required for running modern AI weather models (AIFS, GraphCast, Aurora, Atlas CRPS, FGN/WeatherNext, NeuralGCM, and FuXi).

## Learning Objectives

By the end of this lesson, you will be able to:
- Verify GPU availability using NVIDIA Container Device Interface (CDI)
- Configure Docker rootless mode with proper GPU passthrough
- Build and deploy containerized AI weather forecasting environments
- Identify and resolve common setup issues related to GPU access
- Understand the library dependencies for different AI weather models

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Why Environment Setup Matters | The importance of reproducible AI forecasting environments |
| GPU Verification | Confirming GPU visibility via CDI |
| Docker Configuration | Setting up rootless Docker with GPU support |
| Container Image Management | Building, caching, and transferring container images |
| Essential Libraries | Understanding dependencies for AI weather models |
| Troubleshooting | Resolving common setup issues |
| Exercises | Hands-on practice with environment verification |

## Why Environment Setup Matters

AI weather forecasting models require specialized hardware (GPUs) and software environments to run efficiently. Unlike traditional weather models that run on CPUs, modern AI models like GraphCast, Aurora, and AIFS leverage GPU acceleration for:

- **Fast inference**: 10-day global forecasts in minutes instead of hours
- **Ensemble generation**: Running multiple forecast scenarios simultaneously
- **High-resolution predictions**: 0.25° grid resolution with complex neural architectures

A reproducible environment ensures that:
- Results can be replicated across different DGX Spark nodes
- Model weights and dependencies are consistent
- GPU resources are properly allocated and monitored



## 1. Environment Setup

### 1.1 Access Pathways

**Path A: Pre-configured Spark**
Use a Spark where Demo 3 is already deployed (e.g., `hcwfpgx`). Requires Tailscale access and a user account.

**Path B: Custom Spark Setup**
Execute the full setup sequence below. Allocate ~1 hour for initial container build.

### 1.2 Core Setup Commands

```bash
# 1. Clone the repository
git clone https://github.com/mesfind/ai_weather.git
cd ai_weather/demos/demo3

# 2. Build the container image (~1 hour first time)
docker build -t demo3 -f docker/Dockerfile .

# 3. Start the container
bash docker/run.sh

# 4. Access the UI from your laptop
ssh -L 8501:localhost:8501 <user>@<spark>
# Then browse to http://localhost:8501
```

:::{warning}
#### First-Run Downloads
The first execution of each model downloads weights (~75 GB total). This can take 30+ minutes for large models like Atlas CRPS. Subsequent runs start in seconds.
:::

### 1.3 Runtime Configuration

Prepend these flags to `bash docker/run.sh`:

| Flag | Purpose | Example |
| --- | --- | --- |
| `DEV=1` | Mount local code for live development | `DEV=1 bash docker/run.sh` |
| `DEMO3_CACHE=/path` | Redirect model weight cache | `DEMO3_CACHE=/shared/cache bash docker/run.sh` |
| `DEMO3_IFS_SOURCE=azure` | Use Azure mirror for ECMWF data | `DEMO3_IFS_SOURCE=azure bash docker/run.sh` |

:::{tip}
#### Development Workflow
Use `DEV=1` during active development to avoid rebuilding the container after every code change. Changes to Python files take effect immediately after a page refresh.
:::


## 2. GPU Verification

The DGX Spark systems run Docker in rootless mode with NVIDIA's Container Device Interface (CDI). This provides secure, user-level GPU access without requiring root privileges.

### 1.1 Verify GPU Visibility

Confirm that Docker can access the GPU by running:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
```

**Expected output:**
```
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-xxxxx)
```

:::{warning}
#### Common Issue: Unresolvable CDI Devices
If the command fails with `unresolvable CDI devices nvidia.com/gpu=all`, Docker started before the GPU registry was populated. This commonly occurs immediately after a system reboot.
:::

**Solution:** Restart the Docker service:

```bash
systemctl --user restart docker
```

Then retry the GPU verification command.

### 2.2 Preventative Configuration

To prevent this issue from recurring after every reboot, configure Docker to wait for the GPU registry:

```bash
mkdir -p ~/.config/systemd/user/docker.service.d
cat > ~/.config/systemd/user/docker.service.d/wait-for-gpu.conf <<'EOF'
[Service]
ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; sleep 1; done; exit 0"
EOF
systemctl --user daemon-reload
```

:::{tip}
#### Why This Works
The configuration adds a pre-start hook that waits up to 120 seconds for the CDI registry file (`/var/run/cdi/nvidia.yaml`) to appear before starting Docker. This ensures GPU devices are properly registered before Docker attempts to use them.
:::

## 3. Docker Configuration

### 3.1 Understanding Rootless Docker

Rootless Docker runs the Docker daemon as a non-root user, providing:
- **Enhanced security**: No root privileges required
- **User isolation**: Each user manages their own containers
- **Simplified permissions**: No need for sudo or group membership

### 3.2 Container Resource Management

When running AI weather models, Docker manages two primary storage locations:

| Asset Type | Location | Size | Purpose |
| --- | --- | --- | --- |
| Saved forecasts, event movies, job logs | `ai_weather/demo3/outputs/` | Grows with use | Persistent results |
| Model weights (downloaded on first run) | `~/.cache/demo3/` | Up to ~75 GB | Model parameters |

:::{keypoints}
#### Storage Best Practices
- Model weights are downloaded once per model and cached locally
- Saved forecasts accumulate over time; consider periodic cleanup
- Use `DEMO3_CACHE=/path` to redirect weights to high-performance storage
:::

## 3. Container Image Management

### 3.1 Building the Container Image

The container image includes all dependencies for running AI weather models:

```bash
docker build -t demo3 -f docker/Dockerfile .
```

**Build characteristics:**
- **First build**: ~1 hour (compiling earth2grid, NATTEN libraries)
- **Subsequent builds**: Seconds (using Docker layer cache)
- **Image contents**: Python environments, dependencies, application code
- **Excluded**: Model weights (downloaded on first run)

:::{note}
#### Build Optimization
Most of the build time is spent compiling two specialized libraries:
- **earth2grid**: Grid interpolation for weather data
- **NATTEN**: Neighborhood attention for transformer architectures

These are compiled from source to ensure compatibility with the target GPU architecture.
:::

### 3.2 Transferring Pre-built Images

To avoid rebuilding on every Spark node, transfer a pre-built image:

```bash
# On the source Spark:
docker save demo3 | gzip > demo3-image.tar.gz

# Transfer via scp/rsync over Tailscale:
scp demo3-image.tar.gz <user>@<target-spark>:~/

# On the target Spark:
docker load < demo3-image.tar.gz
```

:::{exercise}
#### Exercise 1: Image Transfer (5 min)
You have a pre-built image on Spark `hcwfpgx` and need to deploy it to a new Spark `hcwfpgy`. Write the complete sequence of commands to:
1. Export the image from the source Spark
2. Transfer it to the target Spark
3. Load it on the target Spark

<details>
<summary>Solution</summary>

```bash
# On hcwfpgx:
docker save demo3 | gzip > demo3-image.tar.gz
scp demo3-image.tar.gz <user>@hcwfpgy:~/

# On hcwfpgy:
docker load < demo3-image.tar.gz
```

</details>
:::

## 4. Essential Libraries

### 4.1 Core Dependencies

AI weather models require specialized libraries for:

| Library | Purpose | Models Using It |
| --- | --- | --- |
| **earth2grid** | Grid interpolation and coordinate transformations | All models |
| **NATTEN** | Neighborhood attention for efficient transformers | GraphCast, NeuralGCM |
| **xarray** | Multi-dimensional array handling | All models |
| **PyTorch** | Deep learning framework | Aurora, AIFS, FuXi |
| **JAX** | High-performance numerical computing | GraphCast, NeuralGCM |
| **NetCDF4** | Weather data I/O | All models |

### 4.2 Model-Specific Environments

Different models require different Python environments:

| Environment | Models | Key Dependencies |
| --- | --- | --- |
| `e2s018` | Atlas CRPS, Aurora 1.5, AIFS v2 single | PyTorch, Earth2Studio |
| `e2s018ens` | AIFS v2 ENS | PyTorch, ensemble utilities |
| `graphcast` | GraphCast, NeuralGCM | JAX, Graph Neural Networks |
| `fgn` | FGN (WeatherNext 2) | JAX, Functional Generative Networks |

:::{warning}
#### Environment Isolation
Each model runs in its own isolated Python environment (`/opt/envs/<env>/bin/python`). Do not attempt to mix dependencies across environments, as this can cause version conflicts.
:::

## 5. Troubleshooting

### 5.1 Common Issues

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `docker run` fails: `unresolvable CDI devices` | Docker started before GPU registry populated | `systemctl --user restart docker`, then `docker start demo3` |
| `docker: Conflict ... name "/demo3" is already in use` | Stale container exists | `docker start demo3` or `docker rm -f demo3 && bash docker/run.sh` |
| Browser can't reach `localhost:8501` | SSH tunnel not open or port conflict | Use `-L 8502:localhost:8501` and navigate to `:8502` |
| Run fails during starting condition download | ECMWF/Google mirrors throttled | Retry or use `DEMO3_IFS_SOURCE=azure` |
| First model run is exceptionally slow | Downloading model weights (~30 min for Atlas CRPS) | Expected behavior; subsequent runs are fast |

### 5.2 Diagnostic Commands

```bash
# Check container status
docker ps -a | grep demo3

# View container logs
docker logs -f demo3

# Check GPU allocation
nvidia-smi

# Verify Docker service
systemctl --user status docker
```

:::{tip}
#### Proactive Monitoring
Run `docker logs -f demo3` in a separate terminal while testing to catch errors in real-time. This is especially useful during first-time model runs when weights are being downloaded.
:::

## 6. Best Practices

### 6.1 Environment Setup Checklist

Before running AI weather models, verify:

- [ ] GPU is visible to Docker (`nvidia-smi -L` works in container)
- [ ] Docker service is running (`systemctl --user status docker`)
- [ ] Container image is built or loaded (`docker images | grep demo3`)
- [ ] Sufficient disk space for model weights (~75 GB)
- [ ] SSH tunnel is active (if accessing remotely)
- [ ] Network connectivity to ECMWF/Google data sources

### 6.2 Performance Optimization

- **Use pre-built images**: Transfer images between Sparks instead of rebuilding
- **Cache model weights**: Use `DEMO3_CACHE` to point to shared storage
- **Pre-run heavy models**: Aurora 1.5 (~26 min) and Atlas CRPS (~19 min) should be pre-run for classroom demonstrations
- **Monitor GPU utilization**: Use `nvidia-smi` to ensure models are using GPU acceleration

## Summary

- GPU visibility must be verified before running AI weather models in containers
- Docker rootless mode with NVIDIA CDI provides secure GPU passthrough
- Container images encapsulate all dependencies for reproducible deployments
- Model weights are downloaded once and cached locally
- Essential libraries include earth2grid, NATTEN, and model-specific frameworks
- Troubleshooting common issues requires understanding Docker, GPU, and network interactions

## Exercises

:::{exercise}
#### Exercise 2: Diagnose GPU Access (10 min)
You SSH into a DGX Spark after a reboot and try to run a container, but get the error:
```
docker: Error response from daemon: unresolvable CDI devices nvidia.com/gpu=all
```

1. What caused this error?
2. What command do you run to fix it immediately?
3. What configuration change prevents this from happening after future reboots?

<details>
<summary>Solution</summary>

1. **Cause**: Docker started before the NVIDIA CDI registry was populated after the reboot.
2. **Immediate fix**: `systemctl --user restart docker`
3. **Prevention**: Create `/etc/systemd/user/docker.service.d/wait-for-gpu.conf` with the ExecStartPre hook that waits for `/var/run/cdi/nvidia.yaml`.

</details>
:::

:::{exercise}
#### Exercise 3: Container Management (5 min)
You need to update the code in your running container without rebuilding the entire image. What environment variable do you set, and what command do you run?

<details>
<summary>Solution</summary>

Set `DEV=1` to mount the local directory into the container:

```bash
DEV=1 bash docker/run.sh
```

This allows you to edit code locally and see changes immediately without rebuilding.

</details>
:::



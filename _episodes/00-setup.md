---
title: Setup
teaching: 15
exercises: 3
questions:
- "What is a virtual environment and how does it manage dependencies?"
- "How can Docker containers simplify environment setup and reproducibility?"
- "What are the capabilities of NVIDIA DGX Spark for climate and AI workflows?"
- "How do Spark, Cloud, and HPC systems compare for different computational tasks?"
objectives:
- "Learn to create and manage Python virtual environments using Conda and venv."
- "Understand how to use Docker for reproducible, dependency-isolated deployments."
- "Identify the appropriate hardware (Spark, Cloud, or HPC) for specific climate modeling tasks."
keypoints:
- "Virtual environments and Docker isolate dependencies to prevent version conflicts."
- "NVIDIA DGX Spark excels at local prototyping, blending forecasts, and training small bias correction models."
- "Large-scale reforecast data generation is best suited for traditional HPC systems due to massive I/O and parallel compute needs."
---

# Python Environment Management

Managing Python environments is essential for keeping project dependencies isolated and avoiding version conflicts. Python provides several tools for managing environments, including **Conda** and the built-in **venv** module. Both help to create isolated environments for Python projects, each with its own set of dependencies.

## Conda Environments

Conda is a package, dependency, and environment management tool that simplifies managing different Python versions and libraries. It is especially popular in the scientific community and on Windows platforms.

### What is a Conda Environment?

A Conda environment is a self-contained directory that contains a specific collection of Conda packages. These environments help isolate different projects from each other, ensuring that dependencies for one project do not interfere with others.

> **Best Practice**
> 
> Avoid installing packages into the base Conda environment. Create a new environment for each project to maintain strict isolation.
{: .tip}

### Creating Environments with Conda (Linux/macOS)

To create a new Conda environment, use the following command:

```bash
admin@MacBook~ $ conda create -n <env_name> python=<version#>
```

For example, to create a Conda environment named `ai-onset` with Python 3.12:

```bash
admin@MacBook~ $ conda create --name ai-onset python=3.12
```

To activate the environment:

```bash
admin@MacBook~ $ conda activate ai-onset
```

### Installing Packages in Conda Environments

Install packages within an environment using `conda` or `pip`:

```bash
admin@MacBook~ $ conda install <package_name>
```

Or, if the package is not available via Conda:

```bash
admin@MacBook~ $ pip install <package_name>
```

### Exporting and Sharing Environments

Export an environment configuration to a file for sharing with colleagues:

```bash
admin@MacBook~ $ conda env export --no-builds --file environment.yaml
```

This creates a `YAML` file that lists all packages and dependencies used in the environment. Others can use this file to recreate the exact same environment.

### Deactivating and Removing Environments

To deactivate a Conda environment:

```bash
admin@MacBook~ $ conda deactivate
```

To remove an environment entirely:

```bash
admin@MacBook~ $ conda env remove --name <env_name>
```

---

## Virtual Environments with `venv`

The `venv` module, included in Python by default, is another tool for creating isolated environments. While Conda is a more comprehensive solution, `venv` is lightweight and works well for basic Python projects.

### Creating a Virtual Environment with `venv`

1. **Open a Terminal**: On Linux/macOS, press `Ctrl + Alt + T` or search for `Terminal`.
2. **Navigate to the Desired Directory**: Use `cd` to move to the project folder:
   ```bash
   admin@MacBook~ $ cd Documents/ai-onset
   ```
3. **Create the Virtual Environment**:
   ```bash
   admin@MacBook~ $ python3 -m venv ai-onset
   ```
4. **Activate the Virtual Environment** (Linux/macOS):
   ```bash
   admin@MacBook~ $ source ai-onset/bin/activate
   ```
5. **Install Packages**: Install dependencies with `pip`:
   ```bash
   admin@MacBook~ $ pip install <package_name>
   ```
6. **Deactivate the Virtual Environment**: When done, exit the environment:
   ```bash
   admin@MacBook~ $ deactivate
   ```

---

## Containerized Workflows with Docker

For complex scientific workflows, managing dependencies across different machines can be challenging. **Docker** provides a solution by packaging the application and all its dependencies into a standardized, isolated container. This ensures that the environment runs identically on any machine that supports Docker, eliminating the "it works on my machine" problem.

### Quick Start: 2-Line Installation Test

Participants can spin up a pre-configured Docker container and verify the installed libraries in just two lines of code:

```bash
admin@MacBook~ $ docker run -it --gpus all climate-workshop:latest /bin/bash
admin@MacBook~ $ python -c "import xarray, dask, cartopy; print('Environment ready!')"
```

*(Note: Replace `climate-workshop:latest` with your organization's actual container registry path.)*

---

## NVIDIA DGX Spark for Climate Workflows

The **NVIDIA DGX Spark** is a desktop AI supercomputer powered by the GB10 Grace Blackwell superchip, featuring 128 GB of unified memory and 1 PetaFLOP of parallel throughput. It is designed to bring datacenter-capable AI, machine learning, and data science workflows directly to your desk.

### What is Possible with a Spark?

- **Ideal For**: Running benchmarks, blending forecasts, and training small bias correction models locally, *provided you already have your reforecast data*.
- **Not Ideal For**: Large-scale reforecast data generation. This task requires massive I/O throughput and distributed parallelism that are better suited for traditional HPC clusters.

For practical guides, playbooks, and interactive options to get started, visit the official [NVIDIA Build: DGX Spark](https://build.nvidia.com/spark) portal. You can also explore the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/common-use-cases.html) for common use cases.

> **Hardware Comparison: Spark vs. Cloud vs. HPC**
> 
> For guidance on optimizing your workflow across these platforms, refer to internal hardware comparison resources alongside the official NVIDIA documentation.
{: .info}

| Feature | NVIDIA DGX Spark | Cloud (e.g., AWS, GCP) | Traditional HPC |
| :--- | :--- | :--- | :--- |
| **Best Use Case** | Local prototyping, bias correction, forecast blending | Scalable burst workloads, collaborative projects | Large-scale reforecast data generation, massive parallelism |
| **Memory/Compute** | 128 GB unified memory, 1 PetaFLOP throughput | Highly scalable, variable by instance type | Massive, distributed across compute nodes |
| **Data Transfer** | None (local data access) | High egress costs for large climate datasets | Managed within the facility's high-speed network |
| **Accessibility** | Immediate, interactive, no queueing | Immediate, but requires setup and billing config | Requires job scheduling (e.g., Slurm) and queueing |
| **Cost Model** | Upfront hardware cost, no ongoing compute fees | Pay-as-you-go, can become expensive for long runs | Institutional funding, allocation-based |

---

## Exercises

> **Exercise 1: Create and Use a `venv` Environment**
> 
> 1. Create a new virtual environment named `myproject`.
> 2. Activate it and install the `requests` library.
> 3. Create a file `fetch_page.py` with the following content:
>    ```python
>    import requests
>    response = requests.get('https://www.example.com')
>    print(response.status_code)
>    ```
> 4. Run the script (`python fetch_page.py`) and then deactivate the environment.
{: .exercise}

> **Exercise 2: Conda Environment Setup on Windows (PowerShell)**
> 
> 1. Verify Conda installation: `conda --version`
> 2. Create a new environment named `ml-flow` with Python 3.10:  
>    `conda create --name ml-flow python=3.10`
> 3. Activate it: `conda activate ml-flow`
> 4. Install packages: `conda install numpy matplotlib`
> 5. Verify installation: `conda list`
> 6. Export the configuration: `conda env export --no-builds --file ml-flow.yaml`
> 7. Clean up: `conda deactivate` followed by `conda env remove --name ml-flow`
{: .exercise}

> **Exercise 3: Docker & Spark Quick Start**
> 
> 1. Ensure Docker is installed and running on your machine.
> 2. Run the 2-line Docker test provided in the "Containerized Workflows" section above.
> 3. Visit [build.nvidia.com/spark](https://build.nvidia.com/spark) and identify one playbook or use case that aligns with your current research or project goals.
{: .exercise}

---

## Appendix: Reference `environment.yaml`

Below is the complete `workshop` environment configuration used for the Docker container, showcasing the comprehensive climate and geospatial Python stack:

<details>
<summary>Click to expand <code>environment.yaml</code></summary>

```yaml
name: workshop
channels:
  - conda-forge
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
dependencies:
  - python=3.13.2
  - numpy=2.2.6
  - pandas=2.3.2
  - xarray=2025.9.0
  - dask=2025.9.0
  - cartopy=0.25.0
  - geopandas=1.1.1
  - netcdf4=1.7.2
  - h5py=3.14.0
  - esmpy=8.9.0
  - xesmf=0.8.10
  - scikit-learn=1.7.1
  - matplotlib=3.10.6
  - jupyterlab=4.4.7
  - pip
  - pip:
      - cfgrib==0.9.15.0
      - rasterio==1.4.3
      - requests==2.32.5
      - bokeh==3.8.0
      - panel==1.8.1
prefix: /opt/conda/envs/workshop
```
</details>
```
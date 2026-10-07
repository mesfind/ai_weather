---
title: "Demo 3"
teaching: 45
exercises: 30
questions:
- "How do we set up and run AI weather models locally in a container?"
- "What are the core commands needed to go from environment setup to generating a forecast figure?"
- "How do we tailor the model execution to specific use cases like onset, cessation, or temperature exceedance?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Execute a streamlined 4-command workflow to build a container, run a model, and generate use-case-specific outputs."
- "Interpret model outputs and compare forecasts against observations."
keypoints:
- "A streamlined 4-command workflow (build container, run model, get output, generate figure) simplifies local AI forecasting."
- "Use-case flags in the code allow groups to tailor outputs for onset/cessation, temperature exceedance, or precipitation exceedance."
- "Streamlit provides an interactive environment for executing and visualizing the AI weather models."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Running AI Weather Forecast Models on a DGX Spark

This module provides a comprehensive guide to deploying, executing, and benchmarking modern AI weather forecasting models locally using containerized workflows on a DGX Spark infrastructure. The Demo 3 Streamlit application serves as the unified interface for orchestrating seven state-of-the-art AI weather models.

## Learning Objectives

By the end of this lesson, you will be able to:
- Deploy and configure the Demo 3 Streamlit application on a DGX Spark
- Execute AI weather models (AIFS, GraphCast, Aurora, Atlas CRPS, FGN, NeuralGCM, FuXi) in containerized environments
- Generate use-case-specific forecasts (temperature exceedance, precipitation exceedance, onset/cessation)
- Compare model forecasts against ERA5 observations using event movies
- Interpret model performance metrics and output formats

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Application Overview | Understanding the Demo 3 architecture and capabilities |
| Environment Setup | Deploying the containerized application |
| Model Execution | Running forecasts with the Streamlit interface |
| Output Interpretation | Understanding NetCDF outputs and visualization |
| Use-Case Customization | Tailoring models for specific forecasting challenges |
| Model Integration | Adding new models to the framework |
| Performance Benchmarks | Comparing model execution times and resource usage |
| Exercises | Hands-on practice with model execution |

## Application Overview

### System Architecture

The Demo 3 application consists of:

- **Streamlit UI**: Interactive web interface for model selection and parameterization
- **Docker Container**: Isolated environment with all dependencies
- **Model Runners**: Specialized execution scripts for each AI model
- **Event Movie Pipeline**: Visualization comparing forecasts to observations
- **Output Storage**: NetCDF files with standardized data contracts

### Supported Models

| Model | Type | Resolution | Ensemble Members | Key Characteristics |
| --- | --- | --- | --- | --- |
| **AIFS v2 single** | Deterministic | 0.25° | 1 | ECMWF operational, fast inference |
| **AIFS v2 ENS** | Probabilistic | 0.25° | 3-50 | Ensemble forecasts, uncertainty quantification |
| **GraphCast** | Deterministic | 0.25° | 1 | Google DeepMind, graph neural network |
| **NeuralGCM** | Hybrid | 2.8° | 3 | Google, combines ML with physics |
| **Aurora 1.5** | Foundation | 0.25° | 1 | Microsoft, multi-variable Earth system |
| **Atlas CRPS** | Probabilistic | 0.25° | 3 | NVIDIA, noise-conditioned transformers |
| **FGN Mini** | Probabilistic | 1° | 3 | Google WeatherNext 2, cyclone tracking |

:::{note}
#### Model Selection Guidance
- **Fast exploration**: AIFS v2 single (~2 min), FGN Mini (~2 min)
- **Probabilistic forecasting**: AIFS v2 ENS, Atlas CRPS
- **High accuracy**: Aurora 1.5, GraphCast (but slower: 26 min and 3.5 min respectively)
:::


## 1. Model Execution

### 1.1 Interactive Workflow

1. **Model Selection**: Choose Deterministic or Probabilistic, then select a specific model
2. **Parameterization**: Set start date, lead time, ensemble members, geographic region, and use case
3. **Execution**: Click "Run forecast" (or "Load saved run" if available)
4. **Analysis**: Explore results tabs, including event movies comparing forecasts to observations

### 1.2 Use-Case Groups

Organize into your designated use-case groups to begin the hands-on exercises. Ensure your team uses the specific flag in the code that tailors the model output to your assigned challenge:

- **Temperature Exceedance**: Led by Docko, supported by Narayana
- **Precipitation Exceedance (Short-run rainfall)**: Led by Koomi, supported by Shruti
- **Onset/Cessation**: Led by Aryan, supported by Panchali

:::{exercise}
#### Exercise 1: Run Your First Forecast (15 min)
1. Select the AIFS v2 single model
2. Choose a start date within the model's valid range
3. Set lead time to 5 days
4. Select your country and use case
5. Run the forecast and explore the results

**Questions:**
- How long did the forecast take to run?
- What variables are shown in the output maps?
- How does the forecast compare to ERA5 observations in the event movie?

<details>
<summary>Discussion Points</summary>

- AIFS v2 single typically runs in ~2 minutes for a 10-day forecast
- Output includes z500 (geopotential), t2m (2m temperature), and tp (precipitation)
- The event movie shows day-by-day comparison; look for systematic biases (e.g., temperature too warm/cool)

</details>
:::

## 2. Output Interpretation

### 2.1 Data Contract

Every model execution produces a NetCDF file with the following structure:

| Attribute | Specification |
| --- | --- |
| **dims** | `ensemble, lead_time, lat, lon` (ensemble size = 1 for deterministic) |
| **lead_time** | Integer hours: 0, 6, 12, … |
| **lat / lon** | Ascending; lon spans −180 to 180 |
| **tp** | Total precipitation (mm), accumulated over 6h (NaN at lead 0) |
| **t2m** | 2-meter temperature (K) |
| **z500** | 500 hPa geopotential (m² s⁻²); divide by 9.80665 for height in meters |
| **coords** | `valid_time`, `init_time` |
| **attrs** | `model`, `model_name`, `init_source`, `members`, `synthetic` |

### 2.2 Loading Outputs in Python

```python
import xarray as xr

# Load a forecast
ds = xr.open_dataset("outputs/aifs_single/20241007T00_240h_m1.nc")

# Access 2m temperature at lead time 24h
t2m_24h = ds["t2m"].sel(lead_time=24)

# Convert geopotential to height (meters)
z500_height = ds["z500"] / 9.80665

print(ds)
```

:::{keypoints}
#### Output Format Essentials
- All models produce the same NetCDF structure for consistency
- Precipitation is accumulated over 6-hour windows
- Geopotential must be divided by gravity (9.80665) to get height in meters
- Ensemble dimension has size 1 for deterministic models
:::

## 3. Use-Case Customization

### 3.1 Temperature Exceedance

Focus: Heatwave tracking and daily maximum 2m temperature thresholds

**Key parameters:**
- Variable: `t2m` (2-meter temperature)
- Metric: Daily maximum temperature
- Threshold: Region-specific (e.g., 35°C for heatwave conditions)

### 3.2 Precipitation Exceedance

Focus: Short-run rainfall accumulation and flood risk mapping

**Key parameters:**
- Variable: `tp` (total precipitation)
- Metric: Accumulated rainfall over 24-72 hours
- Threshold: Region-specific (e.g., 50mm/day for heavy rainfall)

### 3.3 Onset/Cessation

Focus: Seasonal transition markers and agricultural climate indices

**Key parameters:**
- Variable: `tp` (precipitation)
- Metric: Wet-spell and dry-spell detection (see Demo 4 for details)
- Application: Rainy season onset timing

:::{tip}
#### Use-Case Flags
Each use case activates specific visualization pipelines and metric calculations. The Streamlit interface automatically adjusts the output based on your selection.
:::

## 4. Model Integration

### 4.1 Adding a New Model

To integrate a new AI weather model:

1. **Create runner script**: `runners/runner_<model_key>.py`
   ```python
   def run(init, lead_hours, members, report) -> xr.Dataset:
       # Generate forecast
       # Return Dataset with tp, t2m, z500
       pass
   ```

2. **Provision environment**: Ensure `/opt/envs/<env>/bin/python` exists in the container

3. **Register model**: Set `status="ready"` in `demo3/catalog.py`

4. **Benchmark**: Add measured runtime to `timings.json`

:::{warning}
#### Data Contract Compliance
New models must strictly adhere to the output format (tp in mm, t2m in K, z500 in m² s⁻²). Non-compliant outputs will break visualization pipelines.
:::

## 5. Performance Benchmarks

### 5.1 Execution Times (10-day forecast)

| Model | Members | Execution Time | Peak GPU Memory |
| --- | --- | --- | --- |
| AIFS v2 single | 1 | ~2 min | 14 GB |
| FGN Mini (1°) | 3 | ~2 min | 1.8 GB |
| NeuralGCM (2.8°) | 3 | ~2.5 min | 18 GB |
| GraphCast | 1 | ~3.5 min | 16 GB |
| AIFS v2 ENS | 3 | ~5.5 min | 25 GB |
| Atlas CRPS | 3 | ~19 min | 33 GB |
| Aurora 1.5 | 1 | ~26 min | 27 GB |

:::{note}
#### Performance Trade-offs
- **Speed vs. Accuracy**: Faster models (AIFS, FGN) sacrifice some accuracy for rapid inference
- **Resolution vs. Memory**: Higher resolution (0.25°) requires more GPU memory
- **Deterministic vs. Probabilistic**: Ensemble models provide uncertainty but take longer
:::

### 5.2 Initialization Overhead

First-time executions include weight downloads:
- Atlas CRPS: ~31 min
- Aurora 1.5: ~7 min
- AIFS v2 ENS: ~5 min

Subsequent runs start in seconds.

## 6. FGN (WeatherNext 2) Integration

FGN is Google DeepMind's Functional Generative Network for probabilistic forecasting. It requires special handling:

### 6.1 Starting Conditions

FGN is not in Earth2Studio, and Google publishes FGN-ready inputs for only one date (2024-10-07 00Z). For other dates, Demo 3 uses a converter:

```bash
# Converter runs in e2s018 environment
python runners/fgn_convert.py --date 2024-10-08
```

**Converter workflow:**
1. Downloads ECMWF IFS analyses at 13 pressure levels
2. Regrids to FGN's 1° resolution
3. Constructs Google's expected file layout

:::{warning}
#### Sea Surface Temperature Approximation
The converter uses IFS skin temperature over oceans, floored at 271.46 K under sea ice. This introduces a small bias (correlation 0.997, mean error -0.08 K) compared to true SST.
:::

### 6.2 Performance

- **10-day, 3-member forecast**: ~2 minutes, 1.8 GB peak memory
- **Cached inputs**: 53 MB per start date, reused for subsequent runs

## Summary

- Demo 3 provides a unified interface for seven AI weather models
- Containerized deployment ensures reproducibility across DGX Spark nodes
- The 4-command workflow (build, run, get output, generate figure) simplifies execution
- Use-case flags tailor outputs for temperature, precipitation, and onset forecasting
- All models produce standardized NetCDF outputs for consistent analysis
- Performance varies from 2 minutes (AIFS) to 26 minutes (Aurora) for 10-day forecasts

## Exercises

:::{exercise}
#### Exercise 2: Compare Deterministic vs. Probabilistic (10 min)
Run both AIFS v2 single (deterministic) and AIFS v2 ENS (probabilistic, 3 members) for the same start date and lead time.

**Questions:**
1. How do the execution times compare?
2. What additional information does the ensemble provide?
3. How would you use the ensemble spread to assess forecast confidence?

<details>
<summary>Discussion Points</summary>

1. AIFS v2 single: ~2 min; AIFS v2 ENS: ~5.5 min (2.75x slower for 3 members)
2. The ensemble provides uncertainty estimates; you can compute spread (standard deviation across members)
3. Large spread indicates low confidence; small spread suggests high confidence. Compare spread to historical verification to assess reliability.

</details>
:::

:::{exercise}
#### Exercise 3: Interpret Model Outputs (10 min)
Load a forecast NetCDF file and answer:

```python
import xarray as xr
ds = xr.open_dataset("outputs/graphcast/20241007T00_120h_m1.nc")
```

1. What are the dimensions and coordinates?
2. Extract the 500 hPa geopotential height (in meters) at lead time 48h
3. What is the total precipitation accumulated between lead times 24h and 48h?

<details>
<summary>Solution</summary>

```python
# 1. Dimensions and coordinates
print(ds.dims)  # {'ensemble': 1, 'lead_time': 21, 'lat': 721, 'lon': 1440}
print(ds.coords)

# 2. Geopotential height at 48h
z500_48h = ds["z500"].sel(lead_time=48)
z500_height = z500_48h / 9.80665  # Convert to meters

# 3. Precipitation accumulation (24h to 48h)
# Note: tp is accumulated over 6h windows, so sum the relevant time steps
tp_24_48 = ds["tp"].sel(lead_time=slice(24, 48)).sum(dim="lead_time")
```

</details>
:::

:::{exercise}
#### Exercise 4: Use-Case Group Activity (10 min)
In your assigned use-case group:

1. Run forecasts for your specific challenge (temperature, precipitation, or onset)
2. Generate event movies comparing forecasts to observations
3. Identify one systematic bias in the model output
4. Prepare a 2-minute presentation on your findings

**Deliverables:**
- Screenshot of the event movie
- Description of the bias observed
- Hypothesis for why the bias occurs

:::

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Container won't start: "unresolvable CDI devices" | `systemctl --user restart docker && docker start demo3` |
| Port 8501 already in use | Use `ssh -L 8502:localhost:8501` and navigate to `:8502` |
| Model run fails during download | Retry or use `DEMO3_IFS_SOURCE=azure` |
| Out of memory errors | Check GPU memory with `nvidia-smi`; consider smaller lead times or fewer members |

## Next Steps

After completing this demo, proceed to **Demo 4** to learn systematic benchmarking of AI weather models against local observations for rainy season onset detection.
```

Both files now match the professional structure of `04-weather-model.md` with:
- Proper YAML frontmatter
- MathJax support
- Clear section organization
- Callout blocks (tip, warning, exercise, solution, keypoints, note)
- Comprehensive tables
- Hands-on exercises with solutions
- Learning objectives and roadmaps
- Professional formatting throughout
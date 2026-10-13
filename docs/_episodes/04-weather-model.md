---

title: "Demo 4"
teaching: 35
exercises: 60
questions:

* "How do we systematically benchmark AI weather models against local observations for rainy season onset across different regions and agricultural contexts?"
* "How do we configure, execute, and troubleshoot the ROMP/MOMP pipeline using country-specific specification files (`benchmark_spec.json`)?"
* "How do deterministic and probabilistic evaluation tracks differ, and why must they remain strictly separated?"
* "How do we prepare spatial rainfall masks, observation catalogs, and administrative boundaries for a new national or sub-national domain?"
objectives:
* "Explain the four comparison dimensions of ROMP (location, onset thresholds, lead time, model) and how to adapt them to any agricultural region."
* "Construct and apply a JSON specification file (`benchmark_spec.json`) to run benchmarks without altering core pipeline code."
* "Configure and execute deterministic (MAE, FAR, Miss Rate) and probabilistic (BS, RPS, AUC, Reliability) evaluation tracks."
* "Prepare observational data catalogs, spatial rain masks (`nc_mask`), and domain shapefiles for new target countries."
* "Interpret skill score heatmaps, spatial maps, and reliability diagrams relative to a climatological reference."
* "Diagnose and resolve common data, domain, path, and configuration failure modes."
keypoints:
* "Onset is derived identically from observations and forecasts using wet-spell/dry-spell rules tailored to local cropping systems."
* "Specification files (`benchmark_spec.json`) decouple experiment setup from code, making benchmarks reproducible and transferrable across countries."
* "Deterministic and probabilistic tracks use non-comparable metric families and must never be mixed within a single evaluation run."
* "Skill is always relative to a reference (e.g., climatology) and depends on explicit choices (thresholds, tolerance, grid resolution, and search window)."

---

# The AI Weather Model Scorecard: Multi-Country Benchmarking

This lesson documents the complete ROMP (Rainy season Onset Metrics Package) / MOMP benchmarking workflow. Developed as part of the **AI4AG Initiative** and the **Human-Centered Weather Forecasts Initiative at the University of Chicago**, this framework evaluates state-of-the-art AI weather forecast models (AIFS, FuXi, GraphCast, GenCast, AIFS-ENS) against observational rainfall datasets (e.g., CHIRPS).

While earlier examples focused on Ethiopia's *Kiremt* season, the pipeline is fully region- and threshold-agnostic. This lesson teaches you how to adapt, configure, and execute benchmarks for any country or agricultural season globally.

> #### Prerequisites
> 
> 
> * Complete the environment setup in [Demo 1: Setting Up AI Weather Forecasting Lab](https://www.google.com/search?q=/ai_weather/01-setup/) before proceeding.
> * Ensure ROMP is installed (`uv pip install -e .` or `uv sync`).
> * Familiarity with the Streamlit interface (`app.py`) and NetCDF data structures.
> {: .prereq}
> 
> 

## Learning Objectives

By the end of this lesson, you will be able to:

* Explain the motivation for the package and the four dimensions it compares: location, wet-spell threshold, dry-spell threshold, and lead time.
* Construct and apply a JSON specification file (`benchmark_spec.json`) to run benchmarks for any country without modifying pipeline code.
* Configure and execute deterministic benchmarks (MAE, FAR, Miss Rate).
* Configure and execute probabilistic benchmarks (Brier Score, Ranked Probability Score, AUC, Reliability).
* Interpret skill scores, spatial maps, reliability diagrams, and composite plots against climatology.
* Prepare observational catalogs, spatial masks (`nc_mask`), and administrative shapefiles for new domains.
* Diagnose and resolve common configuration, path, domain, and pipeline errors.

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Why AI Benchmarking Matters | Regional agronomic frameworks and customized onset definitions. |
| The ROMP Benchmarking Package | Motivation, capabilities, workflow, and spec-driven system. |
| Spec-Driven Architecture | Using `benchmark_spec.json` and `run_momp.py` to decouple code from setup. |
| Preparing Data for New Domains | Registering observation catalogs, NetCDF spatial masks, and shapefiles. |
| Core Concepts for Benchmarking | Reforecasts, lead times, contingency counts, and climatological baselines. |
| Benchmarking Metrics | Strict separation of Deterministic (`DET`) and Probabilistic (`PROB`) tracks. |
| Models and Evaluation Setup | What is compared across short-lead (Days 1–15) and extended (Days 16–30) windows. |
| Running Multi-Country Benchmarks | CLI execution via `run_momp.py --spec` and `--mode`. |
| Deterministic Evaluation | Spatial metric maps (MAE, FAR, Miss Rate) and skill rankings. |
| Probabilistic Evaluation | Fair Brier Skill Score, RPSS, AUC, and Reliability Diagrams. |
| Composite Metric Plots | Summarising many results at once. |
| Preliminary Model Results | Reference results to replicate and investigate. |
| Best Practices and Troubleshooting | Fixing path, domain, and data errors, plus reporting checklists. |
| Exercises | Hands-on practice with spec building, map interpretation, and scoring. |
| AI Almanac Activity | Interactive multi-country exploration and feedback. |

## Why AI Benchmarking Matters

In rain-fed agricultural systems worldwide, the onset date of the rainy season dictates planting decisions, crop choice, fertilizer application, and seed investment for millions of farmers.

* **East Africa (Ethiopia / Kenya)**: Early or false onset in *Kiremt* or the *Long Rains* leads to seed wash-out or crop failure during subsequent dry spells.
* **South Asia (India)**: The arrival of the *Kharif* monsoon determines planting schedules for rice, cotton, and groundnuts across contrasting climatic zones.
* **West Africa (Sahel / Sudanian zone)**: A narrow rainy window requires precise identification of sustained rainfall before planting millet or sorghum.

Onset is **never read directly** from raw weather model outputs. To ensure fair comparison across observations, baseline models, and AI forecasts, onset is derived using an identical rule-based algorithm:

* **Initial Wet Trigger**: A candidate onset day requires an initial rainfall threshold (`wet_init` mm).
* **Wet-Spell Requirement**: Followed by `wet_spell` consecutive wet days receiving at least `wet_threshold` mm/day.
* **Dry-Spell Veto**: The candidate date is **invalidated** if a dry spell (`dry_spell` consecutive days below `dry_threshold` mm/day) occurs within a `dry_extent`-day window following the trigger.
* **Search Window**: Applied per grid cell and year across a defined seasonal window (`start_date` to `end_date`).

### Regional Onset Parameter Profiles

Because rainfall regimes differ globally, these parameters are configured per region:

| Agro-Ecological Zone | Target Season | Search Window | `wet_init` | `wet_spell` / `wet_threshold` | `dry_spell` / `dry_extent` |
| --- | --- | --- | --- | --- | --- |
| **Ethiopia Highlands** | *Kiremt* (JJAS) | Jun 1 – Aug 31 | 1.0 mm | 3 days @ 20.0 mm/day | 7 days dry within 0 days |
| **India Peninsular** | *Kharif* Monsoon | May 15 – Jul 15 | 2.5 mm | 3 days @ 10.0 mm/day | 5 days dry within 10 days |
| **West Africa (Sudanian)** | Monsoon | Apr 15 – Jun 30 | 1.0 mm | 2 days @ 10.0 mm/day | 7 days dry within 20 days |

## The ROMP Benchmarking Package

### Motivation for a Benchmarking Package

* **Demand**: Model developers and forecasters both want a reproducible, quantitative workflow for routinely evaluating model performance on onset.
* **Region- and threshold-agnostic**: The package is not tied to a single country. It can be applied to any monsoon or rainy season globally and used with any model you choose.
* **Goal functionality**: Compare models across location (grid cell, region, or country), wet-spell thresholds, dry-spell thresholds, and lead times.

### Design Capabilities

| Capability | What it means in practice |
| --- | --- |
| Metrics-based evaluation | Assumptions are objective, explicit, and controllable. |
| Multi-model reforecasts | Several models are evaluated in one run against the same reference. |
| Multiple verification sources | Observations can come from different datasets (e.g., CHIRPS). |
| Custom onset definition | Wet-spell and dry-spell rules are user parameters. |
| Region-agnostic detection | The same detection code runs for any domain and season window. |
| Lead-time evaluation | Skill is reported per verification window and per lead-time bin. |
| Configuration-driven experiments | Each run is fully described by a spec file (`benchmark_spec.json`), ensuring reproducibility. |

### ROMP Workflow

```
benchmark_spec.json  ->  load observations + model reforecasts
                     ->  apply the SAME onset definition to both
                     ->  match forecast onset to observed onset (per grid cell, per year, per init date)
                     ->  compute metrics (DET or PROB track)
                     ->  skill scores vs. climatology (or a named reference model)
                     ->  maps, tables, heatmaps, and reliability diagrams

```

## Spec-Driven Benchmarking Architecture

To eliminate the need to modify Python codebase files when changing countries or experiment parameters, `run_momp.py` utilizes a **JSON Specification System** (`--spec` or `ROMP_SPEC_PATH`).

### Writing a Country Specification File (`benchmark_spec.json`)

To run a benchmark for a new domain (e.g., Kenya or India), create a specification file:

```json
{
  "ground_truth": "CHIRPS",
  "benchmark_coverage": "Kenya",
  "forecast_window": "1-15 days",
  "shared": {
    "wet_day_threshold": 10.0,
    "minimum_wet_day_rainfall": 1.0,
    "wet_spell_length": 3,
    "dry_spell_limit": 5,
    "dry_spell_search_extension": 10,
    "area_mask_file": "data/external/kenya_seasonal_mask_0p25.nc",
    "onset_threshold_file": "default",
    "baseline_forecast": "climatology"
  },
  "models": [
    {
      "exec_key": "AIFS",
      "model_name": "AIFS",
      "eval_start": "2021-03-01",
      "eval_end": "2021-05-31",
      "baseline_start_year": 2015,
      "baseline_end_year": 2022,
      "init_days": [0, 3],
      "run_years_concurrently": true
    }
  ]
}

```

The function `apply_spec_to_cfg()` in `run_momp.py` automatically parses this JSON file and updates internal parameters:

* **Ground Truth**: Sets `cfg.obs` and updates directory paths (`obs_dir`), file patterns (`obs_file_pattern`), and variable names (`obs_var`) from `BENCHMARK_OBS_CATALOG`.
* **Coverage Domain**: Sets `cfg.region`.
* **Shared Parameters**: Sets wet/dry threshold criteria and points to local spatial masks (`nc_mask`).
* **Model Parameters**: Overrides evaluation start/end dates (`start_date`, `end_date`), baseline climate years, and initialization day intervals.

## Preparing Data for a New Country Domain

Before executing a benchmark for a new country, three data components must be registered:

### 1. Register Observational Datasets (`BENCHMARK_OBS_CATALOG`)

Observational rainfall data must be placed in your data directory (configured via `config.DATA_DIR`, defaulting to `~/data`). Register the dataset in `config.py`:

```python
BENCHMARK_OBS_CATALOG = {
    "CHIRPS": {
        "dir": "~/data/external/CHIRPS_025",
        "file_pattern": "chirps_{}.nc",
        "var": "precip",
    }
}

```

### 2. Spatial Rainfall Mask (`nc_mask`)

Rainy season onset evaluation must be restricted to grid points that actually experience a distinct rainy season.

* Provide a 2D NetCDF file (e.g., `kenya_seasonal_mask_0p25.nc`) on the target 0.25° grid.
* Cells outside the domain or without sufficient seasonal rainfall should be set to `NaN` or `0`.

### 3. Model Reforecast Directories (`BENCHMARK_MODEL_CATALOG`)

Model forecasts must be stored in standardized directories with matching spatial grids. `config.py` registers these models:

```python
BENCHMARK_MODEL_CATALOG = {
    "AIFS": {
        "model_name": "AIFS",
        "category": "Deterministic",
        "dir": "~/data/models/AIFS_reforecasts",
        "file_pattern": "aifs_{}.nc",
        "var": "tp",
    },
    "AIFS_ENS": {
        "model_name": "AIFS-ENS",
        "category": "Probabilistic",
        "dir": "~/data/models/AIFS_ENS_reforecasts",
        "file_pattern": "aifs_ens_{}.nc",
        "var": "tp",
    }
}

```

## Core Concepts for Benchmarking

These ideas underpin every table and map in the lesson. Make sure they are clear before interpreting results.

### Forecasts, Reforecasts, and Lead Time

A reforecast (hindcast) is a forecast re-run for past dates with a fixed model version. Benchmarking uses reforecasts because they provide many years of forecasts with a consistent system, which is the only way to estimate skill for a seasonal event such as onset.

An initialization date is when the forecast starts. Lead time is the number of days after initialization. Onset skill depends strongly on lead time and on how close the initialization is to the climatological onset date.

Verification is therefore organized by lead-time windows: Days 1–15 and Days 16–30 for deterministic models, and 5-day bins within those windows for probabilistic models.

### Verification Data

Observed onset is derived from a gridded rainfall product (e.g., CHIRPS). This product is the "truth" in the benchmark, but it has its own errors, so conclusions can change with the dataset.

Model and observations must be on a common grid and calendar. Very coarse grids are noisier to score but smoother to predict.

### Events, Tolerance, and Contingency Counts

Onset is a binary event per grid cell and year (did the season start within the search window?) with a timing attached (which day).

A matching tolerance decides how close in time a forecast onset must be to count as a hit. Tolerance is a scientific choice that trades strictness against usability, and it is longer at longer lead time.

Counts of hits, false alarms, misses, and correct negatives are the basis for FAR and MR. MAE is computed only where both onsets exist, so it can look good when a model rarely predicts onset. Always read it with MR.

### Ensembles and Probabilities

An ensemble is a set of forecasts (members) from slightly different initial conditions or noise. The forecast probability of onset in a bin is the fraction of members that produce onset in that bin.

With a finite number of members, probabilities are noisy, so fair scores adjust for ensemble size.

Three separate qualities of a probabilistic forecast are measured:

| Quality | Question | Metric |
| --- | --- | --- |
| Accuracy | How close are the probabilities to what happened? | BS, RPS (and skill scores) |
| Discrimination | Can the forecast separate event from non-event cases? | AUC |
| Reliability | Do stated probabilities match observed frequencies? | Reliability diagram |

A forecast can discriminate well and still be unreliable.

### Reference Forecasts and Skill

A model is only useful if it beats a reference that needs no model. The default reference is climatology, the distribution of onset dates over past years.

Skill scores are relative to that reference. They change if the reference period or dataset changes, so always document them.

### Sample Size and Uncertainty

Skill estimates are averages over grid cells, years, and initialization dates. Fewer samples give noisier scores, and rare-event bins can be extremely uncertain (see the error bars on the reliability diagram for Days 16–30).

Eight years (2015–2022) is a short record. Treat small differences between models as inconclusive unless supported by uncertainty estimates.

## Benchmarking Metrics

The pipeline enforces a strict separation between two evaluation tracks. Deterministic and probabilistic models produce fundamentally different outputs and require non-comparable metric families. A single run must be exclusively one or the other—never a mix.

> #### Track Separation
> 
> 
> Deterministic and probabilistic tracks use non-comparable metric families and must never be mixed within a single evaluation run.
> {: .warning}

### 1. Deterministic Track (Single Forecast)

| Metric | Formula Concept | Interpretation |
| --- | --- | --- |
| MAE (Mean Absolute Error) | $\Vert{}\text{forecast onset} - \text{obs onset}\Vert{}$ | Average error in days. |
| FAR (False Alarm Ratio) | $\frac{\text{false alarms}}{\text{hits} + \text{false alarms}}$ | Percentage of predicted onsets that did not occur. |
| MR (Miss Rate) | $\frac{\text{misses}}{\text{hits} + \text{misses}}$ | Percentage of actual onsets that were missed. |

**How deterministic scoring works:**

1. Onset is detected in the observations and in the forecast for each grid cell and year.
2. A forecast onset is a hit if it falls within the matching tolerance of the observed onset. Otherwise it is a false alarm (onset forecast but not observed, or too far off), and an observed onset with no matching forecast is a miss. Cells where neither has an onset are correct negatives.
3. FAR and MR come from these counts. MAE measures the timing error for the matched onsets.
4. Metrics are computed per grid cell and mapped, then optionally aggregated to regions.

> #### Metric Trade-offs
> 
> 
> Because the three metrics trade off against each other, a model that never predicts onset has zero false alarms but a 100% miss rate. A model that always predicts onset has zero misses but many false alarms. Climatology behaves like the second case (Figure 5).
> {: .tip}

### 2. Probabilistic Track (Ensemble Forecasts)

| Metric | Interpretation |
| --- | --- |
| BS (Brier Score) / BSS (Skill Score) | Mean squared error of probability forecasts. BSS represents improvement over a climatological baseline. (Lower is better for BS; Higher is better for BSS) |
| RPS (Ranked Probability Score) / RPSS | Measures the distance between the forecast and observed Cumulative Distribution Function (CDF) across ordered categories. Penalizes forecasts that are "farther" from the correct category. (Lower is better for RPS) |
| AUC (Area Under ROC Curve) | Measures discrimination ability: the probability that the model assigns a higher probability to a randomly chosen event case than to a non-event case. Range: 0 to 1. Perfect: 1.0. No skill: 0.5. (Higher is better) |
| Reliability (Calibration) | Measures the statistical consistency between forecast probabilities and observed frequencies. A perfectly reliable model predicts an event with 70% probability exactly 70% of the time it occurs. Typically visualized via a Reliability Diagram. |

**How probabilistic scoring works:**

1. The onset probability for a lead-time bin (5-day bins in this lesson) is the fraction of ensemble members that produce an onset in that bin.
2. Fair versions of BS and RPS correct for the finite number of members, so ensembles of different sizes can be compared.
3. BS/BSS score a yes/no event, RPS/RPSS score the whole distribution of onset dates, AUC scores discrimination, and reliability scores calibration. Each answers a different question, so read them together.

### 3. Skill Score Definition

Raw metrics alone do not establish whether an AI model beats a naive baseline. Therefore, both tracks report a final Skill Score ($SS$) relative to a reference model (climatology by default, or a named model):

$$SS = 1 - \frac{\text{Metric}_{\text{model}}}{\text{Metric}_{\text{reference}}}$$

**Interpretation:**

* $SS = 1$: Perfect forecast.
* $SS > 0$: The model outperforms the reference (positive skill).
* $SS = 0$: The model performs identically to the reference.
* $SS < 0$: The model performs worse than the reference (negative skill).

## Models and Evaluation Setup

### Models in the Benchmark

Model assignments to specific tracks are defined centrally in the `BENCHMARK_MODEL_CATALOG` within `config.py` to prevent duplication or inconsistency.

| Model | Type | Origin | Resolution |
| --- | --- | --- | --- |
| AIFS | Deterministic | ECMWF | ~25 km (0.25°) |
| FuXi | Deterministic | Fudan University | ~25 km |
| GraphCast | Deterministic | Google DeepMind | ~25 km |
| AIFS-ENS | Probabilistic (50 members; 25 used in evaluation) | ECMWF | ~25 km |
| GenCast | Probabilistic (Diffusion, 52 members) | Google DeepMind | ~25 km |

### Evaluation Windows

Two verification windows are evaluated for each model to assess short-lead versus extended-lead performance:

| Verification Window | Window Length | Matching Tolerance |
| --- | --- | --- |
| Days 1–15 after initialization | 15 days | 3 days |
| Days 16–30 after initialization | 15 days | 5 days |

## Running Multi-Country Benchmarks

Execute benchmarks using `run_momp.py` directly from the command line.

### Command-Line Execution

```bash
# 1. Deterministic benchmark using a country specification JSON file
python run_momp.py --mode det --spec config/benchmark_spec_kenya.json --model AIFS

# 2. Probabilistic benchmark for a specific ensemble model
python run_momp.py --mode prob --spec config/benchmark_spec_india.json --model AIFS_ENS

# 3. In-process multi-model run
python run_momp.py --mode det --spec config/benchmark_spec_ethiopia.json --in-process

```

> #### Track Separation
> 
> 
> Deterministic and probabilistic tracks use non-comparable metric families and must never be mixed within a single evaluation run. Always specify the mode explicitly with `--mode det` or `--mode prob`.
> {: .warning}

### Output Location

Results are written to `ROMP_OUT_DIR` (configured in `config.py`, defaulting to `romp_output/<model_name>/`):

* Spatial metric maps (PNG)
* Skill score tables (CSV)
* Reliability diagrams (PNG)
* NetCDF files with gridded metrics

## Deterministic Evaluation

### ROMP Run Summary

* **Package**: Rainy Season Onset Metrics Package (ROMP), v0.0.1
* **Run Mode**: Deterministic (`DET`)
* **Start Time**: 2026-09-28 13:42:33
* **Models Evaluated**: AIFS, FuXi, GraphCast
* **Reference Dataset**: CHIRPS
* **Evaluation Years**: 2015–2022
* **Spatial Grid**: 49 latitudes × 61 longitudes at 0.2° resolution
* **Spatial Products Generated**: FAR, Miss Rate, yearly MAE, and mean MAE maps.

### How to Read the Spatial Metric Maps

For each model and verification window the pipeline writes a three-panel map (`spatial_metrics_<model>_<window>.png`):

| Panel | Colour scale | What "good" looks like |
| --- | --- | --- |
| MAE (in days) | White/cream → dark red (0–14+ days) | Light colours |
| False Alarm Rate (%) | White → dark red (0–100 %) | Light colours |
| Miss Rate (%) | White → dark blue (0–100 %) | Light colours |

> #### Reading Blank Cells
> 
> 
> Blank (white) grid cells inside the country outline mean the metric is undefined there (for example, no matched onset events, so MAE cannot be computed). Blank does not mean "perfect". Compare each model against the climatology reference maps (Figure 5) before drawing conclusions.
> {: .warning}

### AIFS Results

AIFS is the strongest deterministic model at short lead. Its errors grow quickly in Days 16–30.

*Figure 2a. AIFS, Days 1–15. MAE is low (light) across most of the western and central highlands. Errors and misses concentrate in the east and northeast, where onset is late and the season is short. False alarms are highest in the far southwest, where the rainy season starts earliest.*

*Figure 2b. AIFS, Days 16–30. MAE rises sharply almost everywhere, and the southwest false-alarm area becomes saturated (close to 100 %). Miss rates in the east are similar to Days 1–15.*

### FuXi Results

FuXi rarely issues an onset, so it has few false alarms but a very high miss rate, especially in Days 16–30.

*Figure 3a. FuXi, Days 1–15. Many grid cells are blank because FuXi produced no matched onset there. Where MAE is defined it is low in the west, but the miss-rate panel is dominated by dark blue across the north and east.*

*Figure 3b. FuXi, Days 16–30. Almost every cell is either blank or dark blue in the miss-rate panel. The few cells with an MAE value are mostly dark red. This illustrates why a low false-alarm rate can simply reflect a model that rarely predicts onset at all.*

### GraphCast Results

GraphCast detects onset well in Days 1–15 but pays for it with frequent false alarms.

*Figure 4a. GraphCast, Days 1–15. Miss rates are low over most of the country (light blue) apart from the east. The far-southwest false-alarm area is close to 100 %, reflecting GraphCast's tendency to over-predict onset.*

*Figure 4b. GraphCast, Days 16–30. MAE is high across the north, where it is defined, and many cells are blank. False alarms are large in the northwest and southwest.*

### Climatology Reference Maps

Every skill score is measured against climatology, so inspect the reference maps too. A forecast model must beat these to add value.

*Figure 5. Climatology reference baseline.*

### Deterministic Main Findings

* Short-lead forecasts (Days 1–15) generally outperformed long-lead forecasts (Days 16–30) for every model.
* Extending the lead time increased MAE and misses, and the MAE advantage over climatology largely disappeared.
* GraphCast had the strongest short-lead detection but frequent false alarms. FuXi showed the opposite pattern, with few false alarms and a very high miss rate.
* No single metric is enough. FAR, MR, and MAE must be read together, and always against climatology.

### Model Skill Rankings from the Original Test Run (Reference: Climatology = 0.0)

| Model | Mean MAE Skill | False Alarm Rate Skill | Miss Rate Skill | Overall Skill Score |
| --- | --- | --- | --- | --- |
| Climatology | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| AIFS_ENS | 0.122734 | 0.473197 | -0.308828 | 0.095701 |
| GenCast | -0.043218 | -0.389341 | 0.364237 | -0.022774 |
| AIFS | -0.201814 | -0.275234 | 0.323369 | -0.051226 |
| FuXi | -0.271470 | 0.203178 | -0.262507 | -0.110266 |
| GraphCast | -0.290775 | -0.541115 | 0.329475 | -0.167472 |

## Probabilistic Evaluation (AIFS-ENS)

### Run Configuration

* **Command Executed**: `python run_momp.py --mode prob --model AIFS_ENS`
* **Model Evaluated**: AIFS-ENS (25 ensemble members)
* **Evaluation Mode**: Probabilistic (`PROB`)
* **Observational Reference**: CHIRPS rainfall data
* **Verification Windows**: Days 1–15 and Days 16–30

### Skill Scores: Verification Window (Days 1–15)

*(Lower is better for Brier Score and RPS; higher is better for AUC.)*

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
| --- | --- | --- | --- |
| Fair Brier Score | 0.1107 | 0.0953 | BSS = -0.162 |
| Fair RPS | 0.5439 | 0.4326 | RPSS = -0.257 |
| AUC | 0.692 | 0.819 | – |

**Bin-wise Fair Brier Skill Score (BSS):**

| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
| --- | --- | --- | --- | --- | --- |
| Days 1-5 | 0.0849 | 0.0907 | 0.064 | 0.845 | 0.870 |
| Days 6-10 | 0.1165 | 0.0961 | -0.213 | 0.693 | 0.813 |
| Days 11-15 | 0.1307 | 0.0990 | -0.320 | 0.533 | 0.768 |

**Overall Fair BSS**: `-0.162` | **Overall Fair RPSS**: `-0.257`

The only positive skill is in the first five days (BSS = +6.4 %). Skill then falls below climatology and AUC drops from 0.85 to 0.53 by Days 11–15.

*Figure 7a. AIFS-ENS skill by 5-day bin, Days 1–15. Top row: BSS (%). Bottom row: AUC (climatology reference in brackets). Skill decays quickly with lead time.*

### Skill Scores: Verification Window (Days 16–30)

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
| --- | --- | --- | --- |
| Fair Brier Score | 0.0828 | 0.0659 | BSS = -0.258 |
| Fair RPS | 0.5413 | 0.3643 | RPSS = -0.486 |
| AUC | 0.501 | 0.805 | – |

**Bin-wise Fair Brier Skill Score (BSS):**

| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
| --- | --- | --- | --- | --- | --- |
| Days 16-20 | 0.1061 | 0.0826 | -0.285 | 0.502 | 0.778 |
| Days 21-25 | 0.0807 | 0.0644 | -0.254 | 0.500 | 0.808 |
| Days 26-30 | 0.0617 | 0.0507 | -0.218 | 0.500 | 0.821 |

**Overall Fair BSS**: `-0.258` | **Overall Fair RPSS**: `-0.486`

*Figure 7b. AIFS-ENS skill by 5-day bin, Days 16–30. BSS stays negative in every bin, and AUC is 0.5 throughout, meaning the ensemble cannot discriminate onset from non-onset.*

> #### Watch Out: Base-Rate Effect
> 
> 
> The BSS gets less negative from Days 16–20 to 26–30 (−28 % → −22 %) even though AUC is flat at 0.5. This is a base-rate effect, not real skill. Brier scores are lower later in the window simply because the event is rarer. Always read BSS together with AUC.
> {: .warning}

### Reliability Analysis

A reliability diagram plots the forecast probability (x) against how often the event actually occurred (y). Points on the dashed 1:1 line are perfectly reliable. Grey bars (log scale, right axis) show how many forecasts fall in each probability bin.

*Figure 8a. Reliability, Days 1–15. The curve is flatter than the diagonal. Near-zero forecasts verify about 10 % of the time (under-forecast), and forecasts near 1.0 verify only about 80 % of the time (over-forecast). The ensemble is overconfident. Most forecasts sit in the lowest probability bin.*

*Figure 8b. Reliability, Days 16–30. Nearly all forecasts are below 0.3, the curve has no upward trend, and the last points have huge error bars from small samples. Probabilities carry almost no information, matching the AUC ≈ 0.5.*

## Composite Metric Plots

Composite plots condense many runs into one figure so models, windows, and metrics can be compared at a glance:

* **Portrait panel**: Δ MAE, Δ FAR, and Δ MR for each deterministic model and window, relative to climatology.
* **Skill heatmaps (Figure 7)**: BSS and AUC per 5-day lead bin for probabilistic models.
* **Extension**: The same layout can be repeated across wet-spell thresholds, dry-spell thresholds, or regions, which is how the package supports the goal of comparing models across location, thresholds, and lead time.

> #### Reading Composite Plots
> 
> 
> Read composite plots with care. They hide the spatial detail in the maps and the sample size behind each cell.
> {: .tip}

## Preliminary Model Results

These results compare climatology, AIFS, and GenCast at two resolutions across 2019–2024.

### 0.25° Resolution

GenCast probabilistic skill in 5-day bins: Days 1–15 and Days 16–30.

| Window | AUC | BSS | RPSS |
| --- | --- | --- | --- |
| 1–15 day | 0.85 | 7.4 % | 29.4 % |
| 1–30 day | 0.77 | 0.2 % | 6.8 % |

Probabilistic scores improve over climatology, especially for Days 1–15.

### 1° Resolution

GenCast probabilistic skill in 5-day bins: Days 1–15 and Days 16–30.

| Window | AUC | BSS | RPSS |
| --- | --- | --- | --- |
| 1–15 day | 0.88 | 17 % | 30 % |
| 1–30 day | 0.80 | 8.2 % | 8.2 % |

Improvements over climatology are higher at coarser resolution, but these estimates are noisier because the sample size is smaller.

## Best Practices and Troubleshooting

### Best Practices for Defensible Benchmarks

| Concept | Why it matters |
| --- | --- |
| Same onset definition everywhere | Different rules for observations and forecasts make comparisons meaningless. |
| Threshold sensitivity | Changing wet/dry thresholds can change ranking. Test several. |
| Spec-driven configuration | Decouple setup from code using `benchmark_spec.json` for reproducibility. |
| Matching tolerance | A larger tolerance raises hits and hides timing error. Report it with every result. |
| Reference choice | Climatology, persistence, or another model give different skill scores. |
| Base rate | BSS can move only because the event becomes rarer. |
| Resolution and sample size | Coarse grids give higher skill but noisier estimates. |
| Separate DET and PROB tracks | Metric families are not comparable. Never mix them in one run. |
| Reproducibility | Keep the spec JSON file, package version, and data versions with every result. |

### Reporting Checklist

Every benchmark result should state:

* [ ] Onset definition parameters (`wet_init`, `wet_spell`, `wet_threshold`, `dry_spell`, `dry_threshold`, `dry_extent`, search window)
* [ ] Verification dataset and its spatial resolution
* [ ] Model, reforecast period, initialization dates, and number of members
* [ ] Lead-time windows and matching tolerance
* [ ] Reference used for skill (and its baseline period)
* [ ] Track (`DET` or `PROB`)
* [ ] Package version and the JSON spec file used

### Troubleshooting Common Failure Modes

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `FileNotFoundError: Model data directory not found` | Catalog folder path in `config.py` or spec JSON does not exist. | Update `dir` in `BENCHMARK_MODEL_CATALOG` or verify path in `benchmark_spec.json`. |
| `FileNotFoundError: obs: no file in obs_dir for years...` | Filename pattern or year range missing in observation folder. | Check `file_pattern` (e.g., `chirps_{}.nc`) and verify NetCDF files exist for evaluation years. |
| Run stops or results look meaningless after switching tracks | DET and PROB mixed in one run. | Set mode with `--mode det` or `--mode prob` explicitly. |
| Output maps are completely blank / `NaN` | Spatial mask (`nc_mask`) domain does not overlap model coordinates. | Ensure `nc_mask` NetCDF file matches the 0.25° model grid coordinates. |
| `ModuleNotFoundError: No module named 'config'` | `PYTHONPATH` missing project root when executing from subdirectories. | Ensure script directory is in `sys.path` or set `export PYTHONPATH=.`. |
| Extremely high miss rate for one model | Model rarely reaches the wet-spell threshold (e.g., drizzle bias). | Rainfall distribution vs. observations, `wet_threshold`. |

## Exercises

> #### Exercise 1: Construct a Country Specification File (15 min)
> 
> 
> 1. Create a file named `benchmark_spec_india.json`.
> 2. Configure the spec to evaluate the `AIFS` model against `CHIRPS` observations for India.
> 3. Set the initial rainfall trigger (`minimum_wet_day_rainfall`) to `2.5` mm and `wet_day_threshold` to `10.0` mm.
> 4. Test your spec file by running:
> ```bash
> python run_momp.py --mode det --spec benchmark_spec_india.json --model AIFS
> 
> ```
> 
> 
> 
> 

{: .exercise}

> #### Exercise 2: Read the Configuration (10 min)
> 
> 
> 1. Open `benchmark_spec.json` or inspect `apply_spec_to_cfg` in `run_momp.py`.
> 2. Identify the onset parameters, the verification windows, the matching tolerance, and the run mode.
> 3. Predict what happens to FAR and MR if `wet_threshold` is increased. Explain your reasoning.
> {: .exercise}
> 
> 

> #### Exercise 3: Interpret Deterministic Maps (15 min)
> 
> 
> Use Figures 2–5.
> 1. Where is the miss rate highest for AIFS in Days 1–15, and why might onset be harder to forecast there?
> 2. Compare FuXi and GraphCast. Which one over-predicts onset and which one under-predicts it?
> 3. Why can a model with a lower FAR than climatology still be a poor forecast?
> {: .exercise}
> 
> 

> #### Exercise 4: Compute a Skill Score by Hand (10 min)
> 
> 
> Given an ensemble forecast with a Fair Brier Score of `0.085` and a Climatology Reference Brier Score of `0.100`:
> 1. Calculate the Brier Skill Score ($BSS = 1 - \frac{BS_{\text{model}}}{BS_{\text{ref}}}$).
> 2. Compare with the Days 1–5 bin in the AIFS-ENS probabilistic table.
> {: .exercise}
> 
> 

> #### Solution
> 
> 
> $$BSS = 1 - \frac{0.085}{0.100} = 1 - 0.85 = +0.15 \text{ (or +15\% skill improvement)}$$
> 
> 
> 
> The model shows positive skill over climatology.
> {: .solution}

> #### Exercise 5: Diagnose Reliability (10 min)
> 
> 
> Use Figures 8a and 8b.
> 1. Is the Days 1–15 ensemble over- or under-confident? Give the evidence from the curve.
> 2. Why are the points in Figure 8b so uncertain?
> {: .exercise}
> 
> 

## AI Almanac Exploration and Feedback

### Objective

Explore the AI Almanac interface and gather feedback on the necessary components required to assess metrics and use cases across different countries in an intuitive, interactive manner.

### Activity

* Guide participants to explore the Streamlit interface (`app.py`).
* Connect regional preset selections in the UI with `benchmark_spec.json` files.
* Have paired country groups share their ideas and feedback across both deterministic and probabilistic evaluation tracks.

## Summary

> #### Key Points
> 
> 
> * Onset is a derived quantity. It must be detected with the same wet-spell and dry-spell rules in observations and forecasts.
> * ROMP is specification-driven (`benchmark_spec.json`), so results can be reproduced and compared across location, thresholds, lead time, and models.
> * Deterministic (FAR, MR, MAE) and probabilistic (BS, RPS, AUC, reliability) tracks are separate and their metrics cannot be compared directly.
> * Skill exists only relative to a reference. Models beat climatology at short lead and lose that advantage by Days 16–30.
> * Read metrics together: FAR with MR and MAE, and BSS with AUC and reliability.
> * Resolution, period, tolerance, and sample size all affect scores. Document them with every result.
> {: .keypoints}
> 
>
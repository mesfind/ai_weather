---
title: "Demo 4: The AI Weather Model Scorecard"
teaching: 30
exercises: 60
questions:
  - "How do we systematically benchmark AI weather models against local observations for rainy season onset?"
  - "How do deterministic and probabilistic evaluation tracks differ, and why must they remain strictly separated?"
  - "How do we configure, run, and troubleshoot the ROMP/MOMP pipeline reliably from the command line and within notebooks?"
  - "How does Isotonic Distributional Regression (IDR) calibration improve (or fail to improve) probabilistic onset forecasts, and how do we diagnose this?"
objectives:
  - "Understand the onset detection algorithm and why it must be applied identically to observations and forecasts."
  - "Configure and run deterministic benchmarks (MAE, FAR, Miss Rate) and probabilistic benchmarks (BS, RPS, AUC, Reliability)."
  - "Utilize the swappable, per-run Python configuration system correctly, including CLI mode selection."
  - "Apply and diagnose Isotonic Distributional Regression (IDR) calibration on probabilistic onset forecasts."
  - "Recognize and resolve the most common configuration, data, and pipeline failure modes."
keypoints:
  - "Onset is derived identically from observations and forecasts using a wet-spell/dry-spell veto rule, not read directly from raw model output."
  - "Deterministic and probabilistic tracks use non-comparable metric families (e.g., FAR/MAE/MR vs. BS/RPS/AUC) and must never be mixed within a single evaluation run."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# The AI Weather Model Scorecard

This lesson documents the complete **ROMP** (Rainy season Onset Metrics Package) / **MOMP** benchmarking workflow. It is used to evaluate AI weather forecast models (e.g., AIFS, FuXi, GraphCast, GenCast, AIFS-ENS) against observational rainfall data (e.g., CHIRPS) for rainy season onset prediction. The workflow covers both deterministic and probabilistic evaluation tracks, as well as the configuration system that drives them.

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
1. Understand the theoretical foundations of rainy season onset detection.
2. Configure and execute deterministic benchmarks (MAE, FAR, Miss Rate).
3. Configure and execute probabilistic benchmarks (Brier Score, Ranked Probability Score, AUC, Reliability).
4. Correctly utilize the swappable, per-run Python configuration system and CLI mode selection.
5. Diagnose and resolve common configuration, data, and pipeline errors.
6. Explain why the package exists, what it can compare (location, thresholds, lead time, model) and how the ROMP workflow is organised.
7. Interpret composite metric plots and compare preliminary results across resolutions (0.25° vs 1°).

---

## ROMP Overview

### Motivation for a Benchmarking Package

- **Demand:** model developers and forecasters both want a reproducible, quantitative workflow for routinely evaluating model performance on onset.
- **Region- and threshold-agnostic:** the package is not tied to Ethiopia. It can be applied to Kiremt rains and to other regions, and used with any model you choose.
- **Goal functionality:** compare models across
  1. **location** (grid cell, region or country),
  2. **wet-spell thresholds**,
  3. **dry-spell thresholds**, and
  4. **lead times**.

### Design Capabilities

We want your feedback, in person and online, to improve this package.

| Capability | What it means in practice |
|------------|---------------------------|
| Metrics-based evaluation | Assumptions are objective, explicit and controllable. |
| Multi-model reforecasts | Several models are evaluated in one run against the same reference. |
| Multiple verification sources | Observations can come from different datasets (e.g. CHIRPS, ENACTS). |
| Custom onset definition | Wet-spell and dry-spell rules are user parameters. |
| Region-agnostic detection | The same detection code runs for any domain and season window. |
| Lead-time evaluation | Skill is reported per verification window and per lead-time bin. |
| Configuration-driven experiments | Each run is fully described by its config file, so results can be reproduced. |

### ROMP Workflow

```text
config file  ->  load observations + model reforecasts
             ->  apply the SAME onset definition to both
             ->  match forecast onset to observed onset (per grid cell, per year, per init date)
             ->  compute metrics (DET or PROB track)
             ->  skill scores vs. climatology (or a named reference model)
             ->  maps, tables, heatmaps and reliability diagrams
```

### ROMP Specifications

**1. Onset definition (set in the config)**

| Parameter | Meaning |
|-----------|---------|
| `wet_init` | Minimum initial rainfall (mm) that starts a candidate onset |
| `wet_spell`, `wet_threshold` | Number of consecutive wet days, and the mm/day that counts as wet |
| `dry_spell`, `dry_threshold` | Length of a dry spell, and the mm/day below which a day is dry |
| `dry_extent` | Window (days) after the candidate in which a dry spell vetoes the onset |
| `start_date`, `end_date` | Search window for the season (e.g. Kiremt, June–September) |

**2. Data and domain**

- Observations or reference rainfall: CHIRPS, ENACTS or another gridded product.
- Forecasts: daily rainfall from each model's reforecasts, on a common grid (e.g. 0.25°).
- A seasonal mask (e.g. `jjas_seasonal_mask_0p25.nc`) restricts evaluation to the grid points where a rainy season occurs.
- Models are registered centrally in `BENCHMARK_MODEL_CATALOG`.

**3. Verification set-up**

- Verification windows: Days 1–15 and Days 16–30 after initialization.
- Matching tolerance: 3 days (Days 1–15) and 5 days (Days 16–30).
- Reference for skill scores: climatology by default, or a named model.
- Run mode: `DET` or `PROB`, chosen in the config or overridden on the command line (`--mode prob`). Never both in the same run.

---

##  Why Onset Matters

In Ethiopian agriculture, the **onset of the rainy season** (Kiremt: June–September) dictates planting dates for millions of smallholder farmers. A late or false onset signal can lead to crop failure from planting too early, lost growing days from planting too late, and regional food insecurity. 

![Mean rainy season onset date (day of year) across Ethiopia, 2003–2024. Onset is earliest (mid-to-late May) in the southwestern highlands and progressively later toward the north and east, reaching late July–August in the northeast lowlands.](figures/climatology_onset_2003-2024.png){alt="Map of mean onset date across Ethiopia"}

*Figure 1. Mean rainy season onset date (day of year), 2003–2024. Onset arrives first in the southwest (mid-to-late May) and progressively later toward the north and east (July–August). This strong spatial gradient is why skill must be examined per grid cell and not only as a national average.*

**Onset is not read directly from raw model output.** To ensure genuine comparability, it is derived identically from daily rainfall data for observations, the reference model, and every forecast model using the following rules:

- **Wet-spell trigger**: A candidate onset day requires at least `wet_init` mm of initial rainfall, followed by `wet_spell` consecutive days with ≥ `wet_threshold` mm/day.
- **Dry-spell veto**: A candidate onset is invalidated if a dry spell (`dry_spell` consecutive days below `dry_threshold` mm/day) occurs within a `dry_extent`-day window afterward. This rejects false starts.
- **Application**: These rules are applied per grid cell, per year, within a defined search window (`start_date` to `end_date`).

---

## Benchmarking Metrics

The pipeline enforces a strict separation between two evaluation tracks. Deterministic and probabilistic models produce fundamentally different outputs and require non-comparable metric families. **A single run must be exclusively one or the other—never a mix.**

### 1. Deterministic Track (Single Forecast)
| Metric | Formula Concept | Interpretation |
|--------|----------------|----------------|
| **MAE** (Mean Absolute Error) | $\|\text{forecast onset} - \text{obs onset} \|$ | Average error in days. |
| **FAR** (False Alarm Ratio) | $\frac{\text{false alarms}}{\text{hits} + \text{false alarms}}$ | Percentage of predicted onsets that did not occur. |
=======
| **MAE** (Mean Absolute Error) | $\| \text{forecast onset} - \text{obs\_onset} \|$ | Average error in days. |
| **FAR** (False Alarm Ratio) | $\frac{\text{false alarms}}{\text{hits} + \text{false\_alarms}}$ | Percentage of predicted onsets that did not occur. |
| **MR** (Miss Rate) | $\frac{\text{misses}}{\text{hits} + \text{misses}}$ | Percentage of actual onsets that were missed. |

**How deterministic scoring works**

1. Onset is detected in the observations and in the forecast for each grid cell and year.
2. A forecast onset is a **hit** if it falls within the matching tolerance of the observed onset. Otherwise it is a **false alarm** (onset forecast but not observed, or too far off), and an observed onset with no matching forecast is a **miss**. Cells where neither has an onset are **correct negatives**.
3. FAR and MR come from these counts. MAE measures the timing error for the matched onsets.
4. Metrics are computed per grid cell and mapped, then optionally aggregated to regions.

Because the three metrics trade off against each other, a model that never predicts onset has zero false alarms but a 100 % miss rate. A model that always predicts onset has zero misses but many false alarms. Climatology behaves like the second case (Figure 5).

### 2. Probabilistic Track (Ensemble Forecasts)
| Metric | Interpretation |
|--------|----------------|
| **BS** (Brier Score) / **BSS** (Skill Score) | Mean squared error of probability forecasts. BSS represents improvement over a climatological baseline. *(Lower is better for BS; Higher is better for BSS)* |
| **RPS** (Ranked Probability Score) / **RPSS** | Measures the distance between the forecast and observed Cumulative Distribution Function (CDF) across ordered categories. Penalizes forecasts that are "farther" from the correct category. *(Lower is better for RPS)* |
| **AUC** (Area Under ROC Curve) | Measures discrimination ability: the probability that the model assigns a higher probability to a randomly chosen event case than to a non-event case. Range: 0 to 1. Perfect: 1.0. No skill: 0.5. *(Higher is better)* |
| **Reliability** (Calibration) | Measures the statistical consistency between forecast probabilities and observed frequencies. A perfectly reliable model predicts an event with 70% probability exactly 70% of the time it occurs. Typically visualized via a Reliability Diagram. |

**How probabilistic scoring works**

1. The onset probability for a lead-time bin (5-day bins in this lesson) is the fraction of ensemble members that produce an onset in that bin.
2. **Fair** versions of BS and RPS correct for the finite number of members, so ensembles of different sizes can be compared.
3. **BS/BSS** score a yes/no event, **RPS/RPSS** score the whole distribution of onset dates, **AUC** scores discrimination and **reliability** scores calibration. Each answers a different question, so read them together.
4. A model can have good AUC but poor reliability. Calibration methods such as Isotonic Distributional Regression (IDR) can fix reliability but cannot create discrimination that is not there.

### 3. Skill Score Definition
Raw metrics alone do not establish whether an AI model beats a naive baseline. Therefore, both tracks report a final **Skill Score (SS)** relative to a reference model (climatology by default, or a named model):

$$
SS = 1 - \frac{\text{Metric}_{\text{model}}}{\text{Metric}_{\text{reference}}}
$$

**Interpretation:**
- $SS = 1$: Perfect forecast.
- $SS > 0$: The model outperforms the reference (positive skill).
- $SS = 0$: The model performs identically to the reference.
- $SS < 0$: The model performs worse than the reference (negative skill).

---

## Models in the Benchmark

Model assignments to specific tracks are defined centrally in the `BENCHMARK_MODEL_CATALOG` within the infrastructure `config.py` to prevent duplication or inconsistency.

| Model | Type | Origin | Resolution |
|-------|------|--------|------------|
| **AIFS** | Deterministic | ECMWF | ~25 km (0.25°) |
| **FuXi** | Deterministic | Fudan University | ~25 km |
| **GraphCast** | Deterministic | Google DeepMind | ~25 km |
| **AIFS-ENS** | Probabilistic (50 members) | ECMWF | ~25 km |
| **GenCast** | Probabilistic (Diffusion, 52 members) | Google DeepMind | ~25 km |

---

## Evaluation Setup

Two verification windows are evaluated for each model to assess short-lead versus extended-lead performance:

| Verification Window | Window Length | Matching Tolerance |
|---------------------|---------------|--------------------|
| Days 1–15 after initialization | 15 days | 3 days |
| Days 16–30 after initialization | 30 days | 5 days |

---

# Part 1: Deterministic Evaluation

### ROMP Run Summary
- **Package:** Rainy Season Onset Metrics Package (ROMP), v0.0.1
- **Run Mode:** Deterministic (`DET`)
- **Project:** Test ROMP run with sample data
- **Start Time:** 2026-09-28 13:42:33
- **Models Evaluated:** AIFS, FuXi, GraphCast
- **Reference Dataset:** ENACTS
- **Evaluation Years:** 2015–2022
- **Computational Resources:** 6 cores (of 10 available CPUs)
- **Spatial Grid:** 49 latitudes × 61 longitudes at 0.2° resolution
- **Spatial Products Generated:** FAR, Miss Rate, yearly MAE, and mean MAE maps *(Note: CMZ averages were not calculated as 0.2° resolution is unsupported for this specific aggregation).*

### How to Read the Spatial Metric Maps

For each model and verification window the pipeline writes a three-panel map (`spatial_metrics_<model>_<window>.png`):

| Panel | Colour scale | What "good" looks like |
|-------|--------------|------------------------|
| **MAE (in days)** | White/cream → dark red (0–14+ days) | Light colours |
| **False Alarm Rate (%)** | White → dark red (0–100 %) | Light colours |
| **Miss Rate (%)** | White → dark blue (0–100 %) | Light colours |

Blank (white) grid cells inside the country outline mean the metric is undefined there (for example, no matched onset events, so MAE cannot be computed). **Blank does not mean "perfect".** Compare each model against the climatology reference maps (Figures 5a–5b) before drawing conclusions.

### AIFS Results

AIFS is the strongest deterministic model at short lead. Its errors grow quickly in Days 16–30.

![AIFS spatial MAE, false alarm rate and miss rate, days 1-15](figures/spatial_metrics_AIFS_1-15.png){alt="AIFS days 1-15 spatial metrics"}

*Figure 2a. AIFS, Days 1–15. MAE is low (light) across most of the western and central highlands. Errors and misses concentrate in the east and northeast, where onset is late and the season is short. False alarms are highest in the far southwest, where the rainy season starts earliest.*

![AIFS spatial MAE, false alarm rate and miss rate, days 16-30](figures/spatial_metrics_AIFS_16-30.png){alt="AIFS days 16-30 spatial metrics"}

*Figure 2b. AIFS, Days 16–30. MAE rises sharply almost everywhere, and the southwest false-alarm area becomes saturated (close to 100 %). Miss rates in the east are similar to Days 1–15.*

### FuXi Results

FuXi rarely issues an onset, so it has few false alarms but a very high miss rate, especially in Days 16–30.

![FuXi spatial metrics, days 1-15](figures/spatial_metrics_FuXi_1-15.png){alt="FuXi days 1-15 spatial metrics"}

*Figure 3a. FuXi, Days 1–15. Many grid cells are blank because FuXi produced no matched onset there. Where MAE is defined it is low in the west, but the miss-rate panel is dominated by dark blue across the north and east.*

![FuXi spatial metrics, days 16-30](figures/spatial_metrics_FuXi_16-30.png){alt="FuXi days 16-30 spatial metrics"}

*Figure 3b. FuXi, Days 16–30. Almost every cell is either blank or dark blue in the miss-rate panel. The few cells with an MAE value are mostly dark red. This illustrates why a low false-alarm rate can simply reflect a model that rarely predicts onset at all.*

### GraphCast Results

GraphCast detects onset well in Days 1–15 but pays for it with frequent false alarms.

![GraphCast spatial metrics, days 1-15](figures/spatial_metrics_GraphCast_1-15.png){alt="GraphCast days 1-15 spatial metrics"}

*Figure 4a. GraphCast, Days 1–15. Miss rates are low over most of the country (light blue) apart from the east. The far-southwest false-alarm area is close to 100 %, reflecting GraphCast's tendency to over-predict onset.*

![GraphCast spatial metrics, days 16-30](figures/spatial_metrics_GraphCast_16-30.png){alt="GraphCast days 16-30 spatial metrics"}

*Figure 4b. GraphCast, Days 16–30. Maps are now available for this window. MAE is high across the north, where it is defined, and many cells are blank. False alarms are large in the northwest and southwest.*

### Climatology Reference Maps

Every skill score is measured against climatology, so inspect the reference maps too. A forecast model must beat these to add value.

![Climatology reference spatial metrics, days 1-15](figures/spatial_metrics_climatology_1-15.png){alt="Climatology days 1-15 spatial metrics"}

*Figure 5a. Climatology reference, Days 1–15.*

![Climatology reference spatial metrics, days 16-30](figures/spatial_metrics_climatology_16-30.png){alt="Climatology days 16-30 spatial metrics"}

*Figure 5b. Climatology reference, Days 16–30. Climatology has low miss rates because it always predicts an onset, but it pays for this with false alarms of nearly 100 % across the west and dark-red MAE in the east. This trade-off is why FAR, MR and MAE must be read together.*

### Deterministic Skill Relative to Climatology

![Change in MAE, FAR and MR relative to climatology for AIFS, FuXi and GraphCast in the 1-15 and 16-30 day windows](figures/panel_portrait_mae_far_mr_AIFS_30day.png){alt="Portrait panel of delta MAE, FAR and MR"}

*Figure 6. Change (Δ) in MAE (days), FAR (%) and MR (%) for each deterministic model in each window. Blue cells are reductions and red cells are increases relative to the reference. Days 1–15: all three models reduce MAE by roughly 3.6–4.9 days (FuXi −4.87, AIFS −4.05, GraphCast −3.58). Days 16–30: the MAE advantage largely vanishes (AIFS −0.70, GraphCast +0.33) or reverses (FuXi +2.11). The FAR and MR changes are small in percentage points (all within ±0.6).*

> **Instructor note:** Figure 6 shows MAE improving over the reference in Days 1–15 for all three models. The *Model Skill Rankings* table below reports negative mean-MAE skill for AIFS, FuXi and GraphCast. These come from different summaries (per-window Δ vs. a single pooled score) and possibly different runs or references. Ask participants to find what could explain the difference (reference dataset, window, aggregation) before trusting either number.

### Deterministic Main Findings
- Short-lead forecasts (Days 1–15) generally outperformed long-lead forecasts (Days 16–30) for every model.
- Extending the lead time increased MAE and misses, and the MAE advantage over climatology largely disappeared (Figure 6).
- GraphCast had the strongest short-lead detection but frequent false alarms. FuXi showed the opposite pattern, with few false alarms and a very high miss rate.
- No single metric is enough. FAR, MR and MAE must be read together, and always against climatology.
- All spatial metric files (NetCDF and PNG) were saved successfully for completed model/window combinations, with no fatal errors or tracebacks in the output.

### Model Skill Rankings (Reference: Climatology = 0.0)
| Model | Mean MAE Skill | False Alarm Rate Skill | Miss Rate Skill | Overall Skill Score |
|-------|----------------|------------------------|-----------------|---------------------|
| **Climatology** | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **AIFS_ENS** | 0.122734 | 0.473197 | -0.308828 | 0.095701 |
| **GenCast** | -0.043218 | -0.389341 | 0.364237 | -0.022774 |
| **AIFS** | -0.201814 | -0.275234 | 0.323369 | -0.051226 |
| **FuXi** | -0.271470 | 0.203178 | -0.262507 | -0.110266 |
| **GraphCast** | -0.290775 | -0.541115 | 0.329475 | -0.167472 |

# Part 2: Probabilistic Evaluation (AIFS-ENS)

### Run Configuration
- **Command Executed:** `momp-run -p notebooks/config_et.in --mode prob`
- **Model Evaluated:** AIFS-ENS (25 ensemble members)
- **Evaluation Mode:** Probabilistic (CLI override)
- **Observational Reference:** ENACTS rainfall data
- **Spatial Domain:** 49 lats × 61 lons (2,989 valid grid points via `jjas_seasonal_mask_0p25.nc`)
- **Verification Windows:** Days 1–15 and Days 16–30

### Processing Summary
- **Temporal Scope:** 2015–2022 (8 years)
- **Initializations:** 15 dates per year (May–July)
- **Total Potential Forecasts:** ~1,120,875 per window
- **Valid Forecasts Processed:** ~120,000+ unique member-forecast combinations per window
- **Climatological Reference:** Multi-year climatology (8 years) using day-of-year onset comparison

### Skill Scores: Verification Window (Days 1–15)

*(Lower is better for Brier Score and RPS; higher is better for AUC.)*

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
|--------|-------------------|-----------------------|-------------|
| **Fair Brier Score** | 0.1107 | 0.0953 | BSS = -0.162 |
| **Fair RPS** | 0.5439 | 0.4326 | RPSS = -0.257 |
| **AUC** | 0.692 | 0.819 | – |

**Bin-wise Fair Brier Skill Score (BSS):**

| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
|---|---|---|---|---|---|
| Days 1-5 | 0.0849 | 0.0907 | 0.064 | 0.845 | 0.870 |
| Days 6-10 | 0.1165 | 0.0961 | -0.213 | 0.693 | 0.813 |
| Days 11-15 | 0.1307 | 0.0990 | -0.320 | 0.533 | 0.768 |

> **Overall Fair BSS:** `-0.162` | **Overall Fair RPSS:** `-0.257`

The only positive skill is in the first five days (BSS = +6.4 %). Skill then falls below climatology and AUC drops from 0.85 to 0.53 by Days 11–15.

![Heatmap of Brier Skill Score and AUC for AIFS-ENS in day bins 1-5, 6-10 and 11-15. BSS is +6.4, -21 and -32 percent; AUC is 0.85, 0.69 and 0.53 with climatology values in brackets 0.87, 0.81, 0.77.](figures/skill_scores_heatmap_AIFS_ENS_1-15.png){alt="AIFS-ENS skill heatmap days 1-15"}

*Figure 7a. AIFS-ENS skill by 5-day bin, Days 1–15. Top row: BSS (%). Bottom row: AUC (climatology reference in brackets). Skill decays quickly with lead time.*

### Skill Scores: Verification Window (Days 16–30)

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
|--------|-------------------|-----------------------|-------------|
| **Fair Brier Score** | 0.0828 | 0.0659 | BSS = -0.258 |
| **Fair RPS** | 0.5413 | 0.3643 | RPSS = -0.486 |
| **AUC** | 0.501 | 0.805 | – |

**Bin-wise Fair Brier Skill Score (BSS):**

| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
|---|---|---|---|---|---|
| Days 16-20 | 0.1061 | 0.0826 | -0.285 | 0.502 | 0.778 |
| Days 21-25 | 0.0807 | 0.0644 | -0.254 | 0.500 | 0.808 |
| Days 26-30 | 0.0617 | 0.0507 | -0.218 | 0.500 | 0.821 |

> **Overall Fair BSS:** `-0.258` | **Overall Fair RPSS:** `-0.486`

![Heatmap of Brier Skill Score and AUC for AIFS-ENS in day bins 16-20, 21-25 and 26-30. BSS is -28, -25 and -22 percent; AUC is 0.5 in every bin against climatology values of 0.78, 0.81 and 0.82.](figures/skill_scores_heatmap_AIFS_ENS_16-30.png){alt="AIFS-ENS skill heatmap days 16-30"}

*Figure 7b. AIFS-ENS skill by 5-day bin, Days 16–30. BSS stays negative in every bin, and AUC is 0.5 throughout, meaning the ensemble cannot discriminate onset from non-onset.*

> **Watch out:** the BSS gets *less* negative from Days 16–20 to 26–30 (−28 % → −22 %) even though AUC is flat at 0.5. This is a base-rate effect, not real skill. Brier scores are lower later in the window simply because the event is rarer. Always read BSS together with AUC.

### Data Files

The skill tables above are generated from these CSV files, which you can open directly in a spreadsheet or load with `pandas`:

| File | Contents |
|------|----------|
| [`binned_skill_scores_AIFS_ENS_1-15.csv`](data/binned_skill_scores_AIFS_ENS_1-15.csv) | Fair BS, BSS and AUC per 5-day bin, Days 1–15 |
| [`binned_skill_scores_AIFS_ENS_16-30.csv`](data/binned_skill_scores_AIFS_ENS_16-30.csv) | Fair BS, BSS and AUC per 5-day bin, Days 16–30 |
| [`overall_skill_scores_AIFS_ENS_1-15.csv`](data/overall_skill_scores_AIFS_ENS_1-15.csv) | Whole-window BS, BSS, RPS, RPSS, AUC, Days 1–15 |
| [`overall_skill_scores_AIFS_ENS_16-30.csv`](data/overall_skill_scores_AIFS_ENS_16-30.csv) | Whole-window BS, BSS, RPS, RPSS, AUC, Days 16–30 |

```python
import pandas as pd

binned = pd.concat([
    pd.read_csv("data/binned_skill_scores_AIFS_ENS_1-15.csv"),
    pd.read_csv("data/binned_skill_scores_AIFS_ENS_16-30.csv"),
])
print(binned[["Bin", "Fair_Brier_Skill_Score", "AUC", "AUC_ref"]])
```

### Reliability Analysis

A reliability diagram plots the forecast probability (x) against how often the event actually occurred (y). Points on the dashed 1:1 line are perfectly reliable. Grey bars (log scale, right axis) show how many forecasts fall in each probability bin.

![Reliability diagram for AIFS-ENS days 1-15. The blue curve lies above the diagonal for forecast probabilities below about 0.7 and below it above that, ending near 0.8 observed frequency at forecast probability 0.96.](figures/reliability_AIFS_ENS_1-15.png){alt="Reliability diagram AIFS-ENS days 1-15"}

*Figure 8a. Reliability, Days 1–15. The curve is flatter than the diagonal. Near-zero forecasts verify about 10 % of the time (under-forecast), and forecasts near 1.0 verify only about 80 % of the time (over-forecast). The ensemble is **overconfident**. Most forecasts sit in the lowest probability bin.*

![Reliability diagram for AIFS-ENS days 16-30. Almost all forecasts are below 0.2 probability; observed frequency is about 0.08 in the lowest bin, falls to zero for forecast probabilities of 0.35 to 0.45, and the last point has very wide error bars.](figures/reliability_AIFS_ENS_16-30.png){alt="Reliability diagram AIFS-ENS days 16-30"}

*Figure 8b. Reliability, Days 16–30. Nearly all forecasts are below 0.3, the curve has no upward trend, and the last points have huge error bars from small samples. Probabilities carry almost no information, matching the AUC ≈ 0.5.*

| Window | Forecast Prob. Bin | N_Forecasts | Mean Forecast Prob. | Observed Reliability |
|--------|--------------------|-------------|---------------------|----------------------|
| **1–15 Days** | 0.0 – 0.1 | 111,238 | `0.006` | **`0.078`** |
| **16–30 Days**| 0.0 – 0.1 | 127,466 | `0.000` | **`0.081`** |

*Interpretation:* When the model predicts a near-zero probability of onset, onset still occurs roughly 8 % of the time. The ensemble is under-forecasting at the low end and, in Days 1–15, over-forecasting at the high end. Both are signs of poor calibration that a post-processing step such as IDR is designed to correct.

> **Check the figure:** the first point in Figure 8a looks closer to 0.10 than the 0.078 in the table. Ask participants why these might differ (bin edges, pooling, run version).

### Key Takeaways & Next Steps
1. **Mostly Negative Skill:** Apart from Days 1–5 (BSS = +6.4 %), AIFS-ENS probabilistic onset forecasts perform worse than the climatological baseline (negative BSS and RPSS in every other bin and in both overall windows).
2. **Skill Degradation with Lead Time:** AUC falls from 0.85 (Days 1–5) to 0.53 (Days 11–15) and sits at `0.50` throughout Days 16–30, indicating no discrimination skill beyond random chance at longer lead times.
3. **Reliability Bias:** The ensemble is overconfident. It assigns near-zero probabilities to events that occur ~8 % of the time and gives high probabilities (~0.96) to events that occur only ~80 % of the time.
4. **Recommended Next Steps:** 
   - Apply **ensemble calibration** (e.g., Isotonic Distributional Regression (IDR), or EMOS) to correct the under-forecasting bias.
   - Review onset detection thresholds and rainfall accumulation logic in the AIFS-ENS post-processing pipeline.
   - Investigate whether specific initialization dates or sub-regions are driving the bulk of the negative skill.

<<<<<<< HEAD
---

## Composite Metric Plots

Composite plots condense many runs into one figure so models, windows and metrics can be compared at a glance:

- **Portrait panel (Figure 6):** Δ MAE, Δ FAR and Δ MR for each deterministic model and window, relative to climatology.
- **Skill heatmaps (Figure 7):** BSS and AUC per 5-day lead bin for probabilistic models.
- **Extension:** the same layout can be repeated across wet-spell thresholds, dry-spell thresholds or regions, which is how the package supports the goal of comparing models across location, thresholds and lead time.

Read composite plots with care. They hide the spatial detail in the maps and the sample size behind each cell.

---

# Part 3: Preliminary Model Results

> These results are **preliminary**. The goal is to replicate and further investigate them. They use **2019–2024** and compare climatology, AIFS and GenCast at two resolutions. This is a different period and model set from the hands-on runs in Parts 1 and 2 (2015–2022), so do not expect the numbers to match.

<!-- Add the corresponding map and skill figures for these results when available. -->

## 0.25° Resolution

**Deterministic maps, Days 1–15 (2019–2024):** climatology, AIFS and GenCast. Deterministic scores improve over climatology, but the **miss rate is very high**.

**Deterministic maps, Days 16–30 (2019–2024):** climatology, AIFS and GenCast.

**GenCast probabilistic skill in 5-day bins:** Days 1–15 and Days 16–30.

| Window | AUC | BSS | RPSS |
|--------|-----|-----|------|
| 1–15 day | 0.85 | 7.4 % | 29.4 % |
| 1–30 day | 0.77 | 0.2 % | 6.8 % |

Probabilistic scores improve over climatology, especially for Days 1–15. However, **raw model probabilities are not well calibrated**, so calibration and blending have potential to improve them further.

## 1° Resolution

**Deterministic maps, Days 1–15 and 16–30 (2019–2024):** climatology, AIFS and GenCast. Results are similar to 0.25°, with high miss rates.

**GenCast probabilistic skill in 5-day bins:** Days 1–15 and Days 16–30.

| Window | AUC | BSS | RPSS |
|--------|-----|-----|------|
| 1–15 day | 0.88 | 17 % | 30 % |
| 1–30 day | 0.80 | 8.2 % | 8.2 % |

Improvements over climatology are higher at coarser resolution, but these estimates are **noisier** because the sample size is smaller.

### Discussion Questions

1. Why might skill against climatology look better at 1° than at 0.25°? Consider sample size and spatial averaging.
2. AIFS-ENS in Part 2 has negative BSS but GenCast here has positive BSS. List the differences in years, resolution, model and reference dataset that could explain this.
3. Both DET and PROB results show a high miss rate. What would you change in the onset definition (thresholds, tolerance) to test whether that is a model problem or a definition problem?

---

# Part 4: Other Concepts to Discuss When Benchmarking

| Concept | Why it matters |
|---------|----------------|
| **Same onset definition everywhere** | Different rules for observations and forecasts make comparisons meaningless. |
| **Threshold sensitivity** | Changing wet/dry thresholds can change ranking. Test several. |
| **Lead time and initialization dates** | Skill depends on when the forecast starts relative to onset. |
| **Matching tolerance** | A larger tolerance raises hits and hides timing error. Report it with every result. |
| **Reference choice** | Climatology, persistence or another model give different skill scores. |
| **Base rate** | BSS can move only because the event becomes rarer (see the watch-out in Part 2). |
| **Resolution and sample size** | Coarse grids give higher skill but noisier estimates. |
| **Spatial aggregation** | Grid-cell metrics differ from regional or national averages. |
| **Calibration and blending** | Post-processing (IDR, EMOS) can fix reliability but not missing discrimination. |
| **Separate DET and PROB tracks** | Metric families are not comparable. Never mix them in one run. |
| **Reproducibility** | Keep the config file, package version and data versions with every result. |

---

# Part 5: AI Almanac Exploration and Key Ingredients

### Objective
Explore the AI Almanac and gather feedback on the necessary components required to assess metrics and use cases in an intuitive, interactive manner.

### Activity
- Guide participants to explore the AI Almanac interface. *(Note: Ethiopia and India onset data are pre-loaded as working examples).*
- Have paired country groups share their ideas and feedback across both deterministic and probabilistic evaluation tracks.
=======
>>>>>>> 4ed739ed3a1f672dbd385d2284e7dcf3f88b2406

### Wrap-up Discussion
Conclude the session with a focused discussion on the **"key ingredients needed"** for successful, use-case-driven model assessment, ensuring that technical metrics translate into actionable agricultural insights.

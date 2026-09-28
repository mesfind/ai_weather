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

This lesson documents the complete **ROMP** (Rainy season Onset Metrics Package) / **MOMP** benchmarking workflow. It is used to evaluate AI weather forecast models (e.g., AIFS, FuXi, GraphCast, GenCast, AIFS-ENS) against observational rainfall data (e.g., CHIRPS) for Ethiopian rainy season onset prediction. The workflow covers both deterministic and probabilistic evaluation tracks, as well as the configuration system that drives them.

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
1. Understand the theoretical foundations of rainy season onset detection.
2. Configure and execute deterministic benchmarks (MAE, FAR, Miss Rate).
3. Configure and execute probabilistic benchmarks (Brier Score, Ranked Probability Score, AUC, Reliability).
4. Correctly utilize the swappable, per-run Python configuration system and CLI mode selection.
5. Apply and validate Isotonic Distributional Regression (IDR) calibration for probabilistic onset forecasts.
6. Diagnose and resolve common configuration, data, and pipeline errors.

---

##  Why Onset Matters

In Ethiopian agriculture, the **onset of the rainy season** (Kiremt: June–September) dictates planting dates for millions of smallholder farmers. A late or false onset signal can lead to crop failure from planting too early, lost growing days from planting too late, and regional food insecurity. 

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
| **MR** (Miss Rate) | $\frac{\text{misses}}{\text{hits} + \text{misses}}$ | Percentage of actual onsets that were missed. |

### 2. Probabilistic Track (Ensemble Forecasts)
| Metric | Interpretation |
|--------|----------------|
| **BS** (Brier Score) / **BSS** (Skill Score) | Mean squared error of probability forecasts. BSS represents improvement over a climatological baseline. *(Lower is better for BS; Higher is better for BSS)* |
| **RPS** (Ranked Probability Score) / **RPSS** | Measures the distance between the forecast and observed Cumulative Distribution Function (CDF) across ordered categories. Penalizes forecasts that are "farther" from the correct category. *(Lower is better for RPS)* |
| **AUC** (Area Under ROC Curve) | Measures discrimination ability: the probability that the model assigns a higher probability to a randomly chosen event case than to a non-event case. Range: 0 to 1. Perfect: 1.0. No skill: 0.5. *(Higher is better)* |
| **Reliability** (Calibration) | Measures the statistical consistency between forecast probabilities and observed frequencies. A perfectly reliable model predicts an event with 70% probability exactly 70% of the time it occurs. Typically visualized via a Reliability Diagram. |

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
| **AIFS-ENS** | Probabilistic (50 members) | ECMWF | ~50 km |
| **GenCast** | Probabilistic (Diffusion, 52 members) | Google DeepMind | ~50 km |

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

### AIFS Results
**Verification Window: Days 1–15**
| Year | TP | FP | FN | TN |
|------|----|----|----|----|
| 2015 | 193 | 170 | 344 | 480 |
| 2016 | 243 | 247 | 217 | 617 |
| 2017 | 347 | 443 | 132 | 475 |
| 2018 | 251 | 280 | 155 | 542 |
| 2019 | 994 | 1,362 | 355 | 1,620 |
| 2020 | 880 | 836 | 652 | 2,099 |
| 2021 | 758 | 1,354 | 625 | 3,991 |
| 2022 | 489 | 700 | 874 | 4,130 |

**Verification Window: Days 16–30**
| Year | TP | FP | FN | TN |
|------|----|----|----|----|
| 2015 | 22 | 48 | 372 | 382 |
| 2016 | 18 | 83 | 283 | 450 |
| 2017 | 40 | 63 | 233 | 271 |
| 2018 | 13 | 42 | 293 | 349 |
| 2019 | 199 | 350 | 663 | 763 |
| 2020 | 394 | 320 | 867 | 1,170 |
| 2021 | 387 | 684 | 1,052 | 2,493 |
| 2022 | 412 | 799 | 1,266 | 2,527 |

*Interpretation:* AIFS produced substantially more true positives in the Days 1–15 window. The longer-lead window (16–30 days) yielded significantly more misses, indicating weaker onset detection at extended lead times.

### FuXi Results
**Verification Window: Days 1–15**
| Year | TP | FP | FN | TN |
|------|----|----|----|----|
| 2015 | 138 | 130 | 418 | 501 |
| 2016 | 177 | 178 | 346 | 623 |
| 2017 | 268 | 351 | 232 | 546 |
| 2018 | 223 | 268 | 180 | 557 |
| 2019 | 183 | 248 | 209 | 376 |
| 2020 | 223 | 279 | 266 | 361 |
| 2021 | 160 | 319 | 250 | 892 |
| 2022 | 111 | 164 | 274 | 851 |

**Verification Window: Days 16–30**
| Year | TP | FP | FN | TN |
|------|----|----|----|----|
| 2015 | 10 | 44 | 485 | 380 |
| 2016 | 24 | 67 | 391 | 487 |
| 2017 | 28 | 46 | 363 | 341 |
| 2018 | 5 | 15 | 348 | 369 |
| 2019 | 11 | 29 | 372 | 173 |
| 2020 | 15 | 39 | 360 | 213 |
| 2021 | 47 | 87 | 405 | 603 |
| 2022 | 23 | 75 | 447 | 580 |

*Interpretation:* FuXi shows a marked reduction in true positives for the Days 16–30 window, with high miss counts across all years. Its long-lead onset detection is considerably weaker than its short-lead performance.

### GraphCast Results
**Verification Window: Days 1–15**
| Year | TP | FP | FN | TN |
|------|----|----|----|----|
| 2015 | 280 | 339 | 219 | 349 |
| 2016 | 263 | 444 | 139 | 478 |
| 2017 | 371 | 566 | 92 | 368 |
| 2018 | 248 | 395 | 117 | 468 |
| 2019 | 269 | 476 | 63 | 208 |
| 2020 | 306 | 426 | 114 | 283 |
| 2021 | 230 | 516 | 125 | 750 |
| 2022 | 192 | 318 | 153 | 737 |

*Interpretation:* Among the completed short-lead evaluations, GraphCast generally produced relatively few misses (particularly in 2017–2021). However, it also generated frequent false alarms, meaning its stronger detection rate came at the cost of lower forecast precision. *(Note: The log excerpt ends as GraphCast begins the Days 16–30 evaluation; complete long-lead results are unavailable in this run).*

### Deterministic Main Findings
- The ROMP run successfully completed AIFS and FuXi evaluations for both verification windows.
- Short-lead forecasts (Days 1–15) generally outperformed long-lead forecasts (Days 16–30).
- Extending the verification period substantially increased the number of missed onsets for both AIFS and FuXi.
- GraphCast demonstrated the strongest short-lead detection pattern but suffered from frequent false alarms.
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

---

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
| Metric | AIFS-ENS Forecast | Climatology Reference |
|--------|-------------------|-----------------------|
| **Fair Brier Score** | 0.0959 | **0.0677** *(Lower is better)* |
| **Fair RPS** | 0.4320 | **0.3164** *(Lower is better)* |
| **AUC** | 0.699 | **0.886** *(Higher is better)* |

**Bin-wise Fair Brier Skill Score (BSS):**
- Days 1–5: `-0.282`
- Days 6–10: `-0.394`
- Days 11–15: `-0.558`
> **Overall Fair BSS:** `-0.415` | **Overall Fair RPSS:** `-0.366`

### Skill Scores: Verification Window (Days 16–30)
| Metric | AIFS-ENS Forecast | Climatology Reference |
|--------|-------------------|-----------------------|
| **Fair Brier Score** | 0.0814 | **0.0533** *(Lower is better)* |
| **Fair RPS** | 0.5205 | **0.2936** *(Lower is better)* |
| **AUC** | 0.503 | **0.880** *(Higher is better)* |

**Bin-wise Fair Brier Skill Score (BSS):**
- Days 16–20: `-0.551`
- Days 21–25: `-0.521`
- Days 26–30: `-0.502`
> **Overall Fair BSS:** `-0.528` | **Overall Fair RPSS:** `-0.773`

### Reliability Analysis
The model exhibits a strong systematic tendency to **under-predict** the probability of onset.

| Window | Forecast Prob. Bin | N_Forecasts | Mean Forecast Prob. | Observed Reliability |
|--------|--------------------|-------------|---------------------|----------------------|
| **1–15 Days** | 0.0 – 0.1 | 111,238 | `0.006` | **`0.078`** |
| **16–30 Days**| 0.0 – 0.1 | 127,466 | `0.000` | **`0.081`** |

*Interpretation:* When the model predicts a near-zero probability of onset, the actual observed onset rate is approximately 8%. The ensemble is consistently under-forecasting onset events.

### Key Takeaways & Next Steps
1. **Negative Skill Across the Board:** AIFS-ENS probabilistic onset forecasts currently perform worse than the climatological baseline (consistently negative BSS and RPSS).
2. **Skill Degradation with Lead Time:** Forecast skill drops significantly in the 16–30 day window. The AUC approaches `0.50`, indicating virtually no discrimination skill beyond random chance at longer lead times.
3. **Reliability Bias:** The ensemble heavily under-predicts onset probabilities, assigning near-zero probabilities to events that occur ~8% of the time.
4. **Recommended Next Steps:** 
   - Apply **ensemble calibration** (e.g., Isometric Distribution Regression, or EMOS) to correct the under-forecasting bias.
   - Review onset detection thresholds and rainfall accumulation logic in the AIFS-ENS post-processing pipeline.
   - Investigate whether specific initialization dates or sub-regions are driving the bulk of the negative skill.

---

# Part 3: AI Almanac Exploration and Key Ingredients

### Objective
Explore the AI Almanac and gather feedback on the necessary components required to assess metrics and use cases in an intuitive, interactive manner.

### Activity
- Guide participants to explore the AI Almanac interface. *(Note: Ethiopia and India onset data are pre-loaded as working examples).*
- Have paired country groups share their ideas and feedback across both deterministic and probabilistic evaluation tracks.

### Wrap-up Discussion
Conclude the session with a focused discussion on the **"key ingredients needed"** for successful, use-case-driven model assessment, ensuring that technical metrics translate into actionable agricultural insights.

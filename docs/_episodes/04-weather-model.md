<<<<<<< HEAD
---
title: "Demo 4
teaching: 30
exercises: 60
questions:
  - "How do we systematically benchmark AI weather models against local observations for rainy season onset"
  - "How do deterministic and probabilistic evaluation tracks differ, and why must they remain strictly separated?"
  - "How do we configure, run, and troubleshoot the ROMP/MOMP pipeline reliably from the command line and within notebooks?"
  - "How does Isotonic Distributional Regression (IDR) calibration improve (or fail to improve) probabilistic onset forecasts, and how do we diagnose this?"
objectives:
  - "Explain why a reproducible onset benchmarking package is needed and what it can compare (location, thresholds, lead time, model)."
  - "Understand the onset detection algorithm and why it must be applied identically to observations and forecasts."
  - "Configure and run deterministic benchmarks (MAE, FAR, Miss Rate) and probabilistic benchmarks (BS, RPS, AUC, Reliability)."
  - "Interpret skill scores, spatial maps and composite plots relative to a climatological reference."
  - "Apply and diagnose Isotonic Distributional Regression (IDR) calibration on probabilistic onset forecasts."
  - "Recognize and resolve the most common configuration, data, and pipeline failure modes."
keypoints:
  - "Onset is derived identically from observations and forecasts using a wet-spell/dry-spell veto rule, not read directly from raw model output."
  - "Deterministic and probabilistic tracks use non-comparable metric families (e.g., FAR/MAE/MR vs. BS/RPS/AUC) and must never be mixed within a single evaluation run."
  - "Skill is always relative to a reference. Read FAR, MR and MAE together, and read BSS together with AUC and reliability."
  - "Results depend on explicit choices (thresholds, tolerance, resolution, period, reference). Report them with every score."
---

=======
>>>>>>> 8ac30f9b1273e07af15685044f6f74c0ec8f9d56
# The AI Weather Model Scorecard

This lesson documents the complete **ROMP** (Rainy season Onset Metrics Package) / **MOMP** benchmarking workflow. It is used to evaluate AI weather forecast models (AIFS, FuXi, GraphCast, GenCast, AIFS-ENS) against observational rainfall data (e.g., CHIRPS, ENACTS) for rainy season onset in Ethiopia's Kiremt season. The methodology is designed to be region-agnostic and easily transferable to other seasons and domains.

## Learning Objectives
By the end of this lesson, you will be able to:
1. Explain the motivation for the package and the four dimensions it compares: location, wet-spell threshold, dry-spell threshold, and lead time.
2. Describe how onset is detected and why the same rules must be applied to observations and forecasts.
3. Configure and execute deterministic benchmarks (MAE, FAR, Miss Rate).
4. Configure and execute probabilistic benchmarks (Brier Score, Ranked Probability Score, AUC, Reliability).
5. Interpret skill scores, spatial maps, reliability diagrams, and composite plots against climatology.
6. Explain what calibration (IDR) can and cannot fix.
7. Diagnose and resolve common configuration, data, and pipeline errors.


## Lesson Roadmap

| Section | Purpose |
| :--- | :--- |
| **Why Onset Matters** | The user need and the definition of onset. |
| **The ROMP Benchmarking Package** | Motivation, capabilities, workflow, and configuration. |
| **Core Concepts for Benchmarking** | Vocabulary needed to interpret any result. |
| **Benchmarking Metrics** | Deterministic and probabilistic metrics and skill scores. |
| **Models and Evaluation Setup** | What is compared, over which windows. |
| **Deterministic Evaluation** | Worked example with maps and skill vs. climatology. |
| **Probabilistic Evaluation** | Worked example with skill scores and reliability. |
| **Calibration with IDR** | Diagnosing and improving probabilities. |
| **Composite Metric Plots** | Summarising many results at once. |
| **Preliminary Model Results** | Reference results to replicate and investigate. |
| **Best Practices & Troubleshooting** | Making benchmarks defensible and fixing failures. |
| **Exercises & AI Almanac Activity** | Hands-on practice and feedback. |


## Why Onset Matters

In Ethiopian agriculture, the onset of the rainy season (Kiremt: June–September) dictates planting dates for millions of smallholder farmers. A late or false onset signal can lead to crop failure from planting too early, lost growing days from planting too late, and regional food insecurity.

![Figure 1: Mean rainy season onset date (day of year), 2003–2024. Onset arrives first in the southwest (mid-to-late May) and progressively later toward the north and east (July–August).](figures/fig1_mean_onset_ethiopia.png)
*Figure 1. Mean rainy season onset date across Ethiopia. This strong spatial gradient is why skill must be examined per grid cell and not only as a national average.*

**Onset is not read directly from raw model output.** To ensure genuine comparability, it is derived identically from daily rainfall data for observations, the reference model, and every forecast model using the following rules:

1. **Wet-spell trigger:** A candidate onset day requires at least `wet_init` mm of initial rainfall, followed by `wet_spell` consecutive days with ≥ `wet_threshold` mm/day.
2. **Dry-spell veto:** A candidate onset is invalidated if a dry spell (`dry_spell` consecutive days below `dry_threshold` mm/day) occurs within a `dry_extent`-day window afterward. This rejects false starts.

**Application:** These rules are applied per grid cell, per year, within a defined search window (`start_date` to `end_date`).


## The ROMP Benchmarking Package

### Motivation
* **Demand:** Model developers and forecasters need a reproducible, quantitative workflow for routinely evaluating model performance on onset.
* **Region- and threshold-agnostic:** The package is not tied to Ethiopia. It can be applied to Kiremt rains, other regions, and any chosen model.
* **Goal functionality:** Compare models across **location** (grid cell, region, country), **wet-spell thresholds**, **dry-spell thresholds**, and **lead times**.

### Design Capabilities

| Capability | What it means in practice |
| :--- | :--- |
| **Metrics-based evaluation** | Assumptions are objective, explicit, and controllable. |
| **Multi-model reforecasts** | Several models are evaluated in one run against the same reference. |
| **Multiple verification sources** | Observations can come from different datasets (e.g., CHIRPS, ENACTS). |
| **Custom onset definition** | Wet-spell and dry-spell rules are user parameters. |
| **Region-agnostic detection** | The same detection code runs for any domain and season window. |
| **Lead-time evaluation** | Skill is reported per verification window and per lead-time bin. |
| **Configuration-driven** | Each run is fully described by its config file, ensuring reproducibility. |

### ROMP Workflow
```text
config file  -->  load observations + model reforecasts
              -->  apply the SAME onset definition to both
              -->  match forecast onset to observed onset (per grid cell, year, init date)
              -->  compute metrics (DET or PROB track)
              -->  calculate skill scores vs. climatology (or named reference model)
              -->  generate maps, tables, heatmaps, and reliability diagrams
```

### ROMP Specifications

**1. Onset Definition (Set in the config)**
| Parameter | Meaning |
| :--- | :--- |
| `wet_init` | Minimum initial rainfall (mm) that starts a candidate onset. |
| `wet_spell`, `wet_threshold` | Number of consecutive wet days, and the mm/day that counts as wet. |
| `dry_spell`, `dry_threshold` | Length of a dry spell, and the mm/day below which a day is dry. |
| `dry_extent` | Window (days) after the candidate in which a dry spell vetoes the onset. |
| `start_date`, `end_date` | Search window for the season (e.g., Kiremt, June–September). |

**2. Data and Domain**
* **Observations/Reference:** CHIRPS, ENACTS, or another gridded product.
* **Forecasts:** Daily rainfall from each model's reforecasts, on a common grid (e.g., 0.25°).
* **Mask:** A seasonal mask (e.g., `jjas_seasonal_mask_0p25.nc`) restricts evaluation to grid points where a rainy season occurs.
* **Catalog:** Models are registered centrally in `BENCHMARK_MODEL_CATALOG`.

**3. Verification Setup**
* **Verification windows:** Days 1–15 and Days 16–30 after initialization.
* **Matching tolerance:** 3 days (Days 1–15) and 5 days (Days 16–30).
* **Reference for skill scores:** Climatology by default, or a named model.
* **Run mode:** `DET` or `PROB`, chosen in the config or overridden via CLI (`--mode prob`). *Never both in the same run.*

---

## Core Concepts for Benchmarking

### Forecasts, Reforecasts, and Lead Time
A **reforecast (hindcast)** is a forecast re-run for past dates with a fixed model version. Benchmarking uses reforecasts because they provide many years of forecasts with a consistent system, which is the only way to estimate skill for a seasonal event. 
* **Initialization date:** When the forecast starts. 
* **Lead time:** Days after initialization. Onset skill depends strongly on lead time and how close initialization is to the climatological onset date.

### Verification Data
Observed onset is derived from a gridded rainfall product. This is the "truth", but it has its own errors. Model and observations must be on a common grid and calendar. Very coarse grids are noisier to score but smoother to predict.

### Events, Tolerance, and Contingency Counts
Onset is a binary event per grid cell and year (did the season start?) with a timing attached (which day?). A **matching tolerance** decides how close in time a forecast onset must be to count as a hit. 
* Counts of hits, false alarms, misses, and correct negatives form the basis for FAR and MR. 
* MAE is computed *only* where both onsets exist, so it can look artificially good when a model rarely predicts onset. **Always read it with MR.**

### Ensembles and Probabilities
An ensemble is a set of forecasts (members) from slightly different initial conditions. The forecast probability is the fraction of members that produce an onset in a bin. Because finite members make probabilities noisy, **fair scores** adjust for ensemble size.

Three separate qualities of a probabilistic forecast are measured:
| Quality | Question | Metric |
| :--- | :--- | :--- |
| **Accuracy** | How close are the probabilities to what happened? | BS, RPS (and skill scores) |
| **Discrimination** | Can the forecast separate event from non-event cases? | AUC |
| **Reliability** | Do stated probabilities match observed frequencies? | Reliability diagram |

> **💡 Key Concept:** A forecast can discriminate well and still be unreliable. Reliability can be corrected by calibration. Discrimination *cannot*.

### Reference Forecasts and Skill
A model is only useful if it beats a reference that requires no model (default: climatology). Skill scores are relative to that reference. They change if the reference period or dataset changes, so **always document them**.

---

## Benchmarking Metrics

The pipeline enforces a strict separation between two evaluation tracks. Deterministic and probabilistic models produce fundamentally different outputs and require non-comparable metric families. **A single run must be exclusively one or the other.**

### 1. Deterministic Track (Single Forecast)

| Metric | Formula Concept | Interpretation |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\| \text{forecast onset} - \text{obs onset} \|$ | Average error in days. |
| **FAR** (False Alarm Ratio) | $\frac{\text{false alarms}}{\text{hits} + \text{false alarms}}$ | Percentage of predicted onsets that did not occur. |
| **MR** (Miss Rate) | $\frac{\text{misses}}{\text{hits} + \text{misses}}$ | Percentage of actual onsets that were missed. |

**How it works:** A forecast onset is a *hit* if it falls within the matching tolerance. Otherwise, it is a *false alarm* (forecast but not observed/too far off), and an observed onset with no matching forecast is a *miss*. Cells where neither has an onset are *correct negatives*.
* *Note:* A model that never predicts onset has 0% FAR but 100% MR. Climatology behaves like the "always predict" extreme.

### 2. Probabilistic Track (Ensemble Forecasts)

| Metric | Interpretation |
| :--- | :--- |
| **BS / BSS** (Brier Score / Skill) | Mean squared error of probability forecasts. BSS represents improvement over baseline. *(Lower is better for BS; Higher for BSS)* |
| **RPS / RPSS** (Ranked Prob. Score) | Distance between forecast and observed CDF across ordered categories. Penalizes forecasts "farther" from the correct category. *(Lower is better for RPS)* |
| **AUC** (Area Under ROC Curve) | Discrimination ability: probability that the model assigns a higher probability to a random event case than a non-event case. Range: 0 to 1. *(Higher is better; 0.5 = no skill)* |
| **Reliability** (Calibration) | Statistical consistency between forecast probabilities and observed frequencies. Visualized via a Reliability Diagram. |

### 3. Skill Score Definition
Raw metrics alone do not establish whether an AI model beats a naive baseline. Both tracks report a final Skill Score (SS) relative to a reference model:

$$
SS = 1 - \frac{\text{Metric}_{\text{model}}}{\text{Metric}_{\text{reference}}}
$$

**Interpretation:**
* $SS = 1$: Perfect forecast.
* $SS > 0$: The model outperforms the reference (positive skill).
* $SS = 0$: The model performs identically to the reference.
* $SS < 0$: The model performs worse than the reference (negative skill).

---

## Models and Evaluation Setup

### Models in the Benchmark
Model assignments to specific tracks are defined centrally in the `BENCHMARK_MODEL_CATALOG` within `config.py`.

| Model | Type | Origin | Resolution |
| :--- | :--- | :--- | :--- |
| **AIFS** | Deterministic | ECMWF | ~25 km (0.25°) |
| **FuXi** | Deterministic | Fudan University | ~25 km |
| **GraphCast** | Deterministic | Google DeepMind | ~25 km |
| **AIFS-ENS** | Probabilistic (50 members; 25 used) | ECMWF | ~25 km |
| **GenCast** | Probabilistic (Diffusion, 52 members) | Google DeepMind | ~25 km |

### Evaluation Windows
| Verification Window | Window Length | Matching Tolerance |
| :--- | :--- | :--- |
| Days 1–15 after initialization | 15 days | 3 days |
| Days 16–30 after initialization | 15 days | 5 days |

---

## Deterministic Evaluation

### ROMP Run Summary
* **Package:** Rainy Season Onset Metrics Package (ROMP), v0.0.1
* **Run Mode:** Deterministic (`DET`)
* **Models Evaluated:** AIFS, FuXi, GraphCast
* **Reference Dataset:** ENACTS
* **Evaluation Years:** 2015–2022
* **Spatial Grid:** 49 lats × 61 lons at 0.2° resolution

### How to Read the Spatial Metric Maps
For each model and verification window, the pipeline writes a three-panel map (`spatial_metrics_<model>_<window>.png`):

| Panel | Colour scale | What "good" looks like |
| :--- | :--- | :--- |
| **MAE** (in days) | White/cream → dark red (0–14+ days) | Light colours |
| **False Alarm Rate** (%) | White → dark red (0–100 %) | Light colours |
| **Miss Rate** (%) | White → dark blue (0–100 %) | Light colours |

> **⚠️ Watch Out:** Blank (white) grid cells inside the country outline mean the metric is undefined there (e.g., no matched onset events). Blank does *not* mean "perfect". Always compare against climatology reference maps.

### Model Results

#### AIFS Results
AIFS is the strongest deterministic model at short lead. Its errors grow quickly in Days 16–30.
![Figure 2a: AIFS, Days 1–15. MAE is low across most of the western and central highlands. Errors and misses concentrate in the east and northeast.](figures/fig2a_aifs_1_15.png)
![Figure 2b: AIFS, Days 16–30. MAE rises sharply almost everywhere, and the southwest false-alarm area becomes saturated.](figures/fig2b_aifs_16_30.png)

#### FuXi Results
FuXi rarely issues an onset, so it has few false alarms but a very high miss rate, especially in Days 16–30.
![Figure 3a: FuXi, Days 1–15. Many grid cells are blank. Where MAE is defined it is low in the west, but miss-rate is dominated by dark blue in the north/east.](figures/fig3a_fuxi_1_15.png)
![Figure 3b: FuXi, Days 16–30. Almost every cell is blank or dark blue in the miss-rate panel.](figures/fig3b_fuxi_16_30.png)

#### GraphCast Results
GraphCast detects onset well in Days 1–15 but pays for it with frequent false alarms.
![Figure 4a: GraphCast, Days 1–15. Miss rates are low over most of the country. The far-southwest false-alarm area is close to 100%.](figures/fig4a_graphcast_1_15.png)
![Figure 4b: GraphCast, Days 16–30. MAE is high across the north. False alarms are large in the northwest and southwest.](figures/fig4b_graphcast_16_30.png)

#### Climatology Reference Maps
![Figure 5a: Climatology reference, Days 1–15.](figures/fig5a_climatology_1_15.png)
![Figure 5b: Climatology reference, Days 16–30. Climatology has low miss rates but false alarms of nearly 100% across the west.](figures/fig5b_climatology_16_30.png)

### Deterministic Skill Relative to Climatology
![Figure 6: Portrait panel of delta MAE, FAR and MR for each deterministic model in each window relative to the reference.](figures/fig6_deterministic_skill_delta.png)
*Figure 6. Change (Δ) in MAE, FAR, and MR. Days 1–15: all three models reduce MAE by roughly 3.6–4.9 days. Days 16–30: the MAE advantage largely vanishes or reverses.*

> **💡 Instructor Note:** Figure 6 shows MAE improving over the reference in Days 1–15 for all three models. However, the *Model Skill Rankings* table below reports negative mean-MAE skill. These come from different summaries (per-window Δ vs. a single pooled score). Ask participants to find what could explain the difference (reference dataset, window, aggregation) before trusting either number.

### Deterministic Main Findings
1. Short-lead forecasts (Days 1–15) generally outperformed long-lead forecasts (Days 16–30).
2. Extending lead time increased MAE and misses; the MAE advantage over climatology largely disappeared.
3. GraphCast had the strongest short-lead detection but frequent false alarms. FuXi showed the opposite pattern.
4. **No single metric is enough.** FAR, MR, and MAE must be read together, and always against climatology.

### Model Skill Rankings (Original Test Run)
*Reference: Climatology = 0.0*

| Model | Mean MAE Skill | False Alarm Rate Skill | Miss Rate Skill | Overall Skill Score |
| :--- | :--- | :--- | :--- | :--- |
| **Climatology** | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| **AIFS_ENS** | 0.122734 | 0.473197 | -0.308828 | 0.095701 |
| **GenCast** | -0.043218 | -0.389341 | 0.364237 | -0.022774 |
| **AIFS** | -0.201814 | -0.275234 | 0.323369 | -0.051226 |
| **FuXi** | -0.271470 | 0.203178 | -0.262507 | -0.110266 |
| **GraphCast** | -0.290775 | -0.541115 | 0.329475 | -0.167472 |

---

## Probabilistic Evaluation (AIFS-ENS)

### Run Configuration
```bash
momp-run -p notebooks/config_et.in --mode prob
```
* **Model Evaluated:** AIFS-ENS (25 ensemble members)
* **Observational Reference:** ENACTS rainfall data
* **Spatial Domain:** 49 lats × 61 lons (2,989 valid grid points)
* **Temporal Scope:** 2015–2022 (8 years), 15 initializations per year (May–July)
* **Valid Forecasts Processed:** ~120,000+ unique member-forecast combinations per window

### Skill Scores: Verification Window (Days 1–15)
*(Lower is better for Brier Score and RPS; higher is better for AUC.)*

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
| :--- | :--- | :--- | :--- |
| **Fair Brier Score** | 0.1107 | 0.0953 | **BSS = -0.162** |
| **Fair RPS** | 0.5439 | 0.4326 | **RPSS = -0.257** |
| **AUC** | 0.692 | 0.819 | – |

**Bin-wise Fair Brier Skill Score (BSS):**
| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Days 1-5** | 0.0849 | 0.0907 | **0.064** | 0.845 | 0.870 |
| **Days 6-10** | 0.1165 | 0.0961 | **-0.213** | 0.693 | 0.813 |
| **Days 11-15** | 0.1307 | 0.0990 | **-0.320** | 0.533 | 0.768 |

![Figure 7a: AIFS-ENS skill by 5-day bin, Days 1–15. Top row: BSS (%). Bottom row: AUC. Skill decays quickly with lead time.](figures/fig7a_aifs_ens_skill_1_15.png)
*The only positive skill is in the first five days (BSS = +6.4%). Skill then falls below climatology and AUC drops from 0.85 to 0.53 by Days 11–15.*

### Skill Scores: Verification Window (Days 16–30)

| Metric | AIFS-ENS Forecast | Climatology Reference | Skill Score |
| :--- | :--- | :--- | :--- |
| **Fair Brier Score** | 0.0828 | 0.0659 | **BSS = -0.258** |
| **Fair RPS** | 0.5413 | 0.3643 | **RPSS = -0.486** |
| **AUC** | 0.501 | 0.805 | – |

**Bin-wise Fair Brier Skill Score (BSS):**
| Bin | Fair BS (forecast) | Fair BS (climatology) | Fair BSS | AUC | AUC (climatology) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Days 16-20** | 0.1061 | 0.0826 | **-0.285** | 0.502 | 0.778 |
| **Days 21-25** | 0.0807 | 0.0644 | **-0.254** | 0.500 | 0.808 |
| **Days 26-30** | 0.0617 | 0.0507 | **-0.218** | 0.500 | 0.821 |

![Figure 7b: AIFS-ENS skill by 5-day bin, Days 16–30. BSS stays negative in every bin, and AUC is 0.5 throughout.](figures/fig7b_aifs_ens_skill_16_30.png)

> **⚠️ Watch Out:** The BSS gets less negative from Days 16–20 to 26–30 (−28% → −22%) even though AUC is flat at 0.5. This is a **base-rate effect**, not real skill. Brier scores are lower later in the window simply because the event is rarer. **Always read BSS together with AUC.**

### Data Files
The skill tables are generated from these CSV files:
* `binned_skill_scores_AIFS_ENS_1-15.csv`: Fair BS, BSS, and AUC per 5-day bin, Days 1–15.
* `binned_skill_scores_AIFS_ENS_16-30.csv`: Fair BS, BSS, and AUC per 5-day bin, Days 16–30.
* `overall_skill_scores_AIFS_ENS_1-15.csv`: Whole-window BS, BSS, RPS, RPSS, AUC, Days 1–15.
* `overall_skill_scores_AIFS_ENS_16-30.csv`: Whole-window BS, BSS, RPS, RPSS, AUC, Days 16–30.

```python
import pandas as pd

binned = pd.concat([
    pd.read_csv("../../data/ROMP_OUT/et/output/binned_skill_scores_AIFS_ENS_1-15.csv"),
    pd.read_csv("../../data/ROMP_OUT/et/output/binned_skill_scores_AIFS_ENS_16-30.csv"),
])
print(binned[["Bin", "Fair_Brier_Skill_Score", "AUC", "AUC_ref"]])
```

### Reliability Analysis
A reliability diagram plots the forecast probability (x) against how often the event actually occurred (y). Points on the dashed 1:1 line are perfectly reliable.

![Figure 8a: Reliability, Days 1–15. The curve is flatter than the diagonal. The ensemble is overconfident.](figures/fig8a_reliability_1_15.png)
![Figure 8b: Reliability, Days 16–30. Nearly all forecasts are below 0.3, the curve has no upward trend. Probabilities carry almost no information.](figures/fig8b_reliability_16_30.png)

| Window | Forecast Prob. Bin | N_Forecasts | Mean Forecast Prob. | Observed Reliability |
| :--- | :--- | :--- | :--- | :--- |
| **1–15 Days** | 0.0 – 0.1 | 111,238 | 0.006 | 0.078 |
| **16–30 Days** | 0.0 – 0.1 | 127,466 | 0.000 | 0.081 |

*Interpretation:* When the model predicts a near-zero probability of onset, onset still occurs roughly 8% of the time. The ensemble is under-forecasting at the low end and over-forecasting at the high end (Days 1-15). Both are signs of poor calibration.

### Key Takeaways & Next Steps
1. **Mostly Negative Skill:** Apart from Days 1–5 (BSS = +6.4%), AIFS-ENS performs worse than the climatological baseline.
2. **Skill Degradation:** AUC falls from 0.85 (Days 1–5) to 0.53 (Days 11–15) and sits at `0.50` throughout Days 16–30.
3. **Reliability Bias:** The ensemble is overconfident.
4. **Recommended Next Steps:** Apply ensemble calibration (e.g., IDR), review onset detection thresholds, and investigate specific initialization dates driving negative skill.

---

## Calibration with IDR

**Isotonic Distributional Regression (IDR)** is a non-parametric post-processing method. It learns a monotone mapping from the raw ensemble forecast to a calibrated predictive distribution.

### What IDR Can and Cannot Do

| IDR can | IDR cannot |
| :--- | :--- |
| Correct systematic overconfidence and bias in probabilities | Create discrimination that the raw ensemble does not have |
| Improve reliability, and often BSS and RPSS | Rescue a lead time where AUC is already ≈ 0.5 |
| Work without assuming a fixed distribution shape | Work well with very few training samples |

### How to Diagnose Its Effect
Compare raw and calibrated forecasts on the same verification data:
* **Reliability diagram:** The curve should move toward the 1:1 line.
* **BSS and RPSS:** These should increase where the raw forecasts were overconfident.
* **AUC:** This should change *little*. A large change suggests a problem in the workflow.
* **Sample size:** Check error bars and the number of forecasts per bin.

> **📌 Best Practice:** Train and test on different years (e.g., leave-one-year-out cross-validation). Calibrating and scoring on the same years overstates skill. Calibrate per lead-time bin.

---

## Composite Metric Plots

Composite plots condense many runs into one figure so models, windows, and metrics can be compared at a glance:
* **Portrait panel (Figure 6):** Δ MAE, Δ FAR, and Δ MR for each deterministic model and window.
* **Skill heatmaps (Figure 7):** BSS and AUC per 5-day lead bin for probabilistic models.

> **💡 Tip:** Read composite plots with care. They hide the spatial detail in the maps and the sample size behind each cell.

---

## Preliminary Model Results

*These results are preliminary (2019–2024) and compare climatology, AIFS, and GenCast at two resolutions. Do not expect the numbers to match the 2015-2022 hands-on runs.*

### 0.25° Resolution
**GenCast probabilistic skill in 5-day bins:**

| Window | AUC | BSS | RPSS |
| :--- | :--- | :--- | :--- |
| **1–15 day** | 0.85 | 7.4 % | 29.4 % |
| **1–30 day** | 0.77 | 0.2 % | 6.8 % |

*Probabilistic scores improve over climatology, especially for Days 1–15. Raw model probabilities are not well calibrated, so calibration has potential to improve them further.*

### 1° Resolution
**GenCast probabilistic skill in 5-day bins:**

| Window | AUC | BSS | RPSS |
| :--- | :--- | :--- | :--- |
| **1–15 day** | 0.88 | 17 % | 30 % |
| **1–30 day** | 0.80 | 8.2 % | 8.2 % |

*Improvements over climatology are higher at coarser resolution, but these estimates are noisier because the sample size is smaller.*

### Discussion Questions
1. Why might skill against climatology look better at 1° than at 0.25°? Consider sample size and spatial averaging.
2. AIFS-ENS in the probabilistic evaluation has negative BSS but GenCast here has positive BSS. List the differences in years, resolution, model, and reference dataset that could explain this.
3. Both DET and PROB results show a high miss rate. What would you change in the onset definition to test whether that is a model problem or a definition problem?

---

## Best Practices and Troubleshooting

### Best Practices for Defensible Benchmarks

| Concept | Why it matters |
| :--- | :--- |
| **Same onset definition everywhere** | Different rules for observations and forecasts make comparisons meaningless. |
| **Threshold sensitivity** | Changing wet/dry thresholds can change ranking. Test several. |
| **Matching tolerance** | A larger tolerance raises hits and hides timing error. Report it with every result. |
| **Reference choice** | Climatology, persistence, or another model give different skill scores. |
| **Base rate** | BSS can move only because the event becomes rarer. Read BSS with AUC. |
| **Separate DET and PROB tracks** | Metric families are not comparable. Never mix them in one run. |
| **Reproducibility** | Keep the config file, package version, and data versions with every result. |

### Reporting Checklist
Every benchmark result should state:
- [ ] Onset definition parameters (`wet_init`, `wet_spell`, `wet_threshold`, `dry_spell`, `dry_threshold`, `dry_extent`, search window)
- [ ] Verification dataset and its resolution
- [ ] Model, reforecast period, initialization dates, and number of members
- [ ] Lead-time windows and matching tolerance
- [ ] Reference used for skill (and its period)
- [ ] Track (`DET` or `PROB`), and whether forecasts were calibrated
- [ ] Package version and the config file used

### Troubleshooting Common Failure Modes

| Symptom | Likely cause | What to check |
| :--- | :--- | :--- |
| Run stops / meaningless results after switching tracks | DET and PROB mixed in one run | Set the mode in the config or with `--mode`, and run each track separately. |
| Model is missing from the output | Not registered for that track | Check `BENCHMARK_MODEL_CATALOG` in `config.py`. |
| Many blank (white) grid cells | No onset detected or no matched events | Check onset thresholds, search window, mask, data coverage. |
| Extremely high miss rate for one model | Model rarely reaches the wet-spell threshold | Check rainfall distribution vs. observations, `wet_threshold`. |
| Very high false-alarm rate in the wet southwest | Season starts early there; model triggers too often | Compare against the climatology map. |
| Grid or shape errors when loading data | Forecast and observation grids/masks differ | Ensure common grid, mask file, and resolution. |
| Positive BSS but AUC ≈ 0.5 | Base-rate effect, not real skill | Read BSS together with AUC and reliability. |

---

## Exercises

### Exercise 1: Read the Configuration (10 min)
1. Open `notebooks/config_et.in`.
2. Identify the onset parameters, the verification windows, the matching tolerance, and the run mode.
3. Predict what happens to FAR and MR if `wet_threshold` is increased. Explain your reasoning.

### Exercise 2: Interpret the Deterministic Maps (15 min)
1. Use Figures 2–5. Where is the miss rate highest for AIFS in Days 1–15, and why might onset be harder to forecast there?
2. Compare FuXi and GraphCast. Which one over-predicts onset and which one under-predicts it?
3. Why can a model with a lower FAR than climatology still be a poor forecast?

### Exercise 3: Reconcile Two Summaries (10 min)
Figure 6 shows MAE improvements over the reference for all models in Days 1–15, but the skill ranking table shows negative mean-MAE skill. List at least three reasons the two could differ (reference dataset, years, aggregation, run version) and describe what you would check.

### Exercise 4: Compute a Skill Score by Hand (10 min)
1. Open `binned_skill_scores_AIFS_ENS_1-15.csv`.
2. For the Days 1–5 bin, compute $1 - (BS_{\text{forecast}} / BS_{\text{climatology}})$ from the two Brier score columns.
3. Compare with the `Fair_Brier_Skill_Score` column.
4. Repeat for Days 11–15 and explain the sign.

<details>
<summary><b>Check your answer</b></summary>
<br>
Days 1–5: $1 - (0.0849 / 0.0907) \approx 0.064$ (BSS ≈ +6.4%).<br>
Days 11–15: $1 - (0.1307 / 0.0990) \approx -0.32$, so the ensemble is worse than climatology because its Brier score is higher.
</details>

### Exercise 5: Diagnose Reliability (10 min)
1. Use Figures 8a and 8b. Is the Days 1–15 ensemble over- or under-confident? Give the evidence from the curve.
2. Why are the points in Figure 8b so uncertain?
3. Would calibration help more in Days 1–15 or Days 16–30? Justify with AUC.

### Exercise 6: Design a Benchmark (5 min, discussion)
You must compare two new models for a different country. Using the reporting checklist, list the five decisions you must make before running the package.



## AI Almanac Exploration and Feedback

**Objective:** Explore the AI Almanac and gather feedback on the necessary components required to assess metrics and use cases in an intuitive, interactive manner.

**Activity:**
1. Guide participants to explore the AI Almanac interface. *(Note: Ethiopia and India onset data are pre-loaded as working examples).*
2. Have paired country groups share their ideas and feedback across both deterministic and probabilistic evaluation tracks.



## Wrap-up Discussion & Summary

* **Onset is a derived quantity.** It must be detected with the same wet-spell and dry-spell rules in observations and forecasts.
* **ROMP is configuration-driven**, so results can be reproduced and compared across location, thresholds, lead time, and models.
* **Deterministic and probabilistic tracks are separate.** Their metrics cannot be compared directly.
* **Skill exists only relative to a reference.** In the examples, models beat climatology at short lead and lose that advantage by Days 16–30.
* **Read metrics together:** FAR with MR and MAE; BSS with AUC and reliability.
* **Calibration (IDR) can improve reliability**, but it cannot create missing discrimination.
* **Resolution, period, tolerance, and sample size all affect scores.** Document them with every result.


## Glossary

| Term | Meaning |
| :--- | :--- |
| **Onset** | Start of the rainy season, detected from daily rainfall with wet-spell and dry-spell rules. |
| **Reforecast** | Forecast re-run for past dates with a fixed model version. |
| **Lead time** | Days between initialization and the forecast day. |
| **Hit / False alarm / Miss** | Forecast onset within tolerance of observed / forecast but not observed / observed but not forecast. |
| **MAE** | Mean absolute timing error of matched onsets, in days. |
| **FAR** | Fraction of forecast onsets that did not occur. |
| **MR** | Fraction of observed onsets that were not forecast. |
| **BS / BSS** | Brier Score of a yes/no probability forecast, and its skill relative to a reference. |
| **RPS / RPSS** | Ranked Probability Score over ordered onset-date categories, and its skill. |
| **AUC** | Area under the ROC curve. 0.5 means no discrimination and 1.0 is perfect. |
| **Reliability** | Agreement between forecast probability and observed frequency. |
| **Fair score** | Score corrected for the finite number of ensemble members. |
| **Climatology** | Reference forecast built from past observed onset dates. |
| **IDR** | Isotonic Distributional Regression, a monotone non-parametric calibration method. |
| **ROMP / MOMP** | Rainy season Onset Metrics Package / the benchmarking workflow it implements. |

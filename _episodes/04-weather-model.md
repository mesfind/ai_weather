---
title: Demo 4
teaching: 30
exercises: 15
questions:
- "How do we systematically benchmark AI weather models against local observations for rainy season onset?"
- "How do deterministic and probabilistic evaluation tracks differ, and why must they stay strictly separated?"
- "How do we configure, run, and troubleshoot the ROMP/MOMP pipeline reliably from the command line and from notebooks?"
- How does IDR calibration improve (or fail to improve) probabilistic onset forecasts, and how do we tell which?
objectives:
- "Understand the onset detection algorithm and why it is applied identically to observations and forecasts."
- "Configure and run deterministic benchmarks (MAE, FAR, Miss Rate) and probabilistic benchmarks (BS, RPS, AUC, Reliability).,Use the swappable, per-run Python configuration system correctly, including CLI mode selection."
- "Apply and diagnose Isotonic Distributional Regression (IDR) calibration on probabilistic onset forecasts.,Recognize and fix the most common configuration, data, and pipeline failure modes."
keypoints:
- "Model selection must be driven by specific use-case requirements and relevant evaluation metrics."
- "Interactive tools like the AI Almanac facilitate cross-track collaboration and practical feedback."
- "Pairing countries across weather and agricultural tracks ensures diverse and robust metric evaluation."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# The AI Weather Model Scorecard

## Evaluating AI Candidate Models 

### Part 1: The AI Weather Model Scorecard
- **Objective:**
     - Discuss how to choose between models across evaluation metrics based on their specific use case to inform definition and target forecast.
     - Identify top AI model candidates to further improve skill and ways they are getting the forecast wrong to inform bias correction techniques applied
     - Identify locations of interest where AI models may have some skills at detecting onset this season to inform where dissemination opportunities are possible

List of Models are"
 - Deterministic- AIFS, Graphcaset and FuXi,
 - Probabilistic- AIFS ensemble , GenCast


## Skill Score Definition


To calculate Skill Scores, we use the climatology model as the reference baseline. A Skill Score (SS) is defined as:

\[
SS = 1 - \frac{\text{Metric}_{\text{model}}}{\text{Metric}_{\text{climatology}}}
\]

## Interpretation

- \(SS = 1\): Perfect forecast.  
- \(SS > 0\): The model outperforms climatology (positive skill).  
- \(SS = 0\): The model performs identically to climatology.  
- \(SS < 0\): The model performs worse than climatology (negative skill).

##  Metrics for probablistic forecast evaluation

**1. Brier Score**

The **Brier Score (BS)** is the mean squared difference between predicted probabilities and actual outcomes. It measures the accuracy of probabilistic predictions for binary events.

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2$$

Where:
- $p_i$ = predicted probability for forecast $i$
- $o_i$ = observed outcome (0 or 1) for forecast $i$
- $N$ = total number of forecasts

### Key Properties:
- **Range**: 0 to 1
- **Perfect score**: 0 (all forecasts are perfectly confident and correct)
- **Worst score**: 1 (completely wrong confident forecasts)
- **Lower is better**

---

**Ranked Probability Score (RPS)**

The **Ranked Probability Score (RPS)** extends the Brier Score to **multiple categories**. It's ideal when your event can fall into multiple ordered bins (e.g., "Days 1-5", "Days 6-10", "Days 11-15").

$$\text{RPS} = \frac{1}{K-1} \sum_{k=1}^{K-1} \left( \sum_{j=1}^{k} p_j - \sum_{j=1}^{k} o_j \right)^2$$

Where:
- $K$ = number of categories
- $p_j$ = predicted probability for category $j$
- $o_j$ = 1 if event occurred in category $j$, 0 otherwise

### Key Properties:
- **Range**: 0 to 1 (typically, can exceed 1 in extreme cases)
- **Perfect score**: 0
- **Sensitive to distance**: Penalizes forecasts more when they're "farther" from the correct category
- **Lower is better**

---

## Deterministic Evaluation Setup 

Two verification windows were evaluated for each model

## ROMP Run Summary

### Configuration

- **Package:** Rainy Season Onset Metrics Package (ROMP), version 0.0.1.
- **Run mode:** Deterministic (`DET`).
- **Project:** Test ROMP run with sample data.
- **Start time:** 2026-09-28 13:42:33.
- **Models evaluated:** AIFS, FuXi, and GraphCast.
- **Reference/observation dataset:** ENACTS.
- **Evaluation years:** 2015–2022.
- **Computational resources:** 6 cores used from 10 available CPUs.
- **Grid:** 49 latitudes × 61 longitudes at 0.2° resolution.
- **Spatial products:** FAR, miss rate, yearly MAE, and mean MAE maps.
- **CMZ averages:** Not calculated because the 0.2° resolution is unsupported.

## Evaluation Setup

Two verification windows were evaluated for each model:

| Verification window | Window length | Matching tolerance |
|---|---:|---:|
| Days 1–15 after initialization | 15 days | 3 days |
| Days 16–30 after initialization | 30 days | 5 days |

The log confirms that the following spatial NetCDF and PNG products were successfully created:

- `spatial_metrics_AIFS_1-15.nc`
- `spatial_metrics_AIFS_16-30.nc`
- `spatial_metrics_FuXi_1-15.nc`
- `spatial_metrics_FuXi_16-30.nc`
- `spatial_metrics_GraphCast_1-15.nc`
- The corresponding spatial metric figures.

## AIFS Results

### Verification window: Days 1–15

| Year | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|
| 2015 | 193 | 170 | 344 | 480 |
| 2016 | 243 | 247 | 217 | 617 |
| 2017 | 347 | 443 | 132 | 475 |
| 2018 | 251 | 280 | 155 | 542 |
| 2019 | 994 | 1,362 | 355 | 1,620 |
| 2020 | 880 | 836 | 652 | 2,099 |
| 2021 | 758 | 1,354 | 625 | 3,991 |
| 2022 | 489 | 700 | 874 | 4,130 |

### Verification window: Days 16–30

| Year | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|
| 2015 | 22 | 48 | 372 | 382 |
| 2016 | 18 | 83 | 283 | 450 |
| 2017 | 40 | 63 | 233 | 271 |
| 2018 | 13 | 42 | 293 | 349 |
| 2019 | 199 | 350 | 663 | 763 |
| 2020 | 394 | 320 | 867 | 1,170 |
| 2021 | 387 | 684 | 1,052 | 2,493 |
| 2022 | 412 | 799 | 1,266 | 2,527 |

**Interpretation:** AIFS produced substantially more true positives in the days 1–15 window than in the days 16–30 window. The longer-lead window also produced many more misses, indicating weaker onset detection at extended lead times.

## FuXi Results

### Verification window: Days 1–15

| Year | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|
| 2015 | 138 | 130 | 418 | 501 |
| 2016 | 177 | 178 | 346 | 623 |
| 2017 | 268 | 351 | 232 | 546 |
| 2018 | 223 | 268 | 180 | 557 |
| 2019 | 183 | 248 | 209 | 376 |
| 2020 | 223 | 279 | 266 | 361 |
| 2021 | 160 | 319 | 250 | 892 |
| 2022 | 111 | 164 | 274 | 851 |

### Verification window: Days 16–30

| Year | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|
| 2015 | 10 | 44 | 485 | 380 |
| 2016 | 24 | 67 | 391 | 487 |
| 2017 | 28 | 46 | 363 | 341 |
| 2018 | 5 | 15 | 348 | 369 |
| 2019 | 11 | 29 | 372 | 173 |
| 2020 | 15 | 39 | 360 | 213 |
| 2021 | 47 | 87 | 405 | 603 |
| 2022 | 23 | 75 | 447 | 580 |

**Interpretation:** FuXi shows a marked reduction in true positives for the days 16–30 window, with high miss counts in every year. Its long-lead onset detection is therefore considerably weaker than its short-lead performance.

## GraphCast Results

### Verification window: Days 1–15

| Year | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|
| 2015 | 280 | 339 | 219 | 349 |
| 2016 | 263 | 444 | 139 | 478 |
| 2017 | 371 | 566 | 92 | 368 |
| 2018 | 248 | 395 | 117 | 468 |
| 2019 | 269 | 476 | 63 | 208 |
| 2020 | 306 | 426 | 114 | 283 |
| 2021 | 230 | 516 | 125 | 750 |
| 2022 | 192 | 318 | 153 | 737 |

**Interpretation:** Among the completed short-lead evaluations, GraphCast generally produced relatively few misses, particularly in 2017–2021. However, it also generated many false alarms, so its stronger detection rate came at the cost of lower forecast precision.

The log excerpt ends while GraphCast is beginning the **days 16–30** evaluation; complete GraphCast long-lead results are not available in the supplied run output. citefile:1

## Main Findings

- The ROMP run completed the AIFS and FuXi evaluations for both verification windows.
- GraphCast days 1–15 evaluation completed, but its days 16–30 evaluation is incomplete in the supplied log.
- Short-lead forecasts generally performed better than long-lead forecasts.
- Extending the verification period from days 1–15 to days 16–30 substantially increased the number of missed onsets for both AIFS and FuXi.
- GraphCast had the strongest short-lead detection pattern among the completed results, but it produced frequent false alarms.
- All spatial metric files were saved successfully for the completed model/window combinations.
- No fatal error or traceback appears in the supplied output.


# ROMP Probabilistic Onset Evaluation: AIFS_ENS

## Run Configuration

- **Command Executed**: `momp-run -p notebooks/config_et.in --mode prob`
- **Model Evaluated**: AIFS_ENS (25 ensemble members)
- **Evaluation Mode**: Probabilistic (CLI override)
- **Observational Reference**: ENACTS rainfall data
- **Spatial Domain**: 49 lats × 61 lons (2,989 valid grid points via `jjas_seasonal_mask_0p25.nc`)
- **Verification Windows**: Days 1–15 and Days 16–30

---

## Processing Summary

- **Temporal Scope**: 2015–2022 (8 years)
- **Initializations**: 15 dates per year (May–July)
- **Total Potential Forecasts**: ~1,120,875 per window
- **Valid Forecasts Processed**: ~120,000+ unique member-forecast combinations per window
- **Climatological Reference**: Multi-year climatology (8 years) using day-of-year onset comparison

---

## Skill Scores: Verification Window (Days 1–15)

| Metric | AIFS_ENS Forecast | Climatology Reference |
|--------|-------------------|-----------------------|
| **Fair Brier Score** | 0.0959 | **0.0677** *(Lower is better)* |
| **Fair RPS** | 0.4320 | **0.3164** *(Lower is better)* |
| **AUC** | 0.699 | **0.886** *(Higher is better)* |

### Bin-wise Fair Brier Skill Score (BSS)
- **Days 1–5**: `-0.282`
- **Days 6–10**: `-0.394`
- **Days 11–15**: `-0.558`

> **Overall Fair BSS**: `-0.415` &nbsp;|&nbsp; **Overall Fair RPSS**: `-0.366`

---

## Skill Scores: Verification Window (Days 16–30)

| Metric | AIFS_ENS Forecast | Climatology Reference |
|--------|-------------------|-----------------------|
| **Fair Brier Score** | 0.0814 | **0.0533** *(Lower is better)* |
| **Fair RPS** | 0.5205 | **0.2936** *(Lower is better)* |
| **AUC** | 0.503 | **0.880** *(Higher is better)* |

### Bin-wise Fair Brier Skill Score (BSS)
- **Days 16–20**: `-0.551`
- **Days 21–25**: `-0.521`
- **Days 26–30**: `-0.502`

> **Overall Fair BSS**: `-0.528` &nbsp;|&nbsp; **Overall Fair RPSS**: `-0.773`

---

## Reliability Analysis

The model exhibits a strong systematic tendency to **under-predict** the probability of onset.

| Window | Forecast Prob. Bin | N_Forecasts | Mean Forecast Prob. | Observed Reliability |
|--------|--------------------|-------------|---------------------|----------------------|
| **1–15 Days** | 0.0 – 0.1 | 111,238 | `0.006` | **`0.078`** |
| **16–30 Days**| 0.0 – 0.1 | 127,466 | `0.000` | **`0.081`** |

- **Interpretation**: When the model predicts a near-zero probability of onset, the actual observed onset rate is ~8%. The ensemble is consistently underforecasting onset events.

---

## Key Takeaways & Next Steps

1. **Negative Skill Across the Board**  
   AIFS_ENS probabilistic onset forecasts currently perform **worse than the climatological baseline** (consistently negative BSS and RPSS).
2. **Skill Degradation with Lead Time**  
   Forecast skill drops significantly in the 16–30 day window. The AUC approaches `0.50`, indicating virtually no discrimination skill beyond random chance at longer lead times.
3. **Reliability Bias**  
   The ensemble heavily under-predicts onset probabilities, assigning near-zero probabilities to events that occur ~8% of the time.
4. **Recommended Next Steps**  
   - Apply **ensemble calibration** (e.g., Logistic Regression, EMOS) to correct the underforecasting bias.
   - Review onset detection thresholds and rainfall accumulation logic in the AIFS_ENS post-processing pipeline.
   - Investigate if specific initialization dates or sub-regions are driving the bulk of the negative skill.

**3. Area Under the ROC Curve (AUC)**

The **Area Under the ROC Curve (AUC)** measures how well a forecast can **discriminate** between events that occur and those that don't.

Think of it as answering: "If I pick a random case where the event occurred and a random case where it didn't, what's the probability that my forecast gives a higher probability to the event case?"

### Key Properties:
- **Range**: 0 to 1
- **Perfect score**: 1.0 (perfect discrimination)
- **No skill**: 0.5 (random guessing)
- **Higher is better**

## Model Skill Rankings (Reference: Climatology = 0.0)

| Model       | mean_mae_skill | false_alarm_rate_skill | miss_rate_skill | Overall_Skill_Score |
|-------------|----------------|------------------------|-----------------|---------------------|
| climatology | 0.000000       | 0.000000               | 0.000000        | 0.000000            |
| AIFS_ENS    | 0.122734       | 0.473197               | -0.308828       | 0.095701            |
| gencast     | -0.043218      | -0.389341              | 0.364237        | -0.022774           |
| AIFS        | -0.201814      | -0.275234              | 0.323369        | -0.051226           |
| fuxi        | -0.271470      | 0.203178               | -0.262507       | -0.110266           |
| graphcast   | -0.290775      | -0.541115              | 0.329475        | -0.167472           |


### Part 2: AI Almanac Exploration and Key Ingredients
- **Objective:** Explore the AI Almanac and gather feedback on necessary components to assess metrics and use cases in an easy, interactive way.
- **Activity:** 
  - Guide participants to explore the AI Almanac (Note: Ethiopia and India onset data are already loaded as examples).
  - Have paired country groups share out ideas and feedback across tracks.
- **Wrap-up Discussion:** Conclude the session with a focused discussion on the "key ingredients needed" for successful, use-case-driven model assessment.

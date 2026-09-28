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

**3. Area Under the ROC Curve (AUC)**

The **Area Under the ROC Curve (AUC)** measures how well a forecast can **discriminate** between events that occur and those that don't.

Think of it as answering: "If I pick a random case where the event occurred and a random case where it didn't, what's the probability that my forecast gives a higher probability to the event case?"

### Key Properties:
- **Range**: 0 to 1
- **Perfect score**: 1.0 (perfect discrimination)
- **No skill**: 0.5 (random guessing)
- **Higher is better**


### Part 2: AI Almanac Exploration and Key Ingredients
- **Objective:** Explore the AI Almanac and gather feedback on necessary components to assess metrics and use cases in an easy, interactive way.
- **Activity:** 
  - Guide participants to explore the AI Almanac (Note: Ethiopia and India onset data are already loaded as examples).
  - Have paired country groups share out ideas and feedback across tracks.
- **Wrap-up Discussion:** Conclude the session with a focused discussion on the "key ingredients needed" for successful, use-case-driven model assessment.

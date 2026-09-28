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

## Evaluating Models and Exploring the AI Almanac


### Part 1: The AI Weather Model Scorecard
- **Objective:** Discuss how to choose between models across evaluation metrics based on their specific use case.
- **Activity:** .
- **Facilitation:** Ensure the coding instructions and conceptual explanations are easy to engage with for both weather and agricultural services participants.

### Part 2: AI Almanac Exploration and Key Ingredients
- **Objective:** Explore the AI Almanac and gather feedback on necessary components to assess metrics and use cases in an easy, interactive way.
- **Activity:** 
  - Guide participants to explore the AI Almanac (Note: Ethiopia and India onset data are already loaded as examples).
  - Have paired country groups share out ideas and feedback across tracks.
- **Wrap-up Discussion:** Conclude the session with a focused discussion on the "key ingredients needed" for successful, use-case-driven model assessment.

---
title: Demo 3
teaching: 1
exercises: 0
questions:
- "How do we set up and run AI weather models locally in a container?"
- "What are the core commands needed to go from environment setup to generating a forecast figure?"
- "How do we tailor the model execution to specific use cases like onset, cessation, or temperature exceedance?"
objectives:
- "Gain hands-on experience with the end-to-end process of running AI weather models locally."
- "Execute a streamlined 4-command workflow to build a container, run a model, and generate use-case-specific outputs."
keypoints:
- "A streamlined 4-command workflow (build container, run model, get output, generate figure) simplifies local AI forecasting."
- "Use-case flags in the code allow groups to tailor outputs for onset/cessation, temperature exceedance, or precipitation exceedance."
- "Jupyter notebooks provide an interactive environment for executing and visualizing the AI weather models."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Running  AI Weather Forecast Model

## End-to-End Local Model Execution and Use-Case Application


### 1. Environment and Workflow Setup
- Utilize the Jupyter notebook environment for interactive model execution.
- Introduce the distilled 4-command workflow designed to simplify the end-to-end process:
  1. Build the container.
  2. Run the model.
  3. Get the output for discussion.
  4. Generate the use-case-specific figure.

### 2. Use-Case Group Execution
- Split participants into their designated use-case groups. 
- Ensure each group uses the specific flag in the code that distinguishes their use case:
  - **Temperature Exceedance:** Led by Docko, supported by Narayana.
  - **Precipitation Exceedance (Short-run rainfall):** Led by Koomi, supported by Shruti.
  - **Onset/Cessation:** Led Aryan, supported by Panchali.

### 3. Output Generation and Discussion Prep
- Groups execute their specific model runs within the container.
- Participants generate the final figure (Command 4) and prepare their outputs for discussion and evaluation in subsequent sessions.


**Timings on the Spark for a 10-day forecast:**

| Model | Members | Time | Peak GPU memory |
| --- | --- | --- | --- |
| AIFS v2 single | 1 | ~2 min | 14 GB |
| FGN Mini (1°) | 3 | ~2 min* | 1.8 GB |
| NeuralGCM (2.8°) | 3 | ~2.5 min | 18 GB |
| GraphCast | 1 | ~3.5 min | 16 GB |
| AIFS v2 ENS | 3 | ~5.5 min | 25 GB |
| Atlas CRPS | 3 | ~19 min | 33 GB |
| Aurora 1.5 | 1 | ~26 min | 27 GB |

* **FGN:** measured 106 s for Google’s 7.5-day sample case; the 10-day time is scaled up from that

Times include downloading the starting conditions (from Google’s ERA5 copy or ECMWF), except FGN, which starts from Google’s sample file. Model weights were already downloaded

The first run of each model is slower because the weights download once: about 31 min extra for Atlas CRPS, 7 min for Aurora 1.5 and 5 min for AIFS v2 ENS



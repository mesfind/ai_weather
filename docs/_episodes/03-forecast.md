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

# Running Your First AI Weather Forecast

## End-to-End Local Model Execution and Use-Case Application

**Phase 1 Lead:** Aryan *(Action: Check 2025 AIFS version)*  
**Support Leads:** Panchali (Onset/Cessation), Docko and Narayana (Temperature Exceedance), Koomi and Shruti (Precipitation Exceedance)  
**Audience:** All Participants (Split into Use-Case Groups)  
**Sub Team:** Sub Team 1  

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
  - **Onset/Cessation:** Led by Aryan, supported by Panchali.
  - **Temperature Exceedance:** Led by Docko, supported by Narayana.
  - **Precipitation Exceedance (Short-run rainfall):** Led by Koomi, supported by Shruti.

### 3. Output Generation and Discussion Prep
- Groups execute their specific model runs within the container.
- Participants generate the final figure (Command 4) and prepare their outputs for discussion and evaluation in subsequent sessions.
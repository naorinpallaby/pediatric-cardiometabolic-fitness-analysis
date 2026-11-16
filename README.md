# Pediatric Cardiometabolic Risk and Relative Physical Fitness Analysis

This repository holds the Python code for analyzing how relative physical fitness, specifically cardiorespiratory endurance, upper-body strength, and lower-body explosive strength, relates to clustered cardiovascular and metabolic risk (Z-Score MR) among 250 school-aged children.

## Pipeline Overview
The main script (`pediatric_cardiometabolic_fitness_analysis.py`) runs the entire data workflow from start to finish:
1. **Standardization:** Calculates age- and sex-specific Z-scores and creates size-adjusted fitness variables (`6MWT/Height`, `Jump/Height`, and `Grip/Weight`).
2. **Risk Scoring (Z-Score MR):** Combines waist circumference, WHtR, sum of skinfolds, and mean arterial pressure into a continuous non-invasive risk index.
3. **Statistical Models:**
   - **Table 1:** Baseline traits broken down by WHO weight status groups.
   - **Table 2:** Partial correlation coefficients controlling for age, sex, and zBMI).
   - **Table 3:** Two-way ANCOVA risk matrices across fitness tertiles with 95% confidence intervals.
   - **Figure 1:** Forest plots from logistic regression models predicting elevated metabolic risk (Z-Score MR) > +1.0).
   - **Figure 2:** Factorial ANCOVA plots checking for effect modification across weight categories.

## Setup and Running
Make sure you have Python 3.10+ along with these standard scientific libraries installed:
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn

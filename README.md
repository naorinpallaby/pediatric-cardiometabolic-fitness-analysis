# Pediatric Cardiometabolic Risk and Relative Physical Fitness Analysis

This repository contains the reproducible Python statistical pipeline used to investigate the protective associations of relative physical fitness domains—cardiorespiratory endurance, upper-body muscular strength, and lower-body explosive strength—against clustered cardiovascular/metabolic risk (Z-Score MR) in school-aged children (N = 250).

## Overview of the Analytical Pipeline
The script (`pediatric_cardiometabolic_fitness_analysis.py`) executes the complete statistical workflow:
1. **Data Standardization:** Computes internal age- and sex-specific Z-scores and derives body-size normalized physical fitness indices (`6MWT/Height`, `Jump/Height`, and `Grip/Weight`).
2. **Clustered Risk Score (Z-Score MR):** Constructs a continuous, non-invasive composite cardiovascular risk score from waist circumference, waist-to-height ratio (WHtR), sum of skinfolds, and mean arterial pressure (MAP).
3. **Statistical Modeling:**
   - **Table 1:** Descriptive baseline characteristics stratified by WHO BMI-for-age weight status.
   - **Table 2:** Partial Pearson correlations adjusted for age, sex, and zBMI.
   - **Table 3:** Two-way ANCOVA risk-gradient matrices across combined tertiles of endurance and muscular strength with 95% confidence intervals.
   - **Figure 1:** Multivariable logistic regression forest plots for high metabolic risk clustering (Z-Score MR > +1.0).
   - **Figure 2:** Factorial ANCOVA moderation plots evaluating effect modification by weight status.

## Requirements & Execution
To run the script locally or in Google Colab, ensure the following Python 3.10+ libraries are installed:
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn

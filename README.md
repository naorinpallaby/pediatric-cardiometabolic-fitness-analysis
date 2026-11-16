# Pediatric Cardiometabolic Risk and Physical Fitness Analysis

This repository contains the reproducible Python statistical pipeline used to investigate the independent associations of cardiorespiratory and muscular fitness with clustered cardiovascular/metabolic risk (Z-Score MR) in school-aged children (N = 250).

## Summary

* **Background:**  
  Childhood obesity increasingly drives early cardiovascular and metabolic complications, yet the independent and combined contributions of cardiorespiratory and muscular fitness to clustered metabolic risk in pediatric populations require further regional exploration. This study investigates the independent associations of cardiorespiratory endurance and muscular strength with clustered cardiovascular risk (Z-Score MR) in school-aged children.

* **Methods:**  
  This cross-sectional study evaluated 250 school-aged children stratified by WHO BMI-for-age weight status. Cardiorespiratory endurance was assessed via the six-minute walk test ($6\text{MWT}$), while muscular fitness was evaluated using standing long jump and handgrip strength. A continuous, non-invasive composite risk score (Z-Score MR) was constructed from waist circumference, waist-to-height ratio, skinfold thickness, and mean arterial pressure. Analyses utilized partial correlations (adjusted for age, sex, and zBMI), two-way ANCOVA risk matrices, and multivariable logistic regression.

* **Results:**  
  Cardiorespiratory fitness and muscular strength were each inversely associated with clustered metabolic risk, independent of one another and after adjusting for confounders ($p < 0.001$). Children in the lowest fitness tertiles faced significantly higher odds of high metabolic clustering (Z-Score MR) > +1.0, Adjusted Odds Ratio = 3.42, 95\% CI: 1.85, 6.31) compared to their most fit peers. Furthermore, two-way ANCOVA matrices demonstrated that significant differences in metabolic risk across fitness levels persisted across varying weight status categories.

* **Conclusions:**  
  Muscular and cardiorespiratory fitness are independently associated with metabolic risk in children. These results support current physical activity recommendations for youth, emphasizing that multi-component interventions incorporating both aerobic and strength-based activities are essential for mitigating early metabolic risk. Future longitudinal studies are recommended to establish causal relationships and evaluate the impact of targeted interventions.

## Pipeline Overview
The main script (`pediatric_cardiometabolic_fitness_analysis.py`) runs the entire data workflow from start to finish:
1. **Standardization:** Calculates age- and sex-specific Z-scores and creates size-adjusted fitness variables (6MWT/Height, Jump/Height, and Grip/Weight).
2. **Risk Scoring ($Z\text{-Score MR}$):** Combines waist circumference, WHtR, sum of skinfolds, and mean arterial pressure into a continuous non-invasive risk index.
3. **Statistical Models:**
   - **Table 1:** Baseline traits broken down by WHO weight status groups.
   - **Table 2:** Partial correlation coefficients controlling for age, sex, and zBMI.
   - **Table 3:** Two-way ANCOVA risk matrices across fitness tertiles with $95\%$ confidence intervals.
   - **Figure 1:** Forest plots from logistic regression models predicting elevated metabolic risk (Z-Score MR > +1.0).
   - **Figure 2:** Factorial ANCOVA plots checking for effect modification across weight categories.

## Setup and Running
Make sure you have Python 3.10+ along with these standard scientific libraries installed:
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn

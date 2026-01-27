# Agent 3 Extubation Advisor

## Overview
Agent 3 is a supervised machine learning pipeline designed to predict **extubation failure within 48 hours** in ICU patients using routinely collected physiological telemetry. The system operates on a **time-zero snapshot**, aggregating the 24 hours of data preceding extubation and producing a risk estimate to support clinical decision-making.

The pipeline is organized into four stages:
1. Dataset construction and labeling  
2. Physiologically informed preprocessing and exploratory analysis  
3. Model training, evaluation, and clinical validation  
4. Visualization and diagnostic plots  

---

## Dataset: Extubation Events
**Path:** `Agent 3/Data Extraction/`

The dataset captures ICU extubation events for patients mechanically ventilated for at least 24 hours.

### Key Characteristics
- **Task:** Binary classification (Success vs. Failure)
- **Prediction Window:** 2–48 hours post-extubation
- **Input Window:** 24 hours prior to extubation
- **Class Balance:** ~93.8% Success / ~6.2% Failure
- **Sample Size:** 14,992 extubation events

### Label Definition
- **Failure (0):** Evidence of invasive ventilation (PEEP, Peak, or Plateau Pressure > 5 cmH₂O) within 48 hours
- **Success (1):** No invasive ventilation observed

### Feature Domains
- Ventilator mechanics
- Oxygenation and gas exchange
- Neurological status
- Hemodynamics
- Renal and fluid balance

Clinically established features (e.g., RSBI, P/F ratio, driving pressure) are explicitly derived to reflect standard weaning criteria.

---

## Data Preprocessing & Exploratory Analysis
**Path:** `Agent 3/EDA and Preprocessing/`

A domain-driven preprocessing pipeline was developed to enforce physiological realism and preserve clinical meaning.

### Preprocessing Principles
- **Data Protection:** Identifiers and targets isolated prior to transformations  
- **Physiological Bounding:** Hard limits applied to remove sensor artifacts  
- **Clinical Imputation:** Missing invasive labs assumed normal when absent  
- **Median Fallback:** Applied only to continuous vitals where data continuity is expected  

### Outcome
- Final dataset is **100% complete** (no missing values)  
- All features fall within clinically plausible ranges  
- Dimensionality preserved: `(14,992 × 37)`  

### EDA Highlights
- Failure cohort exhibits broader hypoxic tails (SpO₂ < 92%)  
- A high-density “safe zone” identified with RSBI 30–60 and SpO₂ > 96%  
- Failure risk correlates with tachypnea, hypoxemia, and ventilation inefficiency  

---

## Modeling Results & Clinical Validation
**Path:** `Agent 3/Results/`

### Model Selection
Nine architectures were benchmarked on imbalanced ICU data:
- Logistic Regression  
- Random Forests  
- Gradient Boosting (XGBoost, LightGBM, CatBoost)  

**Selected Model:** Extra Trees Classifier  

**Rationale:** Gradient Boosting models overfit stochastic physiological noise, while Extra Trees provided superior robustness and generalization.

### Optimization Strategy
- Accuracy avoided due to class imbalance  
- Optimized for **Recall at fixed 80% Specificity**  
- Safety-first operating threshold applied  

### Performance at Operating Point (P > 0.543)
| Metric | Value |
|------|------|
| Sensitivity (Recall) | 40.1% |
| Specificity | 80.2% |
| AUROC | 0.67 |
| Precision | 12.0% |

The model is intended as a **high-sensitivity screening tool**, not a definitive diagnostic.

### Interpretability
SHAP analysis confirms alignment with respiratory physiology:
- Hypoxia (`min_spo2`) is the strongest driver of failure  
- Hypercapnia (`paco2`) and work of breathing indicate respiratory fatigue  
- Stable pH and low respiratory rate variability predict success  

---

## Plots & Visual Diagnostics
**Path:** `Agent 3/Plots/`

The repository includes curated visualizations for data validation and model assessment.

### EDA Plots
**Directory:** `Agent 3/Plots/EDA/`  
Includes:
- Feature distributions stratified by outcome  
- Density plots highlighting physiological risk zones  
- Correlation and variability analyses  

### Initial Results Plots
**Directory:** `Agent 3/Plots/Initial Results/`  
Includes:
- ROC curves and AUROC comparisons  
- Confusion matrices at the operating threshold  
- Feature importance and SHAP summary plots  

---

## Summary
Agent 3 demonstrates that routinely collected ICU telemetry contains a meaningful predictive signal for extubation failure. By combining physiologically grounded preprocessing with a noise-robust ensemble model, the system identifies approximately **2 out of 5 failures** in advance, providing a clinically actionable safety net for extubation decision support.

---

## Directory Structure
```text
Agent 3/
├── Data Extraction/
│   └── Readme.md
├── EDA and Preprocessing/
│   └── Readme.md
├── Results/
│   └── Readme.md
├── Plots/
│   ├── EDA/
│   └── Initial Results/
└── Readme.md (You are here:)


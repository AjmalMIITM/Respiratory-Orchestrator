# Data Quality Assessment & Preprocessing Strategy

This section documents the key data quality issues identified through exploratory statistics and outlines the corrective actions taken prior to model training. Given the clinical setting and high-noise ICU telemetry data, addressing these issues was necessary to ensure physiologically valid inputs and reliable downstream model behavior.

---

## Identified Issues

### 1. Mixed Unit Scaling (SpO₂ Inconsistency)

**Evidence**  
The `min_spo2` feature exhibits inconsistent scaling:
- Mean value: **35.6**
- 25th percentile: **0.95** (ratio scale)
- 75th percentile: **90.0** (percentage scale)

**Issue**  
This column contains values recorded in two different units:
- Ratios in the range **0.0–1.0**
- Percentages in the range **0–100**

Without correction, the model interprets a healthy SpO₂ value such as `0.98` as substantially worse than a hypoxic reading like `80`, leading to incorrect clinical inference.

---

### 2. Physiologically Impossible Outliers

**Evidence**  
The `last_peep` feature shows extreme values:
- Maximum: **570**
- Minimum: **-5.4**

**Issue**  
Positive End-Expiratory Pressure (PEEP) typically ranges from **5–20 cmH₂O** in clinical practice. Negative values and extreme maxima represent sensor artifacts or data ingestion errors. These outliers distort feature scaling and bias tree-based model splits.

---

### 3. Invalid Zero Values Representing Missing Data

**Evidence**  
Several features have minimum values of **0.0**, including:
- `min_spo2`
- `median_rr`
- `driving_pressure`

**Issue**  
A living patient cannot have a respiratory rate or oxygen saturation of zero. These values function as *implicit missing data* rather than true measurements. If left uncorrected, the model may incorrectly learn that “zero” represents extreme physiological collapse.

---

### 4. Severe Class Imbalance

**Evidence**  
The target variable `target_failure` has a mean of **0.062**, indicating a **6.2% failure rate**.

**Issue**  
The dataset is dominated by non-failure cases (~94%). Without mitigation, a naïve model can achieve high accuracy by predicting “no failure” for all patients, while completely missing clinically critical events.

---

## Preprocessing Plan

The following steps were applied to address these issues prior to model development.

### Step 1: Standardize SpO₂ Units

**Action**  
Apply conditional scaling:
- Values **≤ 1.0** are multiplied by 100
- Values **> 1.0** are left unchanged

**Outcome**  
All SpO₂ measurements are standardized to a **0–100 percentage scale**, ensuring consistent physiological interpretation.

---

### Step 2: Outlier Clipping (Physiological Sanity Bounds)

**Action**
- Clip `last_peep` to the range **0–40 cmH₂O**
- Clip `last_ppeak` to a maximum of **100 cmH₂O**

**Outcome**  
Prevents rare sensor artifacts from dominating feature normalization or influencing model structure.

---

### Step 3: Correction of False Zero Values

**Action**
- Convert `0.0` values to `NaN` for:
  - `min_spo2`
  - `median_rr`
  - `driving_pressure`
- Impute missing values using the **median** of each feature

**Outcome**  
Distinguishes true physiological signals from missing or disconnected sensors, improving model interpretability and stability.

---

### Step 4: Class Imbalance Handling

**Action**  
Configure the downstream classifier with:
```python
class_weight = "balanced"

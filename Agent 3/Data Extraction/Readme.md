# Agent 3 (Extubation Advisor)

**Source:** Extracted ICU Telemetry Features  
**Task:** Binary Classification — Extubation Failure Prediction

---

## Overview

This dataset captures extubation events from the intensive care unit (ICU) and is designed for a supervised machine learning task: predicting whether a patient will experience extubation failure (requiring re-intubation) within 48 hours.

Physiological time-series data are aggregated over a **24-hour rolling window** preceding the extubation decision.

---

## Cohort Definition and Labeling

### 1. Cohort Definition

Extubation events were identified based on the following criteria:

- **Ventilation Duration:** Patients were mechanically ventilated for a clinically meaningful duration, representing true weaning candidates.
- **Event Definition:** Extubation was defined as cessation of invasive mechanical ventilation.
- **Exclusions:** Short-term disconnections and transient measurement gaps were excluded to avoid false events.

---

### 2. Labeling Logic

The target variable is `success_48h` (mapped to `target_failure` during model training).

- **Evaluation Window:** `[t_extubation + 2h, t_extubation + 48h]`
- **Failure (0):** Presence of invasive ventilation measurements (PEEP > 5 cmH₂O) within the evaluation window, indicating re-intubation.
- **Success (1):** No invasive ventilation measurements detected.
- **Class Distribution:** Approximately 93.8% Success / 6.2% Failure.

| Scenario | Outcome Description | Label (`success_48h`) |
|--------|---------------------|----------------------|
| Patient A | Extubated → Transferred to floor → No re-intubation | 1 (Success) |
| Patient B | Extubated → Clinical deterioration → Re-intubated at hour 20 | 0 (Failure) |
| Patient C | Extubated → No subsequent measurements (discharged or deceased) | 1 (Success) |

---

## Feature Engineering

All features are derived from the **24 hours prior to extubation**.

### Final Model Feature Set (Core 7)

Based on feature importance analysis, the following seven features drive the final Agent 3 model:

| Feature | Aggregation | Description | Mean |
|-------|-------------|-------------|------|
| `driving_pressure` | Calculated | Plateau Pressure − PEEP (lung stiffness) | ~8.9 cmH₂O |
| `last_ppeak` | Last | Peak inspiratory pressure | ~15.0 cmH₂O |
| `median_rr` | Median | Respiratory rate trend (24h) | ~19.5 breaths/min |
| `min_spo2` | Minimum | Lowest oxygen saturation | Mixed scale |
| `last_fio2` | Last | Fraction of inspired oxygen | ~0.41 |
| `last_peep` | Last | Positive end-expiratory pressure | ~6.5 cmH₂O |
| `last_ps` | Last | Pressure support level | ~7.2 cmH₂O |

---

### Additional Available Features

The full feature extraction pipeline included additional variables across multiple domains:

- **Calculated:** `rsbi_last` (Rapid Shallow Breathing Index), `pf_ratio_last`
- **Laboratory:** `last_ph`, `last_lactate`, `last_paco2`, `total_urine_24h`
- **Neurological:** `median_rass` (sedation score)

---

## Data Processing Notes

1. **Class Imbalance**  
   The dataset exhibits significant imbalance (~15:1 success-to-failure ratio). Model training required explicit class weighting to maintain sensitivity to failure events.

2. **Missing vs. Structural Zeros**  
   Certain variables (e.g., `last_ps`) contain zero values representing specific ventilation modes (such as T-piece trials) rather than missing data. These were preserved where clinically appropriate.

3. **Unit Consistency**  
   Oxygen saturation (`SpO₂`) values were recorded using mixed units (0–1 ratios and 0–100 percentages). Values were standardized during preprocessing to ensure consistent interpretation.

---

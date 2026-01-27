# Dataset: Agent 3 (Extubation Advisor)

**File Name:** `amsterdam_stage3_advisor_matrix_v6.csv`

---

##  Overview
This dataset captures **Extubation Events** from the ICU. It is designed for a Supervised Machine Learning task: predicting whether a patient will successfully breathe on their own or fail (require reintubation) within 48 hours.

The dataset aggregates physiological time-series data into a **24-hour rolling window** prior to the extubation decision.

---

## 🛠️ Data Construction Logic

### 1. Cohort Definition (The "Who")
We identified extubation events based on the following criteria:
*   **Ventilation Duration:** Patient must have been mechanically ventilated for **≥ 24 hours**.
*   **Event Detection:** An extubation is defined as the end of a continuous ventilation sequence where the subsequent gap in ventilation data is **≥ 24 hours** (or the data ends, implying discharge).
*   **Exclusions:** Short-term ventilation (<24h) and transient disconnects (<24h gap) were filtered out.

### 2. Labeling Logic (The "Target")
The target variable is `success_48h`.
*   **Window:** `[t_extubation + 2h, t_extubation + 48h]`
*   **Failure (0):** Presence of invasive ventilation measurements (PEEP, Peak Pressure, or Plateau Pressure > 5 cmH2O) within the window. This indicates **Reintubation**.
*   **Success (1):** No invasive ventilation measurements found in the window.
*   **Class Balance:** ~93.8% Success / ~6.2% Failure.

### 3. Feature Engineering (The "Inputs")
Features are aggregated from the **24 hours prior** to extubation (`t-24h` to `t0`).

| Feature Category | Aggregations Used | Key Concepts (OMOP/Source) |
| :--- | :--- | :--- |
| **Ventilator Mechanics** | Max, Last, Mean | PEEP, Peak Pressure, Pressure Support (PSV), FiO2, Tidal Volume (Expired), Respiratory Rate |
| **Oxygenation** | Min, Mean, Last, StdDev | SpO2, PaO2, P/F Ratio (Derived) |
| **Gas Exchange (ABG)** | Min, Last | pH, PaCO2, Bicarbonate, Lactate |
| **Neurological** | Min, Median, Last | RASS Score (Sedation), GCS Total |
| **Hemodynamics** | Max | Heart Rate (EKG/PulseOx) |
| **Renal/Fluids** | Sum, Last | Total Urine Output (24h), Creatinine, BUN |

---

##  Data Dictionary

### Identifiers
*   `visit_occurrence_id`: Unique admission identifier.
*   `extubation_time`: Timestamp of the extubation event (`t0`).

### Calculated "SOTA" Features
*   `rsbi_last`: Rapid Shallow Breathing Index (RR / TV_Liters) at `t0`.
*   `rsbi_median`: Median RSBI over the last 24h.
*   `pf_ratio_last`: PaO2 / FiO2 ratio at `t0`.
*   `driving_pressure`: Plateau Pressure - PEEP (Lung Compliance proxy).

### Ventilator Features
*   `last_ps`, `max_ps`: Pressure Support (cmH2O). **Crucial for weaning assessment.**
*   `last_peep`, `median_peep`: PEEP settings.
*   `last_tv_liters`, `median_tv_liters`: Tidal Volume (Normalized to Liters).
*   `last_rr`, `median_rr`, `std_rr`: Respiratory Rate and variability.
*   `last_fio2`, `max_fio2`: Fraction of Inspired Oxygen (0.21 - 1.0).

### Physiological Features
*   `median_spo2`, `min_spo2`, `std_spo2`: Oxygen Saturation (%).
*   `last_ph`, `min_ph`: Arterial pH (Acidosis check).
*   `last_paco2`: Arterial CO2 (Ventilation efficiency).
*   `last_lactate`: Serum Lactate (Tissue perfusion).
*   `median_rass`, `last_rass`: Richmond Agitation-Sedation Scale (-5 to +4).
*   `total_urine_24h`: Cumulative Urine Output (mL).

### Target
*   `success_48h`: **1** = Success, **0** = Failure (Reintubation).
  
| Scenario  | Did they find PEEP > 5 in the next 48h?           | failure_signals Count | Label (success_48h) |
| --------- | ------------------------------------------------- | --------------------- | ------------------- |
| Patient A | No. They went to the floor and stayed safe.       | 0                     | 1 (Success)         |
| Patient B | Yes. They crashed and got reintubated at hour 20. | 24 (hourly checks)    | 0 (Failure)         |
| Patient C | No measurements found (Discharged/died off-vent). | 0                     | 1 (Success)         |
---

## Notes for Modeling
1.  **Missing Data:** Labs (Lactate, pH, BUN) may have high missingness (~40-60%) as stable patients do not get frequent blood draws.
2.  **Imbalance:** The dataset is imbalanced (15:1). Use `scale_pos_weight` in XGBoost/LightGBM or stratified sampling.
3.  **Tidal Volume:** All Tidal Volumes have been normalized to **Liters** (e.g., 0.500) to ensure the RSBI calculation (`RR / TV`) is mathematically correct.

---

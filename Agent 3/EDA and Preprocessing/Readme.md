# Data Preprocessing Pipeline


## Overview
The pipeline enforces physiological constraints, standardizes units, and manages missing data through domain-specific clinical assumptions rather than generic statistical imputation.

## Preprocessing Methodology

The transformation logic follows a five-phase execution order:

### 1. Safe Split (Data Protection)
Before mathematical operations begin, specific metadata columns are isolated to prevent data corruption or leakage.
*   **Excluded Columns:** `visit_occurrence_id`, `extubation_time`, `success_48h`.
*   **Objective:** Protect patient identifiers and target variables from accidental scaling or imputation during the cleaning process.

### 2. Physics Engine (Artifact Removal)
Hard physiological boundaries are applied to eliminate sensor artifacts and standardize unit scales.

| Feature | Issue Addressed | Logic Applied |
| :--- | :--- | :--- |
| **SpO2** | Mixed Scaling (0.98 vs 98.0) | Values less than or equal to 1.0 are multiplied by 100. The entire column is clipped to the range [50, 100]. |
| **RSBI** | Unit Confusion | Clipped to [0, 300] to remove outliers caused by incorrect tidal volume units. |
| **Pressure Support** | Sensor Error Codes | Clipped to [0, 40] cmH2O. |
| **Driving Pressure** | Physiological Outliers | Clipped to [0, 40] cmH2O. |
| **Urine Output** | Negative Balance Calculations | Clipped to [0, 10,000] mL (24h total). |
| **Heart Rate** | Machine Noise | Clipped to [30, 250] BPM. |

### 3. Clinical Normal Imputation
For invasive laboratory tests and neurological scores, a "Missingness Implies Stability" logic is utilized. Missing values are filled with clinically normal baselines, assuming that the absence of a test indicates the clinician deemed the patient stable.

| Feature | Imputed Value | Clinical Rationale |
| :--- | :--- | :--- |
| **pH (Last/Min)** | 7.40 | Perfect physiological balance. |
| **PaCO2 (Last/Max)** | 40.0 mmHg | Normal alveolar ventilation. |
| **PaO2 (Last/Min)** | 90.0 mmHg | Normal arterial oxygenation. |
| **Lactate (Last/Max)** | 11.0 mg/dL | Normal non-septic baseline (~1.2 mmol/L). |
| **Bicarbonate** | 24.0 mEq/L | Normal metabolic buffer. |
| **RASS (Median/Last)** | 0.0 | Alert and Calm state. |
| **Creatinine** | 0.9 mg/dL | Normal kidney filtration. |
| **BUN** | 15.0 mg/dL | Normal urea nitrogen levels. |
| **Hemoglobin** | 12.0 g/dL | Normal oxygen-carrying capacity. |

### 4. Median Fallback
For continuous vital signs (Heart Rate, Respiratory Rate, SpO2) where data is expected to be continuous, missing values are treated as sensor loss.
*   **Method:** Missing entries in these remaining columns are filled using the **cohort median**.
*   **Rationale:** Preserves the central tendency of the distribution for common vitals.

### 5. Recombination
The cleaned feature set is concatenated with the originally isolated metadata columns to produce the final dataset.

## Execution Summary
*   **Process:** The script loads the raw V6 matrix, executes the cleaning function, and exports the result.
*   **Original Dimensions:** (14,992, 37)
*   **Final Dimensions:** (14,992, 37)
*   **Missing Values:** 0 (100% Complete)

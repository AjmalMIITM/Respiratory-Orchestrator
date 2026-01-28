# Agent 3: Extubation Advisor
**A Clinical AI Pipeline for Predicting Post-Extubation Failure**

##  Overview
**Agent 3** is a supervised machine learning system designed to predict extubation failure (re-intubation requirement) within a **48-hour window** for ICU patients. By synthesizing 24 hours of physiological telemetry into a risk score, Agent 3 acts as a high-sensitivity screening tool to support clinical weaning decisions.

### Key Technical Achievements
* **Architecture:** Soft Voting Ensemble (Extra Trees + Logistic Regression).
* **Performance:** Achieved **43.9% Recall** on highly imbalanced data (6.2% prevalence) where standard Boosting models failed (0% recall).
* **Validation:** Passed physiological "Stress Tests," successfully identifying the **Amato Threshold** for lung stiffness and **P/F Ratio dynamics**.

---

##  Repository Structure
* [**Data Extraction**](./Agent%203/Data%20Extraction/): Cohort definition, labeling logic, and 24-hour rolling window aggregation.
* [**EDA & Preprocessing**](./Agent%203/EDA%20and%20Preprocessing/): Unit standardization (SpO₂), outlier clipping, and leakage-free imputation.
* [**Results & Validation**](./Agent%203/Results/): Model benchmarking, Ensemble optimization, and clinical stress test reports.
* [**Plots**](./Agent%203/Plots/): Visual diagnostics, SHAP interpretability, and feature distribution analysis.

---

##  The "Golden 7" Features
Rather than using high-dimensional noise, Agent 3 utilizes seven core features that represent the "work of breathing" and lung mechanics:

| Feature | Aggregation | Clinical Significance |
| :--- | :--- | :--- |
| **Driving Pressure** | Calculated | Plateau − PEEP (Indicates lung stiffness/compliance) |
| **Last Peak Pressure** | Last | Maximum inspiratory effort and airway resistance |
| **Median Resp Rate** | Median | 24-hour respiratory frequency trend |
| **Min SpO₂** | Minimum | Captures the "Hypoxic Tail" (lowest oxygen events) |
| **Last FiO₂** | Last | Degree of oxygen dependency |
| **Last PEEP** | Last | Positive End-Expiratory Pressure requirements |
| **Last Pressure Support**| Last | Level of mechanical assistance during weaning |

---

## Data Pipeline & Preprocessing
To ensure biological plausibility and model integrity, the pipeline includes:

1.  **Leakage Prevention:** All imputation statistics (Median) are derived strictly from the Training Set and applied to the Test Set.
2.  **SpO₂ Standardization:** Unified mixed-unit records (0.0–1.0 vs 0–100%), shifting the mean from a corrupted 35.6% to a physiologically valid 90.8%.
3.  **Physiological Clipping:** Hard bounds applied to remove sensor artifacts (PEEP capped at 40, FiO₂ floored at 0.21).
4.  **Signal Discovery:** Identified that **Driving Pressure** (lung mechanics) is a significantly stronger discriminator for failure than oxygenation metrics alone in this cohort.

---

##  Modeling & Clinical Validation

### The Ensemble Strategy
Standard models like XGBoost and AdaBoost achieved **0% Recall** due to the extreme class imbalance (93.8% Success). To solve this, we implemented a **Soft Voting Ensemble**:
* **Components:** Weighted Logistic Regression + Extra Trees Classifier.
* **Rationale:** Combines the high linear sensitivity of Logistic Regression with the non-linear interaction capturing of Extra Trees.

### Quantitative Performance
| Metric | Value | Clinical Interpretation |
| :--- | :--- | :--- |
| **Failure Recall** | **43.9%** | Correctly identifies ~2 out of 5 failures in advance |
| **Safe NPV** | **91.2%** | High reliability for "Safe to Extubate" predictions |
| **AUROC** | **0.63** | Stronger discrimination than standard clinical baselines |



### Clinical Stress Testing (Proof of Medicine)
To verify the model learned medical logic rather than statistical noise, we subjected it to "synthetic" patient stress tests:
* **Lung Mechanics Test:** Risk scores spike exponentially once Driving Pressure exceeds **15 cmH₂O**, aligning with the **Amato Threshold**.
* **Oxygen Masking Test:** The model correctly flags "High Risk" for patients on 100% FiO₂ even if their SpO₂ is normal (95%), proving it understands **P/F Ratio dynamics**.
* **Deterioration Trajectory:** Risk scores escalate continuously as Respiratory Rate rises and lungs stiffen, providing a graded "Early Warning" signal.

---

##  Conclusion
Agent 3 demonstrates that routinely collected ICU telemetry contains a meaningful predictive signal for extubation failure. By prioritizing **Recall** and **Physiological Grounding**, the system provides a clinically actionable safety net for decision support.

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
└── Readme.md (You are here:)


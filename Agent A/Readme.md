# ICU Mortality Optimization (Amsterdam Cohort)

[![Project Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/AjmalMIITM/email-verification)
[![Research: SOTA](https://img.shields.io/badge/Research-SOTA%20Clinical-blue)](https://pubmed.ncbi.nlm.nih.gov/)
[![License: Research](https://img.shields.io/badge/License-Research-orange)](https://vcl.amsterdamumc.org/)

This repository implements a **State-of-the-Art (SOTA)** Deep Phenotyping pipeline for ICU mortality prediction on the AmsterdamUMCdb dataset. By integrating representation learning with density-based clustering and ensemble stacking, we match the published AUC benchmark (0.786) while achieving **2x higher precision (AUC-PR 0.42 vs 0.19)**.

---

##  Key Results Table

| Metric | Thoral et al. [P21] | **Our Model (Ensemble)** | Status |
| :--- | :---: | :---: | :---: |
| **AUC-ROC** | 0.78 | **0.7861** |  Matched |
| **AUC-PR** | 0.19 | **0.4193** |  +120% |
| **Brier Score**| 0.12 | **0.0845** |  Safer |

---

##  Pipeline Architecture

###  Phase 1: Exploratory Data Analysis & Feature Pivot
- **Objective:** Mitigate the "Curse of Missingness" (Source P11).
- **Transformation:** Dropped variables with >70% missingness; pivoted to high-coverage proxies (`last_tidal_volume`, `avg_heart_rate`, `approximate_age`).
- **Validation:** Spearman correlation audit (|r| < 0.7) and 0.02/0.98 percentile clipping (Source P1).

###  Phase 2: Deep Representation Learning (UMAP)
- **Objective:** Preserve multi-dimensional physiological variance.
- **Methods:** Multiple Imputation by Chained Equations (MICE) followed by UMAP (8 components for discovery).
- **Visual Evidence:** Manifold verified biological continuity without imputation artifacts.
![UMAP Latent Space](https://github.com/AjmalMIITM/Respiratory-Orchestrator/blob/main/Agent%20A/Results/Plots/umap_representation_learning.png)

###  Phase 3: HDBSCAN Phenotype Discovery
- **Objective:** Isolate latent clinical sub-states.
- **Results:** Discovered **31 distinct phenotypes**. Identified Phenotype 12 (High Respiratory/Metabolic Failure) with **20% mortality**.
- **Methods:** Soft-membership probabilities generated for 100% of the cohort.
![Phenotype Risk](https://github.com/AjmalMIITM/Respiratory-Orchestrator/blob/main/Agent%20A/Results/Plots/hdbscan_phenotypes.png)

###  Phase 4: Championship Ablation Matrix
- **Objective:** Quantify the "Value of Information" of phenotyping.
- **Protocol:** 20-run grid (5 Models x 4 Feature Tiers).
- **Winner:** LightGBM on `Raw + Hard ID` (AUC 0.786).

###  Phase 5: Diversity Ensemble Stacking
- **Objective:** Break the discrimination ceiling.
- **Logic:** Weighted Soft Voting combining **Decision Trees** (LGBM/XGB) with **Geometric Experts** (Scaled/Calibrated LinearSVC).
- **Calibration:** Achieved high clinical utility across risk thresholds via Isotonic scaling.
![Calibration Audit](https://github.com/AjmalMIITM/Respiratory-Orchestrator/blob/main/Agent%20A/Results/Plots/phenotype_risk_calibration.png)
---

##  Repository Structure
```text
Amsterdam_SOTA_Optimization/
├── src/            # Core algorithms (EDA, UMAP, HDBSCAN, Ensemble)
├── scripts/        # Safe execution wrappers for Windows
├── docs/           # In-depth Research & Technical Documentation
│   ├── TECHNICAL_PROJECT_WHITEPAPER.md  <-- Core Research Synthesis
│   ├── 01_EDA_DOCUMENTATION.md
│   ├── 02_UMAP_DOCUMENTATION.md
│   ├── 03_HDBSCAN_DOCUMENTATION.md
│   └── 05_ENSEMBLE_DOCUMENTATION.md
├── results/        # Visualization assets (Plots, Heatmaps)
├── logs/           # Execution logs & console captures
├── artifacts/      # Model embeddings (GitIgnored)
└── requirements.txt # Dependencies
```

---
##  One-Click Hackathon Demo

Want to see the model in action without running the full pipeline? Use our pre-packaged **Inference Engine**:

1. **Navigate to the Demo Folder:**
   
   a)
   ```bash
   git clone https://github.com/AjmalMIITM/Respiratory-Orchestrator
   ```
   b)
   ```bash
   cd Respiratory-Orchestrator
   cd "Agent A\demo"
   ```
3. **Run the Prediction Script:**
   ```bash
   python run_demo.py
   ```
4. **What Happens?**
   - The script loads the serialized SOTA "Brain" (`demo/models/sota_ensemble_pkg.pkl`).
   - It processes the sample patients in `demo/data/dummy.csv`.
   - It outputs a **Risk Score** and **Clinical Persona** (e.g., "Respiratory Failure").

*This demonstrates the "Invisible GPS" strategy: users provide raw vitals, and the model automatically maps them to the correct Phenotype in the background.*
---
Or 
---

##  Replication Workflow (From Scratch)

To fully replicate the SOTA results, execute the pipeline phases in this exact order:

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure raw data is correctly positioned or update `src/01_EDA.py` to point to your dataset.*

2. **Phase 1: Exploratory Data Analysis**
   ```bash
   python scripts/run_eda_safe.py
   ```
   *Outcome: Validates data health and generates `artifacts/X_processed_final.csv`.*

3. **Phase 2: Representation Learning (UMAP)**
   ```bash
   python scripts/run_umap_safe.py
   ```
   *Outcome: Generates 8D latent embeddings in `artifacts/umap_embeddings.npy`.*

4. **Phase 3: Phenotype Discovery (HDBSCAN)**
   ```bash
   python scripts/run_hdbscan_safe.py
   ```
   *Outcome: Identifies 31 phenotypes and assigns soft probabilities.*

5. **Phase 4: Modeling Championship**
   ```bash
   python scripts/run_championship_safe.py
   ```
   *Outcome: Benchmarks 20 models; identifies LightGBM as the single-model winner.*

6. **Phase 5: Ensemble Stacking (Final SOTA)**
   ```bash
   python scripts/run_ensemble_safe.py
   ```
   *Outcome: Generates final ensemble predictions and calibration audit.*

---

##  Research Citations
- **Source P1 [Zhang et al.]:** MICE and Outlier Clipping protocols.
- **Source P2 [Li et al.]:** Phenotype stability and mortality stratification.
- **Source P21 [Thoral et al.]:** AmsterdamUMCdb SOTA Benchmarking (0.78 AUC).

---




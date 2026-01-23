# 🏥 Live Hackathon Demo Instructions

**Goal:** Demonstrate the "Invisible GPS" Phenotyping Engine in under 2 minutes.

## 1. Setup
Open your terminal to this `demo/` folder:
```bash
cd demo
```

## 2. The Narrative
*"Instead of a black box that just says 'High Risk', our model automatically maps the patient to one of 31 discovered phenotypes."*

## 3. The Action
Run the prediction engine on our 2 dummy patients (Respiratory Failure vs. Healthy):
```bash
python run_demo.py
```

## 4. The Result
You will see:
- **Patient 0:** "Respiratory Failure (High Risk)" -> **CRITICAL**
- **Patient 1:** "Standard Recovery (Low Risk)" -> **STABLE**

*Note: The `sota_ensemble_pkg.pkl` contains the entire heavy lifting (Imputation, UMAP, Ensemble).*


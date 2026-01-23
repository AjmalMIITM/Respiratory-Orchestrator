"""
SOTA Inference Engine (The "Easy Button")
Usage: python run_demo.py
Result: outputs risk scores and phenotype assignments for the dummy patients.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

def predict_demo_patients():
    print("="*80)
    print("🏥 SOTA ICU PREDICTOR: LOADING BRAIN...")
    print("="*80)
    
    # 1. Load the Brain (Structured Path)
    pkg_path = os.path.join("models", "sota_ensemble_pkg.pkl")
    input_csv = os.path.join("data", "dummy.csv")
    
    if not os.path.exists(pkg_path):
        print(f"[ERROR] Model package not found at {pkg_path}.")
        return
        
    pkg = joblib.load(pkg_path)
    print("[OK] Model Package Loaded.")
    
    # 2. Load Data
    if not os.path.exists(input_csv):
        print(f"[ERROR] Input file {input_csv} not found.")
        return
    else:
        data = pd.read_csv(input_csv)
        # Ensure columns match
        missing_cols = [c for c in pkg['features'] if c not in data.columns]
        if missing_cols:
            print(f"[ERROR] Input CSV missing columns: {missing_cols}")
            return
        data = data[pkg['features']]

    # 3. Micro-Pipeline (The "App")
    print("\n[PROCESSING] Running Transformations...")
    
    # A. Impute
    X_imputed = pkg['imputer'].transform(data)
    
    # B. UMAP "GPS" Location
    X_umap = pkg['umap_reducer'].transform(X_imputed)
    
    # C. Phenotype Assignment (Centroids strategy)
    centroids = pkg['umap_centroids'].values
    centroid_labels = pkg['umap_centroids'].index.values
    
    assigned_phenos = []
    soft_probs = [] 
    
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X_umap, centroids)
    
    for i in range(len(data)):
        # Hard Assignment
        closest_idx = np.argmin(dists[i])
        assigned_phenos.append(centroid_labels[closest_idx])
        
        # Soft Assignment
        weights = 1 / (1 + dists[i])
        probs = weights / np.sum(weights)
        
        p_dict = {f'prob_pheno_{label}': probs[idx] for idx, label in enumerate(centroid_labels)}
        soft_probs.append(p_dict)
        
    df_phenos = pd.DataFrame(soft_probs)
    df_phenos = df_phenos.reindex(sorted(df_phenos.columns), axis=1) 
    
    # D. Prepare Model Inputs
    X_hard = pd.DataFrame(X_imputed, columns=pkg['features'])
    X_hard['phenotype_id'] = assigned_phenos
    
    X_soft = pd.concat([pd.DataFrame(X_imputed, columns=pkg['features']), df_phenos], axis=1)
    
    # 4. Predict
    print("\n[PREDICTING] Running Ensemble Vote...")
    p1 = pkg['lgbm_model'].predict_proba(X_hard)[:, 1]
    p2 = pkg['xgb_model'].predict_proba(X_hard)[:, 1]
    p3 = pkg['svc_model'].predict_proba(X_soft)[:, 1]
    
    w = pkg['weights']
    p_final = (w[0]*p1 + w[1]*p2 + w[2]*p3) / sum(w)
    
    # 5. Report
    print("\n" + "="*80)
    print(f"{'Patient':<10} | {'Phenotype':<20} | {'Risk Score':<12} | {'Outlook'}")
    print("-" * 80)
    
    clinical_map = {
        12: "Respiratory Failure (High Risk)",
        13: "Sepsis/Renal (High Risk)",
        27: "Frailty/Cardiac (High Risk)",
        1:  "Standard Recovery (Low Risk)"
    }
    
    for i in range(len(p_final)):
        pid = assigned_phenos[i]
        desc = clinical_map.get(pid, f"Phenotype {pid}")
        score = p_final[i]
        status = "🔴 CRITICAL" if score > 0.5 else "🟢 STABLE"
        
        print(f"{i:<10} | {desc:<20} | {score:>10.1%} | {status}")
        
    print("="*80)

if __name__ == "__main__":
    predict_demo_patients()

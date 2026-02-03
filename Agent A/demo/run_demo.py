"""
SOTA Inference Engine (The "Easy Button")
Usage: python run_inference.py --input dummy.csv
Result: outputs risk scores and phenotype assignments.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import sys

def predict_new_patients(input_csv):
    print("="*80)
    print("SOTA ICU PREDICTOR: LOADING BRAIN...")
    print("="*80)
    
    # 1. Load the Brain
    # UPDATED PATH: Points to 'Agent A/demo/models/sota_ensemble_pkg.pkl'
    # relative to the script location in 'Agent A/demo/'
    pkg_path = os.path.join("models", "sota_ensemble_pkg.pkl")
    
    if not os.path.exists(pkg_path):
        print(f"[ERROR] Model package not found at {pkg_path}.")
        print("        Ensure 'sota_ensemble_pkg.pkl' is in the 'models' folder.")
        return
        
    pkg = joblib.load(pkg_path)
    print("[OK] Model Package Loaded.")
    
    # 2. Load New Data
    if not os.path.exists(input_csv):
        # Create Dummy Data for Demo if file doesn't exist
        print(f"[WARN] Input file '{input_csv}' not found. Generating DEMO patients...")
        data = pd.DataFrame([
            # Patient A: High Risk (Resp Failure Profile)
            [32, 340, 12, 160, 115, 4.2, 110, 68, 1, 0.6, 28],
            # Patient B: Low Risk (Stable)
            [18, 480, 5, 12, 72, 1.1, 85, 42, 0, 0.3, 14]
        ], columns=pkg['features'])
        print("[INFO] Generated 2 synthetic patients for demonstration.")
    else:
        data = pd.read_csv(input_csv)
        # Ensure columns match
        missing_cols = [c for c in pkg['features'] if c not in data.columns]
        if missing_cols:
            print(f"[ERROR] Input CSV missing columns: {missing_cols}")
            return
        data = data[pkg['features']]

    # 3. Micro-Pipeline
    print("\n[PROCESSING] Running Transformations...")
    
    # Ensure float
    data = data.astype(float)
    
    # A. Preprocessing Pipeline
    # 1. Log Transform
    skewed_feats = ['avg_lactate', 'avg_crp', 'first_creatinine']
    for col in skewed_feats:
        if col in data.columns:
            data.loc[:, col] = np.log1p(data[col])
            
    # 2. Scale
    if 'scaler' in pkg:
        X_scaled_array = pkg['scaler'].transform(data)
        X_scaled = pd.DataFrame(X_scaled_array, columns=data.columns)
    else:
        print("[WARNING] Scaler missing from package! Using raw data.")
        X_scaled = data
        
    # 3. Impute
    X_imputed = pkg['imputer'].transform(X_scaled)
    
    # B. UMAP "GPS" Location
    X_umap = pkg['umap_reducer'].transform(X_imputed)
    
    # C. Phenotype Assignment (Distance to Centroids)
    # Find closest centroid in UMAP space
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
        
        # Soft Assignment (1 / (1+d)) normalized
        weights = 1 / (1 + dists[i])
        probs = weights / np.sum(weights)
        
        # Construct soft probability dictionary
        p_dict = {f'prob_pheno_{label}': probs[idx] for idx, label in enumerate(centroid_labels)}
        soft_probs.append(p_dict)
        
    df_phenos = pd.DataFrame(soft_probs)
    
    # FIX: Robustly remove noise column (-1) which wasn't in training
    cols_to_drop = [c for c in df_phenos.columns if str(c).endswith('_-1') or str(c).endswith('_-1.0')]
    if cols_to_drop:
        df_phenos = df_phenos.drop(columns=cols_to_drop)

    df_phenos = df_phenos.reindex(sorted(df_phenos.columns), axis=1) # Sort cols
    
    # D. Prepare Model Inputs
    # Trees need: Raw + PhenotypeID
    X_hard = pd.DataFrame(X_imputed, columns=pkg['features'])
    X_hard['phenotype_id'] = assigned_phenos
    
    # SVC needs: Raw + SoftProbs
    X_soft = pd.concat([pd.DataFrame(X_imputed, columns=pkg['features']), df_phenos], axis=1)
    
    # 4. Predict
    print("\n[PREDICTING] Running Ensemble Vote...")
    p1 = pkg['lgbm_model'].predict_proba(X_hard)[:, 1]
    p2 = pkg['xgb_model'].predict_proba(X_hard)[:, 1]
    
    # Ensure columns match for SVC
    if hasattr(pkg['svc_model'], 'feature_names_in_'):
         expected_cols = pkg['svc_model'].feature_names_in_
         X_soft = X_soft.reindex(columns=expected_cols, fill_value=0.0)

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
        status = "CRITICAL" if score > 0.5 else "STABLE"
        
        print(f"{i:<10} | {desc:<20} | {score:>10.1%} | {status}")
        
    print("="*80)
    return p_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA ICU Inference")
    # UPDATED DEFAULT: Points to 'Agent A/demo/dummy.csv'
    parser.add_argument("--input", type=str, default="dummy.csv", help="Path to input CSV file")
    args = parser.parse_args()
    
    predict_new_patients(args.input)

"""
SOTA Inference Engine (The "Easy Button")
Usage: python run_demo.py
Result: outputs risk scores and phenotype assignments for the dummy patients.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.metrics.pairwise import euclidean_distances

# Suppress warnings for cleaner demo output
warnings.filterwarnings("ignore")

def predict_demo_patients():
    print("=" * 80)
    print("🏥 SOTA ICU PREDICTOR: LOADING BRAIN...")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Paths (robust, repo-safe)
    # ------------------------------------------------------------------
    pkg_path = os.path.join("models", "sota_ensemble_pkg.pkl")
    # Path to the external HDBSCAN model you just uploaded
    hdb_external_path = os.path.join("models", "hdbscan_clusterer.joblib")
    
    input_csv = "dummy.csv"   # demo-local CSV

    if not os.path.exists(pkg_path):
        raise FileNotFoundError(f"Model package not found at {pkg_path}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found at {input_csv}")

    # ------------------------------------------------------------------
    # 2. Load model package
    # ------------------------------------------------------------------
    pkg = joblib.load(pkg_path)
    print("[OK] Model Package Loaded.")

    required_keys = [
        "features",
        "imputer",
        "umap_reducer",
        "lgbm_model",
        "xgb_model",
        "svc_model",
        "weights"
    ]

    for k in required_keys:
        if k not in pkg:
            raise KeyError(f"Model package missing required key: '{k}'")

    # ------------------------------------------------------------------
    # 3. Load and validate input data
    # ------------------------------------------------------------------
    data = pd.read_csv(input_csv)
    print(f"[OK] Loaded input CSV with shape {data.shape}")

    missing_cols = [c for c in pkg["features"] if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")

    data = data[pkg["features"]]

    # ------------------------------------------------------------------
    # 4. Micro-Pipeline (EDA → UMAP → Phenotype)
    # ------------------------------------------------------------------
    print("\n[PROCESSING] Running Transformations...")

    # A. Imputation (NO fit at inference)
    X_imputed = pkg["imputer"].transform(data)

    # B. UMAP projection
    X_umap = pkg["umap_reducer"].transform(X_imputed)

    # ------------------------------------------------------------------
    # C. Phenotype inference (Updated Logic)
    # ------------------------------------------------------------------
    assigned_phenos = []
    df_phenos = None
    hdb_model = None

    # Logic: Check external file -> Check inside package -> Fallback
    if os.path.exists(hdb_external_path):
        try:
            hdb_model = joblib.load(hdb_external_path)
            print(f"[OK] Loaded HDBSCAN from external file: {hdb_external_path}")
        except Exception as e:
            print(f"[WARN] Failed to load external HDBSCAN: {e}")
    elif "hdbscan_model" in pkg:
        print("[OK] Loaded HDBSCAN from model package.")
        hdb_model = pkg["hdbscan_model"]

    # --- Execution Path A: HDBSCAN Inference ---
    if hdb_model is not None:
        try:
            # 1. Hard Assignment
            labels, strengths = hdb_model.approximate_predict(X_umap)
            assigned_phenos = labels.tolist()

            # 2. Soft Probabilities (Deep Phenotypes)
            soft_probs = hdb_model.membership_vector(X_umap)
            
            # Create DataFrame with correct column names
            df_phenos = pd.DataFrame(
                soft_probs,
                columns=[f"prob_pheno_{i}" for i in range(soft_probs.shape[1])]
            )
            
        except AttributeError:
             # Handle case where approximate_predict isn't available (version mismatch)
             print("[WARN] HDBSCAN version mismatch/method missing. Reverting to centroids.")
             hdb_model = None 

    # --- Execution Path B: Centroid Fallback ---
    if hdb_model is None and "umap_centroids" in pkg:
        print("[WARN] HDBSCAN not usable — using centroid fallback.")

        centroids = pkg["umap_centroids"].values
        centroid_labels = pkg["umap_centroids"].index.values

        dists = euclidean_distances(X_umap, centroids)
        soft_rows = []

        for i in range(len(X_umap)):
            # Temperature-scaled Softmax based on distance
            weights = np.exp(-dists[i] / 0.5)   
            probs = weights / np.sum(weights)

            # Hard Assignment (Closest Centroid)
            assigned_phenos.append(
                centroid_labels[np.argmax(probs)]
            )

            # Soft Assignment
            soft_rows.append({
                f"prob_pheno_{label}": probs[idx]
                for idx, label in enumerate(centroid_labels)
            })

        df_phenos = pd.DataFrame(soft_rows)

    elif hdb_model is None:
        raise RuntimeError(
            "CRITICAL ERROR: No phenotype inference method found. "
            "Ensure 'hdbscan_clusterer.joblib' is in models/ or 'umap_centroids' is in pkg."
        )

    # Clean up: Drop noise column if present (usually -1)
    # We filter columns ending in "_-1" just in case
    df_phenos = df_phenos.loc[
        :, ~df_phenos.columns.astype(str).str.endswith("_-1")
    ]

    # Sort columns to ensure feature alignment matches training
    df_phenos = df_phenos.reindex(sorted(df_phenos.columns), axis=1)

    # ------------------------------------------------------------------
    # 5. Prepare model inputs
    # ------------------------------------------------------------------
    # Input for Trees (Raw + Hard ID)
    X_hard = pd.DataFrame(X_imputed, columns=pkg["features"])
    X_hard["phenotype_id"] = assigned_phenos

    # Input for SVC (Raw + Soft Probs)
    X_soft = pd.concat(
        [pd.DataFrame(X_imputed, columns=pkg["features"]), df_phenos],
        axis=1
    )

    # Align SVC features exactly (Critical for Sklearn)
    if hasattr(pkg["svc_model"], "feature_names_in_"):
        # This fills missing phenotype columns with 0.0 if the test set didn't hit them
        X_soft = X_soft.reindex(
            columns=pkg["svc_model"].feature_names_in_,
            fill_value=0.0
        )

    # ------------------------------------------------------------------
    # 6. Ensemble prediction
    # ------------------------------------------------------------------
    print("\n[PREDICTING] Running Ensemble Vote...")

    # LightGBM Prediction
    p1 = pkg["lgbm_model"].predict_proba(X_hard)[:, 1]
    
    # XGBoost Prediction
    p2 = pkg["xgb_model"].predict_proba(X_hard)[:, 1]
    
    # SVC Prediction
    p3 = pkg["svc_model"].predict_proba(X_soft)[:, 1]

    # Weighted Average
    w = pkg["weights"]
    p_final = (w[0] * p1 + w[1] * p2 + w[2] * p3) / sum(w)

    # ------------------------------------------------------------------
    # 7. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Patient':<10} | {'Phenotype':<28} | {'Risk Score':<12} | Outlook")
    print("-" * 80)

    clinical_map = {
        12: "Respiratory Failure (High)",
        13: "Sepsis / Renal Failure",
        27: "Frailty / Cardiac",
        1:  "Standard Recovery"
    }

    for i, score in enumerate(p_final):
        pid = assigned_phenos[i]
        label = clinical_map.get(pid, f"Phenotype {pid}")
        
        # Determine Status
        if score > 0.7:
            status = "🔴 CRITICAL"
        elif score > 0.3:
            status = "🟡 GUARDED"
        else:
            status = "🟢 STABLE"

        print(f"{i:<10} | {label:<28} | {score:>10.1%} | {status}")

    print("=" * 80)


if __name__ == "__main__":
    predict_demo_patients()
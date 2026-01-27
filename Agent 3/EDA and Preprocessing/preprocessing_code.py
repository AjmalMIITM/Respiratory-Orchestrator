import pandas as pd
import numpy as np

def smart_preprocessing(df):
    # ==========================================
    # 1. SAFE SPLIT (Protect IDs/Dates)
    # ==========================================
    # We identify metadata columns to exclude from processing
    meta_cols = ['visit_occurrence_id', 'extubation_time', 'success_48h']
    
    # Separate them: df_meta stays raw, df_feats gets cleaned
    df_meta = df[meta_cols].copy()
    df_feats = df.drop(columns=meta_cols).copy()
    
    # ==========================================
    # 2. PHYSICS ENGINE (Fixing Errors)
    # ==========================================
    # SpO2: Fix 0.98 vs 98 issue and Clip
    if 'median_spo2' in df_feats.columns:
        mask_low_spo2 = (df_feats['median_spo2'] > 0) & (df_feats['median_spo2'] <= 1.0)
        df_feats.loc[mask_low_spo2, 'median_spo2'] = df_feats.loc[mask_low_spo2, 'median_spo2'] * 100
        df_feats['median_spo2'] = df_feats['median_spo2'].clip(50, 100)
    
    # Clip Valid Ranges (Physiological Limits)
    if 'rsbi_last' in df_feats.columns: df_feats['rsbi_last'] = df_feats['rsbi_last'].clip(0, 300)
    if 'last_ps' in df_feats.columns: df_feats['last_ps'] = df_feats['last_ps'].clip(0, 40)
    if 'driving_pressure' in df_feats.columns: df_feats['driving_pressure'] = df_feats['driving_pressure'].clip(0, 40)
    if 'total_urine_24h' in df_feats.columns: df_feats['total_urine_24h'] = df_feats['total_urine_24h'].clip(0, 10000)
    if 'max_hr' in df_feats.columns: df_feats['max_hr'] = df_feats['max_hr'].clip(30, 250)

    # ==========================================
    # 3. SMART IMPUTATION ("Missing = Normal")
    # ==========================================
    # We only fill LABS/NEURO with "Healthy Values" if missing.
    normal_values = {
        'last_ph': 7.40,        
        'min_ph': 7.40,
        'last_paco2': 40.0,     
        'max_paco2': 40.0,
        'last_pao2': 90.0,      
        'min_pao2': 90.0,
        'last_bicarb': 24.0,    
        'last_lactate': 11.0,   # (approx 1.2 mmol/L on mg/dL scale)
        'max_lactate': 11.0,
        'median_rass': 0.0,     # Alert/Calm
        'last_rass': 0.0,
        'last_creatinine': 0.9, 
        'last_bun': 15.0,       
        'last_hemoglobin': 12.0 
    }
    
    print(" Applying Clinical Normal Imputation to Labs...")
    df_feats = df_feats.fillna(normal_values)
    
    # ==========================================
    # 4. MEDIAN FALLBACK (Vitals Only)
    # ==========================================
    # If Vitals (HR, RR, SpO2) are missing, it's data loss, not stability.
    # We use median for these specific columns only.
    print(" Cleaning remaining missing vitals with Median...")
    df_feats = df_feats.fillna(df_feats.median())
    
    # ==========================================
    # 5. RECOMBINE
    # ==========================================
    # Put the ID/Target columns back so we have a complete, clean dataset
    df_clean = pd.concat([df_meta, df_feats], axis=1)
    
    return df_clean

# --- EXECUTION ---
# Load your raw V6 data
filename = 'amsterdam_stage3_advisor_matrix_v6.csv'
df_raw = pd.read_csv(filename)

# Clean it
df_ready = smart_preprocessing(df_raw)

# Save
output_filename = 'amsterdam_stage3_advisor_matrix_CLEANED.csv'
df_ready.to_csv(output_filename, index=False)

print(f"\n Data Processed successfully.")
print(f"   Original Shape: {df_raw.shape}")
print(f"   Cleaned Shape:  {df_ready.shape}")
print(f"   Missing Values Remaining: {df_ready.isnull().sum().sum()}")

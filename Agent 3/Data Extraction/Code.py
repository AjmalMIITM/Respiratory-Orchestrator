from google.cloud import bigquery
import pandas as pd
import os

# =================================================================
# PROJECT: Agent 3 - The Advisor (Extubation Readiness)
# SCRIPT: 03_Extract_Stage3_Matrix_v6.py
# PURPOSE: Extract final dataset with ALL Corrected Concept IDs & Logic.
# STATUS: GOLD MASTER (V6)
# CHANGELOG:
#   1. Fixed Outcome Logic: Uses COUNT(column) instead of COUNT(*) to avoid 100% failure rate.
#   2. Fixed Missing Data: Uses IGNORE NULLS in ARRAY_AGG.
#   3. Fixed IDs: Includes high-volume IDs for Pressure Support, FiO2, PaO2.
# =================================================================

# !!! UPDATE THIS TO YOUR ACTUAL PROJECT ID !!!
PROJECT_ID = 'main-audio-484603-n8' 
client = bigquery.Client(project=PROJECT_ID)

print(f" Starting Stage-3 Extraction (v6 - Final Gold Master)...")

SQL_QUERY = """
WITH 
const AS (
    SELECT 
        24 as LOOKBACK_WINDOW_HOURS,
        48 as OUTCOME_WINDOW_HOURS,
        2 as RECOVERY_BUFFER_HOURS
),

-- 1. IDENTIFY THE PULL (T0)
-- We find sequences of ventilation. A 'gap' of >= 24h implies the tube was pulled.
vent_sequences AS (
    SELECT 
        v.visit_occurrence_id,
        v.person_id,
        m.measurement_datetime,
        TIMESTAMP_DIFF(
            LEAD(m.measurement_datetime) OVER(PARTITION BY v.visit_occurrence_id ORDER BY m.measurement_datetime),
            m.measurement_datetime, 
            HOUR
        ) as gap_until_next,
        TIMESTAMP_DIFF(m.measurement_datetime, MIN(m.measurement_datetime) OVER(PARTITION BY v.visit_occurrence_id), HOUR) as hours_ventilated
    FROM `amsterdamumcdb.van_gogh_2026_datathon.visit_occurrence` v
    JOIN `amsterdamumcdb.van_gogh_2026_datathon.measurement` m 
      ON v.person_id = m.person_id 
      AND m.measurement_datetime BETWEEN v.visit_start_datetime AND v.visit_end_datetime
    WHERE m.measurement_concept_id IN (3022875, 2000000238, 44782825) -- PEEP, Ppeak OR Pplateau
),

vent_episodes AS (
    SELECT 
        visit_occurrence_id,
        person_id,
        MIN(measurement_datetime) as t0 
    FROM vent_sequences
    WHERE hours_ventilated >= 24 
      -- Handle NULL gaps (Implies Last event = Successful Extubation or Discharge)
      AND (gap_until_next >= 24 OR gap_until_next IS NULL) 
    GROUP BY 1, 2
),

-- 2. "STABILITY" PREDICTORS (T-24h to T0)
predictors_raw AS (
    SELECT 
        ve.visit_occurrence_id,
        m.measurement_concept_id,
        m.measurement_datetime, 
        
        -- NORMALIZE FiO2 (Handle 0.21-1.0 and 21-100 ranges)
        CASE 
            WHEN m.measurement_concept_id = 2000000204 AND m.value_as_number > 1.0 THEN m.value_as_number / 100.0
            WHEN m.measurement_concept_id = 2000000204 AND m.value_as_number <= 1.0 THEN m.value_as_number
            ELSE m.value_as_number
        END as value_clean,
        
        -- Apply Unit Conversions
        CASE 
            WHEN m.measurement_concept_id IN (3018405) THEN m.value_as_number * 9.008   -- Lactate mmol->mg/dL
            WHEN m.measurement_concept_id IN (3020564) THEN m.value_as_number / 88.4    -- Cr umol->mg/dL
            WHEN m.measurement_concept_id IN (40762351) THEN m.value_as_number * 1.61   -- Hb mmol->g/dL
            WHEN m.measurement_concept_id IN (3043995) THEN m.value_as_number / 17.1    -- Bili umol->mg/dL
            WHEN m.measurement_concept_id IN (3020779) THEN m.value_as_number * 2.8     -- Urea->BUN
            ELSE m.value_as_number
        END as val_std

    FROM vent_episodes ve
    CROSS JOIN const c
    JOIN `amsterdamumcdb.van_gogh_2026_datathon.measurement` m 
      ON ve.person_id = m.person_id 
      AND m.measurement_datetime BETWEEN DATETIME_SUB(ve.t0, INTERVAL c.LOOKBACK_WINDOW_HOURS HOUR) AND ve.t0
    WHERE m.measurement_concept_id IN (
        -- Vent Mechanics (Validated IDs)
        2000000238, 44782825, 3022875, 3015016, 3024882, 3024171, 
        2000000204, 21490752, 36303816, 
        3000461, 2000000211, 2000000209, -- High Volume Pressure Support
        
        -- Oxygenation & Gas
        40762499, 3019977, 3008152, 3018405, 3013290,
        3035357, 3027801, 3027315, -- High Volume PaO2
        3016502, -- Secondary SpO2
        
        -- Neuro
        3014582, 3016335, 3009094, 3021119, 2000000016,
        
        -- Labs/Vitals
        3014315, 3020564, 40762351, 43534077, 21490872, 3020891, 3020779
    )
    AND m.value_as_number IS NOT NULL
),

predictors_agg AS (
    SELECT 
        visit_occurrence_id,
        
        -- === VENTILATOR ===
        MAX(CASE WHEN measurement_concept_id IN (2000000238, 44782825) THEN val_std END) as max_ppeak,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (2000000238, 44782825) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_ppeak,
        
        AVG(CASE WHEN measurement_concept_id IN (3022875) THEN val_std END) as median_peep,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3022875) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_peep,
        
        -- PRESSURE SUPPORT (All IDs)
        MAX(CASE WHEN measurement_concept_id IN (3000461, 2000000211, 2000000209) THEN val_std END) as max_ps,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3000461, 2000000211, 2000000209) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_ps,

        -- TIDAL VOLUME (Coalesce Expired > Inspired, Normalize mL->L)
        AVG(CASE 
            WHEN measurement_concept_id IN (3015016, 21490752, 36303816) AND val_std > 10 THEN val_std / 1000.0
            WHEN measurement_concept_id IN (3015016, 21490752, 36303816) AND val_std <= 10 THEN val_std
            ELSE NULL END
        ) as median_tv_liters,
        
        ARRAY_AGG(CASE 
            WHEN measurement_concept_id IN (3015016, 21490752, 36303816) AND val_std > 10 THEN val_std / 1000.0 
            WHEN measurement_concept_id IN (3015016, 21490752, 36303816) AND val_std <= 10 THEN val_std
            ELSE NULL END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_tv_liters,

        -- FiO2 (Normalized)
        MAX(CASE WHEN measurement_concept_id IN (2000000204, 3024882) THEN value_clean END) as max_fio2,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (2000000204, 3024882) THEN value_clean END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_fio2,

        AVG(CASE WHEN measurement_concept_id IN (3024171) THEN val_std END) as median_rr,
        STDDEV(CASE WHEN measurement_concept_id IN (3024171) THEN val_std END) as std_rr,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3024171) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_rr,

        -- === OXYGENATION & LABS ===
        MIN(CASE WHEN measurement_concept_id IN (40762499, 3016502) THEN val_std END) as min_spo2, 
        AVG(CASE WHEN measurement_concept_id IN (40762499, 3016502) THEN val_std END) as median_spo2,
        
        MIN(CASE WHEN measurement_concept_id IN (3019977) THEN val_std END) as min_ph,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3019977) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_ph,

        MAX(CASE WHEN measurement_concept_id IN (3013290) THEN val_std END) as max_paco2, 
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3013290) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_paco2,
        
        -- PaO2 (All IDs)
        MIN(CASE WHEN measurement_concept_id IN (3027315, 3035357, 3027801) THEN val_std END) as min_pao2,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3027315, 3035357, 3027801) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_pao2,
        
        MAX(CASE WHEN measurement_concept_id IN (3018405) THEN val_std END) as max_lactate,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3018405) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_lactate,
        
        -- === NEURO & FLUIDS ===
        COALESCE(
            MIN(CASE WHEN measurement_concept_id = 3014582 THEN val_std END),
            SUM(CASE WHEN measurement_concept_id IN (3016335, 3009094, 3021119) THEN val_std END) / NULLIF(COUNT(DISTINCT CASE WHEN measurement_concept_id IN (3016335, 3009094, 3021119) THEN measurement_datetime END), 0)
        ) as min_gcs,
        
        AVG(CASE WHEN measurement_concept_id IN (2000000016) THEN val_std END) as median_rass,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (2000000016) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_rass,

        SUM(CASE WHEN measurement_concept_id IN (3014315) THEN val_std END) as total_urine_24h,
        
        MAX(CASE WHEN measurement_concept_id IN (21490872, 3020891) THEN val_std END) as max_hr,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3020564) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_creatinine,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (40762351) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_hemoglobin,
        
        -- BUN (Urea)
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3020779, 43534077) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_bun,
        ARRAY_AGG(CASE WHEN measurement_concept_id IN (3008152) THEN val_std END IGNORE NULLS ORDER BY measurement_datetime DESC LIMIT 1)[OFFSET(0)] as last_bicarb

    FROM predictors_raw
    GROUP BY 1
),

-- 3. THE OUTCOME
outcome_audit AS (
    SELECT 
        ve.visit_occurrence_id,
        -- FIX: Count the ACTUAL MATCHING ROWS, not just the patient group
        COUNT(m.measurement_concept_id) as failure_signals
    FROM vent_episodes ve
    CROSS JOIN const c
    LEFT JOIN `amsterdamumcdb.van_gogh_2026_datathon.measurement` m 
      ON ve.person_id = m.person_id 
      AND m.measurement_datetime BETWEEN DATETIME_ADD(ve.t0, INTERVAL c.RECOVERY_BUFFER_HOURS HOUR) 
                                     AND DATETIME_ADD(ve.t0, INTERVAL c.OUTCOME_WINDOW_HOURS HOUR)
      AND m.measurement_concept_id IN (3022875, 2000000238, 44782825) -- Invasive Vent IDs
      AND m.value_as_number > 5 
    GROUP BY 1
)

-- 4. FINAL MATRIX
SELECT 
    ve.visit_occurrence_id,
    ve.t0 as extubation_time,
    
    -- Features
    p.last_rr / NULLIF(p.last_tv_liters, 0) as rsbi_last,
    p.median_rr / NULLIF(p.median_tv_liters, 0) as rsbi_median,
    p.last_pao2 / NULLIF(p.last_fio2, 0) as pf_ratio_last,
    
    CASE 
        WHEN p.last_ppeak IS NOT NULL AND p.last_peep IS NOT NULL 
        THEN p.last_ppeak - p.last_peep 
        ELSE NULL 
    END as driving_pressure,
    
    p.max_ppeak, p.last_ppeak,
    p.max_ps, p.last_ps,
    p.last_fio2, p.max_fio2,
    p.median_peep, p.last_peep,
    p.median_rr, p.last_rr, p.std_rr,
    p.min_spo2, p.median_spo2, 
    p.min_ph, p.last_ph,
    p.max_paco2, p.last_paco2,
    p.min_pao2, p.last_pao2,
    p.max_lactate, p.last_lactate,
    p.min_gcs,
    p.median_rass, p.last_rass,
    p.total_urine_24h,
    p.max_hr,
    p.last_creatinine,
    p.last_hemoglobin,
    p.last_bun,
    p.last_bicarb,

    -- Target: 0 = Failure (Signal > 0), 1 = Success
    CASE WHEN COALESCE(oa.failure_signals, 0) > 0 THEN 0 ELSE 1 END as success_48h

FROM vent_episodes ve
LEFT JOIN predictors_agg p ON ve.visit_occurrence_id = p.visit_occurrence_id
LEFT JOIN outcome_audit oa ON ve.visit_occurrence_id = oa.visit_occurrence_id
"""

try:
    print(" Running Grand Join Query (v6)...")
    df = client.query(SQL_QUERY).to_dataframe()
    print(f" Extraction Complete! Shape: {df.shape}")
    
    # Validation
    print("\nSuccess Rate (Should be ~80-90%):")
    print(df['success_48h'].value_counts(normalize=True))
    
    filename = 'amsterdam_stage3_advisor_matrix_v6.csv'
    df.to_csv(filename, index=False)
    print(f" Saved to: {filename}")
    
except Exception as e:
    print(f" Error during extraction: {e}")

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output



# 1. DEFINE THE MISSING CLASS
# (This MUST come before joblib.load)
class InvertedModel:
    """
    Wrapper that inverts predict_proba output.
    Assumes base_model.predict_proba returns [P(class0), P(class1)]
    and swaps success/failure interpretation.
    """
    def __init__(self, base_model):
        self.base_model = base_model

    def predict_proba(self, X):
        probs = self.base_model.predict_proba(X)
        return 1 - probs

    def predict(self, X):
        return self.base_model.predict(X)



model = joblib.load('/content/agent3_FAILURE_READY.joblib')



# 3. PREDICTION + PLOT FUNCTION
# =========================
def predict_and_plot(min_spo2, last_fio2, median_rr, last_ppeak, last_peep, last_ps):

    # ---- Driving Pressure ----
    driving_pressure = last_ppeak - last_peep

    # ---- Feature Vector ----
    features = [
        'min_spo2',
        'last_fio2',
        'median_rr',
        'last_ppeak',
        'driving_pressure',
        'last_peep',
        'last_ps'
    ]

    input_df = pd.DataFrame([[
        min_spo2,
        last_fio2,
        median_rr,
        last_ppeak,
        driving_pressure,
        last_peep,
        last_ps
    ]], columns=features)

    # ---- Prediction ----
    prob_success = model.predict_proba(input_df)[0, 1]
    risk = (1 - prob_success) * 100

    # ---- Visualization ----
    clear_output(wait=True)

    plt.figure(figsize=(8, 2))
    color = 'green' if risk < 50 else 'orange' if risk < 60 else 'red'

    plt.barh(['Failure Risk'], [risk], color=color)
    plt.axvline(50, color='black', linestyle='--', label='Threshold')
    plt.xlim(0, 100)
    plt.xlabel("Probability of Extubation Failure (%)")
    plt.title(f"Agent 3 Prediction: {risk:.1f}% Risk")
    plt.legend()
    plt.show()

    # ---- Clinical Logic Output ----
    print("--- Clinical Summary ---")
    print(f"Calculated Driving Pressure: {driving_pressure:.1f} cmH₂O")

    if driving_pressure > 15:
        print(" WARNING: High Driving Pressure (>15 cmH₂O)")

    if risk > 70:
        print(" ALERT: High risk of extubation failure — reassess weaning readiness")


# =========================
# 4. INTERACTIVE SLIDERS
# =========================
widgets.interact(
    predict_and_plot,
    min_spo2=widgets.IntSlider(
        min=80, max=100, step=1, value=98, description='SpO₂ %'
    ),
    last_fio2=widgets.FloatSlider(
        min=0.21, max=1.0, step=0.01, value=0.35, description='FiO₂'
    ),
    median_rr=widgets.IntSlider(
        min=10, max=45, step=1, value=18, description='Resp Rate'
    ),
    last_ppeak=widgets.IntSlider(
        min=10, max=50, step=1, value=20, description='Peak Press'
    ),
    last_peep=widgets.IntSlider(
        min=5, max=20, step=1, value=5, description='PEEP'
    ),
    last_ps=widgets.IntSlider(
        min=5, max=25, step=1, value=5, description='Press Support'
    )
);

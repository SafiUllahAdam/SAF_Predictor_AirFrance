import gradio as gr
import joblib
import pandas as pd
import numpy as np
import shap


# 1️ Load Trained Model

model = joblib.load("lightgbm_model.pkl")


# 2 Define Mappings (same encodings as in training)

MAP_TIER   = {"Silver": 0, "Gold": 1, "Platinum": 2}
MAP_CLASS  = {"Economy": 0, "Premium": 1, "Business": 2}
MAP_REGION = {"Domestic": 0, "EU": 1, "Intercontinental": 2}

# Features collected from user (8)
APP_FEATURES = [
    "loyalty_tier", "booking_class", "past_saf_purchase", "adds_paid_options",
    "num_passengers", "destination_region", "flight_distance_km", "ticket_price_eur"
]

# Features used by model during training (auto-read)
MODEL_FEATURES = list(getattr(model, "feature_name_", []))



# 3 Helper Functions

def _build_app_row(loyalty_tier, booking_class, past_saf_purchase, adds_paid_options,
                   num_passengers, destination_region, flight_distance_km, ticket_price_eur) -> pd.DataFrame:
    """Create one encoded row from UI input."""
    row = [[
        MAP_TIER[loyalty_tier],
        MAP_CLASS[booking_class],
        int(past_saf_purchase),
        int(adds_paid_options),
        int(num_passengers),
        MAP_REGION[destination_region],
        float(flight_distance_km),
        float(ticket_price_eur),
    ]]
    return pd.DataFrame(row, columns=APP_FEATURES)


def _align_to_model_columns(app_row: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the row matches model training columns exactly.
    Missing columns are filled with zero.
    """
    if not MODEL_FEATURES:
        return app_row.copy()

    full = pd.DataFrame({c: [0] for c in MODEL_FEATURES})
    for c in app_row.columns:
        if c in full.columns:
            full[c] = app_row[c].values
    return full



# 4️ Core Functions (Prediction + Recommendation)

def predict_saf(loyalty_tier, booking_class, past_saf_purchase, adds_paid_options,
                num_passengers, destination_region, flight_distance_km, ticket_price_eur):
    try:
        app_row  = _build_app_row(loyalty_tier, booking_class, past_saf_purchase, adds_paid_options,
                                  num_passengers, destination_region, flight_distance_km, ticket_price_eur)
        sample   = _align_to_model_columns(app_row)

        pred = int(model.predict(sample)[0])
        return ("Higher Probability of Contributing to SAF according to Customer Details"
                if pred == 1
                else "Very Less Probability of Contributing to SAF according to the Details")
    except Exception as e:
        return f"Error in prediction: {e}"


def recommend_saf_offer(loyalty_tier, booking_class, past_saf_purchase, adds_paid_options,
                        num_passengers, destination_region, flight_distance_km, ticket_price_eur):
    try:
        app_row = _build_app_row(loyalty_tier, booking_class, past_saf_purchase, adds_paid_options,
                                 num_passengers, destination_region, flight_distance_km, ticket_price_eur)
        sample  = _align_to_model_columns(app_row)

        pred = int(model.predict(sample)[0])
        if pred == 0:
            return ("Low probability of SAF contribution.\n"
                    "Suggestion: show awareness message or small incentive (€5–€10).")

        # SHAP Explainability 
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(sample)

            impacts = shap_vals.values[0] if hasattr(shap_vals, "values") else shap_vals[0]
            importance = (pd.DataFrame({"feature": sample.columns, "impact": impacts})
                            .assign(abs_impact=lambda d: d["impact"].abs())
                            .sort_values("abs_impact", ascending=False))
            top_feats = ", ".join(importance.head(3)["feature"].tolist())
        except Exception as e:
            return f"Error during SHAP explanation: {e}"

        # Simple Offer Logic 
        saf_pct = 25 if loyalty_tier == "Platinum" else (15 if booking_class == "Business" else 10)
        price_suggestion = "€20–€50" if saf_pct >= 15 else "€10–€25"

        return (f"High probability of contribution.\n"
                f"Recommended offer: SAF {saf_pct}% at {price_suggestion}.\n"
                f"Key drivers: {top_feats}")

    except Exception as e:
        return f"Error in recommendation: {e}"



# 5️ Gradio UI

with gr.Blocks(title="Air France — SAF Contribution") as demo:
    gr.Markdown("# Air France — SAF Contribution Predictor\nEnter customer details and click an action below.")

    with gr.Row():
        with gr.Column(scale=1):
            loyalty = gr.Dropdown(["Silver", "Gold", "Platinum"], label="Loyalty Tier", value="Gold")
            bclass  = gr.Dropdown(["Economy", "Premium", "Business"], label="Booking Class", value="Premium")
            past    = gr.Radio([0, 1], label="Past SAF Purchase (0 = No, 1 = Yes)", value=0)
            addopt  = gr.Radio([0, 1], label="Adds Paid Options (0 = No, 1 = Yes)", value=0)
            pax     = gr.Slider(1, 5, step=1, label="Number of Passengers", value=2)
            region  = gr.Dropdown(["Domestic", "EU", "Intercontinental"], label="Destination Region", value="Domestic")
            dist    = gr.Number(label="Flight Distance (km)", value=1200)
            price   = gr.Number(label="Ticket Price (€)", value=500)

            with gr.Row():
                btn_predict = gr.Button("Predict SAF Contribution", variant="primary")
                btn_offer   = gr.Button("Recommend SAF Offer")

        with gr.Column(scale=1):
            pred_out = gr.Textbox(label="Prediction", lines=3)
            rec_out  = gr.Textbox(label="Recommendation", lines=5)

    inputs = [loyalty, bclass, past, addopt, pax, region, dist, price]
    btn_predict.click(predict_saf, inputs, pred_out)
    btn_offer.click(recommend_saf_offer, inputs, rec_out)

if __name__ == "__main__":
    demo.launch()

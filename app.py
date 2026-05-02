"""
StaySure — Hotel Booking Cancellation Predictor
Gradio web demo. Loads a pre-trained Random Forest pipeline and predicts
whether a hotel booking will be cancelled.

Run locally:  uv run python app.py
Deploy to:    Hugging Face Spaces (see README)
"""
import joblib
import pathlib
import pandas as pd
import gradio as gr

# ---------------------------------------------------------------------------
# Load the trained pipeline (train_and_save.py must have been run first)
# ---------------------------------------------------------------------------
MODEL_PATH = pathlib.Path("rf_pipeline.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "rf_pipeline.joblib not found. "
        "Run: uv run python scripts/train_and_save.py"
    )

pipeline = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Default values for features not exposed in the UI
# (median / most-common values from the training set)
# ---------------------------------------------------------------------------
DEFAULTS = {
    'arrival_date_year':              2024,
    'arrival_date_month':             'July',
    'arrival_date_week_number':       27,
    'arrival_date_day_of_month':      15,
    'stays_in_weekend_nights':        1,
    'adults':                         2,
    'children':                       0.0,
    'babies':                         0,
    'meal':                           'BB',
    'country':                        'GBR',
    'distribution_channel':           'TA/TO',
    'is_repeated_guest':              0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type':             'A',
    'assigned_room_type':             'A',
    'booking_changes':                0,
    'days_in_waiting_list':           0,
    'adr':                            100.0,
    'required_car_parking_spaces':    0,
    'agent':                          'Unknown',
    'company':                        'Unknown',
}

# ---------------------------------------------------------------------------
# Prediction function — called on every UI submit
# ---------------------------------------------------------------------------
def predict(
    hotel,
    lead_time,
    deposit_type,
    customer_type,
    market_segment,
    total_of_special_requests,
    previous_cancellations,
    stays_in_week_nights,
):
    row = {**DEFAULTS}
    row['hotel']                      = hotel
    row['lead_time']                  = lead_time
    row['deposit_type']               = deposit_type
    row['customer_type']              = customer_type
    row['market_segment']             = market_segment
    row['total_of_special_requests']  = total_of_special_requests
    row['previous_cancellations']     = previous_cancellations
    row['stays_in_week_nights']       = stays_in_week_nights

    df = pd.DataFrame([row])
    prob = pipeline.predict_proba(df)[0][1]   # probability of cancellation
    label = pipeline.predict(df)[0]            # 0 = stays, 1 = cancels

    if label == 1:
        verdict = "Likely to CANCEL"
        detail  = f"Cancellation probability: **{prob:.1%}**"
    else:
        verdict = "Likely to STAY"
        detail  = f"Cancellation probability: **{prob:.1%}**"

    tips = []
    if deposit_type == "Non Refund":
        tips.append("Non-refundable deposits strongly predict cancellation.")
    if lead_time > 150:
        tips.append(f"Long lead time ({lead_time} days) increases cancellation risk.")
    if previous_cancellations > 0:
        tips.append(f"{previous_cancellations} prior cancellation(s) on record.")
    if total_of_special_requests >= 3:
        tips.append("High special requests suggest a committed guest.")

    tip_text = "\n\n**Key factors:**\n" + "\n".join(f"- {t}" for t in tips) if tips else ""

    return f"{verdict}\n\n{detail}{tip_text}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="StaySure — Cancellation Predictor") as demo:
    gr.Markdown(
        """
        # StaySure — Hotel Booking Cancellation Predictor
        Fill in the booking details below to predict whether the booking will be cancelled.
        Model: Random Forest · F1 = 0.85 · AUC = 0.958 · Trained on 95k hotel bookings.
        """
    )

    with gr.Row():
        with gr.Column():
            hotel = gr.Dropdown(
                choices=["City Hotel", "Resort Hotel"],
                value="City Hotel",
                label="Hotel type"
            )
            lead_time = gr.Slider(
                minimum=0, maximum=500, step=1, value=60,
                label="Lead time (days between booking and arrival)"
            )
            deposit_type = gr.Dropdown(
                choices=["No Deposit", "Non Refund", "Refundable"],
                value="No Deposit",
                label="Deposit type"
            )
            customer_type = gr.Dropdown(
                choices=["Transient", "Contract", "Group", "Transient-Party"],
                value="Transient",
                label="Customer type"
            )

        with gr.Column():
            market_segment = gr.Dropdown(
                choices=["Online TA", "Offline TA/TO", "Direct",
                         "Corporate", "Groups", "Aviation", "Complementary"],
                value="Online TA",
                label="Market segment"
            )
            total_of_special_requests = gr.Slider(
                minimum=0, maximum=5, step=1, value=0,
                label="Number of special requests"
            )
            previous_cancellations = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Previous cancellations by this customer"
            )
            stays_in_week_nights = gr.Slider(
                minimum=0, maximum=14, step=1, value=2,
                label="Number of week nights"
            )

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Markdown(label="Prediction")

    predict_btn.click(
        fn=predict,
        inputs=[
            hotel, lead_time, deposit_type, customer_type,
            market_segment, total_of_special_requests,
            previous_cancellations, stays_in_week_nights,
        ],
        outputs=output
    )

    gr.Examples(
        examples=[
            ["City Hotel",   200, "Non Refund",  "Transient",       "Online TA",      0, 2, 3],
            ["Resort Hotel",  14, "No Deposit",  "Transient",       "Direct",         3, 0, 7],
            ["City Hotel",    30, "No Deposit",  "Contract",        "Corporate",      1, 0, 2],
            ["Resort Hotel", 300, "Non Refund",  "Transient-Party", "Offline TA/TO",  0, 1, 5],
        ],
        inputs=[
            hotel, lead_time, deposit_type, customer_type,
            market_segment, total_of_special_requests,
            previous_cancellations, stays_in_week_nights,
        ],
    )

if __name__ == "__main__":
    demo.launch()

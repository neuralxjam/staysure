# StaySure — Hotel Booking Cancellation Predictor

Predict whether a hotel booking will be cancelled before it happens.
Trained on 119,390 real hotel bookings. Deployed as a live web demo.

**[Live Demo](https://huggingface.co/spaces/neuralxjam/staysure)** &nbsp;|&nbsp; **[Portfolio write-up](https://neuralxjam.github.io/projects/staysure)**

---

## Results

| Metric | Score |
|--------|-------|
| F1 Score | 0.85 |
| AUC-ROC | 0.958 |
| Accuracy | 89% |
| Test set size | 23,878 bookings |

Model comparison (all evaluated on the same held-out test set):

| Model | F1 |
|-------|----|
| Logistic Regression (baseline) | 0.759 |
| XGBoost (tuned) | 0.826 |
| **Random Forest (winner)** | **0.848** |

---

## Demo

Enter booking details — hotel type, lead time, deposit type, customer type, and a few others — and the model returns a cancellation probability with the key factors driving the prediction.

Try these examples:
- **High risk**: City Hotel, 200-day lead time, Non Refund deposit, 2 prior cancellations → ~74% cancellation probability
- **Low risk**: Resort Hotel, 14-day lead time, No Deposit, 3 special requests → low cancellation probability

---

## Stack

| Layer | Tool |
|-------|------|
| Data | [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119k rows, 32 features |
| EDA | pandas, seaborn, matplotlib |
| Preprocessing | scikit-learn `Pipeline` + `ColumnTransformer` |
| Models | Logistic Regression, Random Forest, XGBoost |
| Tuning | `GridSearchCV` (3-fold CV) |
| Evaluation | F1 score, ROC-AUC, confusion matrix |
| App | Gradio |
| Deploy | Hugging Face Spaces |

---

## Project Structure

```
staysure/
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_baseline.ipynb     # sklearn Pipeline + Logistic Regression
│   ├── 03_models.ipynb       # Random Forest, XGBoost, GridSearchCV
│   └── 04_evaluation.ipynb   # Confusion matrix, ROC curve
├── scripts/
│   └── train_and_save.py     # Retrain and serialize the model
├── assets/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── results/
│   └── runs.json             # Experiment log
├── app.py                    # Gradio web app
└── pyproject.toml
```

---

## Local Setup

```bash
git clone https://github.com/neuralxjam/staysure
cd staysure
uv sync

# Download dataset
curl -L -o data/raw/hotel_bookings.csv \
  https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv

# Train and save model (1-3 minutes)
uv run python scripts/train_and_save.py

# Run the app
uv run python app.py
```

---

## What I Learned

- **Data leakage** is the easiest way to build a model that looks great but fails in production. `reservation_status` had to be dropped because it directly encodes the answer.
- **Accuracy is a bad metric for imbalanced data.** 63/37 class split means a model that always predicts "stays" gets 63% accuracy. F1 score penalises that.
- **Random Forest beat tuned XGBoost** on this dataset. Default RF handled the mix of numeric and high-cardinality categorical features better than XGBoost's default settings.
- **sklearn Pipelines** prevent the most common ML bug: fitting the scaler on all data instead of just training data.

---

## Limitations

- Dataset is from 2015–2017. Booking patterns post-pandemic may differ significantly.
- The model predicts based on booking metadata only — it has no access to pricing data, reviews, or real-time demand signals.
- `agent` and `company` IDs are treated as opaque categories; the model cannot generalise to unseen agents.
- No fairness analysis performed across customer nationality or booking channel.

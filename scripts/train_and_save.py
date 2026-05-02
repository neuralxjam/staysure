"""
Train the winning Random Forest and save it to model/rf_pipeline.joblib.
Run once: uv run python scripts/train_and_save.py
The saved model is loaded by app.py at startup.
"""
import joblib
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

MODEL_PATH = pathlib.Path("model/rf_pipeline.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/raw/hotel_bookings.csv")
df['agent'] = df['agent'].fillna('Unknown').astype(str)
df['company'] = df['company'].fillna('Unknown').astype(str)
df = df.drop(columns=['reservation_status', 'reservation_status_date'])

X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests'
]
categorical_cols = [
    'hotel', 'arrival_date_month', 'meal', 'country',
    'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'agent', 'company', 'customer_type'
]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print("Training Random Forest... (1-3 minutes)")
pipeline.fit(X_train, y_train)

f1 = f1_score(y_test, pipeline.predict(X_test))
print(f"Done! F1 on test set: {f1:.4f}")

joblib.dump(pipeline, MODEL_PATH)
size_mb = MODEL_PATH.stat().st_size / 1_000_000
print(f"Model saved to {MODEL_PATH} ({size_mb:.1f} MB)")

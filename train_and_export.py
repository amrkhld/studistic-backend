"""
Train the Linear Regression model on StudentPerformanceFactors.csv
and export all artefacts needed by the FastAPI prediction endpoint.

Run once:  python train_and_export.py
Produces:  ml/model.pkl, ml/feature_columns.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── 1. Load data ─────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "ML", "StudentPerformanceFactors.csv")
df = pd.read_csv(CSV_PATH)

# Add Student_ID
if "Student_ID" not in df.columns:
    df["Student_ID"] = range(1, len(df) + 1)

# ── 2. Clean ─────────────────────────────────────────────────────────────────
# Fill missing values with mode
for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove impossible scores
df = df[df["Exam_Score"] != 101]

# ── 3. Encode ────────────────────────────────────────────────────────────────
ordinal_mapping = {"Low": 1, "Medium": 2, "High": 3}
for col in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level",
            "Family_Income", "Teacher_Quality"]:
    df[col] = df[col].map(ordinal_mapping)

binary_mapping = {"Yes": 1, "No": 0}
for col in ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]:
    df[col] = df[col].map(binary_mapping)

nominal_cols = ["School_Type", "Peer_Influence", "Parental_Education_Level",
                "Distance_from_Home", "Gender"]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# ── 4. Split ─────────────────────────────────────────────────────────────────
y = df["Exam_Score"]
X = df.drop(["Exam_Score", "Student_ID"], axis=1, errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train ─────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"  Test R²:   {r2_score(y_test, y_pred):.4f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print("=" * 50)

# ── 6. Compute dataset-wide statistics (used by /stats endpoint) ─────────
# Reload original CSV for raw stats
raw = pd.read_csv(CSV_PATH)
raw = raw[raw["Exam_Score"] != 101]
dataset_stats = {
    "total_students": int(len(raw)),
    "avg_exam_score": round(float(raw["Exam_Score"].mean()), 2),
    "avg_attendance": round(float(raw["Attendance"].mean()), 2),
    "avg_hours_studied": round(float(raw["Hours_Studied"].mean()), 2),
    "avg_sleep_hours": round(float(raw["Sleep_Hours"].mean()), 2),
    "avg_previous_scores": round(float(raw["Previous_Scores"].mean()), 2),
    "avg_physical_activity": round(float(raw["Physical_Activity"].mean()), 2),
    "avg_tutoring_sessions": round(float(raw["Tutoring_Sessions"].mean()), 2),
}

# ── 7. Export ─────────────────────────────────────────────────────────────────
ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
os.makedirs(ML_DIR, exist_ok=True)

with open(os.path.join(ML_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(ML_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

with open(os.path.join(ML_DIR, "dataset_stats.pkl"), "wb") as f:
    pickle.dump(dataset_stats, f)

print("\n[OK] Exported:")
print(f"  ml/model.pkl            ({os.path.getsize(os.path.join(ML_DIR, 'model.pkl'))} bytes)")
print(f"  ml/feature_columns.pkl  ({os.path.getsize(os.path.join(ML_DIR, 'feature_columns.pkl'))} bytes)")
print(f"  ml/dataset_stats.pkl    ({os.path.getsize(os.path.join(ML_DIR, 'dataset_stats.pkl'))} bytes)")
print(f"\n  Feature columns ({len(X.columns)}):")
for c in X.columns:
    print(f"    - {c}")
print(f"\n  Dataset stats: {dataset_stats}")

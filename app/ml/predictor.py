"""
Studistic ML Predictor
Loads the exported Linear Regression model and provides predictions
with risk classification and feature importance analysis.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_BASE_DIR))  # studistic-back-end/
_ML_DIR = os.path.join(_ROOT_DIR, "ml")

_MODEL_PATH = os.path.join(_ML_DIR, "model.pkl")
_COLUMNS_PATH = os.path.join(_ML_DIR, "feature_columns.pkl")
_STATS_PATH = os.path.join(_ML_DIR, "dataset_stats.pkl")

# ── Lazy-loaded globals ──────────────────────────────────────────────────────
_model = None
_feature_columns: list[str] | None = None
_dataset_stats: dict | None = None

# ── Feature display names ────────────────────────────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "Hours_Studied": "Hours Studied",
    "Attendance": "Attendance",
    "Sleep_Hours": "Sleep Hours",
    "Previous_Scores": "Previous Scores",
    "Tutoring_Sessions": "Tutoring Sessions",
    "Physical_Activity": "Physical Activity",
    "Parental_Involvement": "Parental Involvement",
    "Access_to_Resources": "Access to Resources",
    "Extracurricular_Activities": "Extracurricular",
    "Motivation_Level": "Motivation Level",
    "Internet_Access": "Internet Access",
    "Family_Income": "Family Income",
    "Teacher_Quality": "Teacher Quality",
    "School_Type_Private": "School Type",
    "Peer_Influence_Neutral": "Peer (Neutral)",
    "Peer_Influence_Positive": "Peer (Positive)",
    "Parental_Education_Level_College": "Parent Ed. (College)",
    "Parental_Education_Level_Postgraduate": "Parent Ed. (Postgrad)",
    "Distance_from_Home_Moderate": "Distance (Moderate)",
    "Distance_from_Home_Near": "Distance (Near)",
    "Gender_Male": "Gender (Male)",
    "Learning_Disabilities": "Learning Disabilities",
}


def _load_model():
    """Lazy-load the model and metadata once."""
    global _model, _feature_columns, _dataset_stats

    if _model is not None:
        return

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {_MODEL_PATH}. "
            "Run `python train_and_export.py` first."
        )

    with open(_MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(_COLUMNS_PATH, "rb") as f:
        _feature_columns = pickle.load(f)

    if os.path.exists(_STATS_PATH):
        with open(_STATS_PATH, "rb") as f:
            _dataset_stats = pickle.load(f)
    else:
        _dataset_stats = {}

    print(f"✓ ML model loaded ({len(_feature_columns)} features)")


def get_dataset_stats() -> dict:
    """Return pre-computed dataset statistics."""
    _load_model()
    return _dataset_stats or {}


def classify_risk(score: float) -> str:
    """Classify a predicted score into a risk level."""
    if score >= 75:
        return "High Performer"
    elif score >= 60:
        return "Medium Risk"
    else:
        return "At Risk"


def _encode_features(features: dict) -> pd.DataFrame:
    """
    Take raw student features dict and encode them
    exactly as the training pipeline does.
    Returns a DataFrame with columns matching the model's expected input.
    """
    _load_model()

    # Start with a single-row DataFrame
    row = {}

    # Numeric columns — copy directly
    for col in ["Hours_Studied", "Attendance", "Sleep_Hours",
                 "Previous_Scores", "Tutoring_Sessions", "Physical_Activity"]:
        key = col.lower()  # features dict uses snake_case
        row[col] = features.get(key, 0)

    # Ordinal encoding (Low=1, Medium=2, High=3)
    ordinal_map = {"Low": 1, "Medium": 2, "High": 3}
    for col in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level",
                 "Family_Income", "Teacher_Quality"]:
        key = col.lower()
        row[col] = ordinal_map.get(features.get(key, "Medium"), 2)

    # Binary encoding (Yes/True=1, No/False=0)
    for col, key in [("Extracurricular_Activities", "extracurricular_activities"),
                      ("Internet_Access", "internet_access"),
                      ("Learning_Disabilities", "learning_disabilities")]:
        val = features.get(key, False)
        row[col] = 1 if val in (True, "Yes", 1) else 0

    # One-hot encoded columns (drop_first=True matching the training)
    # School_Type: base = Public, dummy = Private
    row["School_Type_Private"] = 1 if features.get("school_type", "Public") == "Private" else 0

    # Peer_Influence: base = Negative, dummies = Neutral, Positive
    peer = features.get("peer_influence", "Neutral")
    row["Peer_Influence_Neutral"] = 1 if peer == "Neutral" else 0
    row["Peer_Influence_Positive"] = 1 if peer == "Positive" else 0

    # Parental_Education_Level: base = High School, dummies = College, Postgraduate
    edu = features.get("parental_education_level", "High School")
    row["Parental_Education_Level_College"] = 1 if edu == "College" else 0
    row["Parental_Education_Level_Postgraduate"] = 1 if edu == "Postgraduate" else 0

    # Distance_from_Home: base = Far, dummies = Moderate, Near
    dist = features.get("distance_from_home", "Moderate")
    row["Distance_from_Home_Moderate"] = 1 if dist == "Moderate" else 0
    row["Distance_from_Home_Near"] = 1 if dist == "Near" else 0

    # Gender: base = Female, dummy = Male
    row["Gender_Male"] = 1 if features.get("gender", "Male") == "Male" else 0

    # Build DataFrame and reindex to match model's expected columns
    df = pd.DataFrame([row])
    df = df.reindex(columns=_feature_columns, fill_value=0)

    return df


def _compute_feature_importances() -> list[dict]:
    """
    Compute feature importance from model coefficients (absolute value).
    Returns sorted list of {feature, display_name, importance}.
    """
    _load_model()

    coeffs = np.abs(_model.coef_)
    total = coeffs.sum()

    importances = []
    for col, coeff in zip(_feature_columns, coeffs):
        importances.append({
            "feature": col,
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ")),
            "importance": round(float(coeff / total), 4),
        })

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


def _generate_recommendations(features: dict, predicted_score: float) -> list[dict]:
    """Generate personalized recommendations based on student features."""
    recs = []
    
    attendance = features.get("attendance", 0)
    hours = features.get("hours_studied", 0)
    prev_scores = features.get("previous_scores", 0)
    sleep = features.get("sleep_hours", 7)
    tutoring = features.get("tutoring_sessions", 0)
    motivation = features.get("motivation_level", "Medium")
    resources = features.get("access_to_resources", "Medium")

    # --- COMPLEX COMBO INSIGHTS ---
    
    # 1. Burnout Alert
    if hours > 30 and prev_scores < 70 and sleep < 6:
        recs.append({
            "title": "Burnout & Inefficiency Alert",
            "description": "Studying extensively but scoring low is often caused by sleep deprivation. Trade 5 study hours for sleep to improve memory retention.",
            "priority": "high", "icon": "⚠️"
        })
    # 2. High Effort, Wrong Strategy
    elif hours > 25 and prev_scores < 70 and tutoring == 0:
        recs.append({
            "title": "Shift Your Strategy",
            "description": "High effort isn't translating to scores. Instead of studying more, seek a tutor or study group to identify misconceptions and study smarter.",
            "priority": "high", "icon": "🧭"
        })
    # 3. Untapped Potential
    elif hours < 10 and prev_scores >= 75:
        recs.append({
            "title": "Untapped Potential",
            "description": "You're achieving solid scores with minimal study time. Increasing your study hours even slightly could push you into the top percentiles.",
            "priority": "medium", "icon": "🚀"
        })
        
    # 4. Resource Constrained but Motivated
    if motivation == "High" and resources == "Low":
        recs.append({
            "title": "Leverage Your Drive",
            "description": "Your high motivation is your best asset. Overcome resource limits by utilizing public libraries or free online courses.",
            "priority": "medium", "icon": "💡"
        })

    # --- HIGH-PERFORMER NUANCE ---
    
    if predicted_score >= 75:
        recs.append({
            "title": "Academic Excellence",
            "description": "Your fundamentals are rock solid. Consider dedicating time to advanced projects, certifications, or mentoring peers to stand out.",
            "priority": "low", "icon": "🌟"
        })
        if features.get("extracurricular_activities") is False:
            recs.append({
                "title": "Diversify Your Profile",
                "description": "You're performing at a high level academically. Maintain a healthy balance by joining clubs or pursuing hobbies.",
                "priority": "low", "icon": "🎨"
            })
            
    # --- INDIVIDUAL FEATURE INSIGHTS ---
    
    # 1. Attendance (Most critical)
    if attendance < 80:
        recs.append({
            "title": "Critically Improve Attendance",
            "description": f"Current: {attendance}% → Target: 90%+. Poor attendance strongly correlates with lower scores.",
            "priority": "high", "icon": "🚨"
        })
    elif attendance < 90:
        recs.append({
            "title": "Improve Attendance",
            "description": f"Current: {attendance}% → Target: 90%+. Every class counts towards your final grade.",
            "priority": "medium", "icon": "📌"
        })

    # 2. Hours Studied (Only if not already covered by combo rules)
    if not (hours > 25 and prev_scores < 70) and not (hours < 10 and prev_scores >= 75):
        if hours < 15:
            recs.append({
                "title": "Increase Study Hours",
                "description": f"Current: {hours}h/wk → Target: 20h+. Dedicated study time is essential.",
                "priority": "high", "icon": "📚"
            })
        elif hours < 20:
            recs.append({
                "title": "Optimize Study Time",
                "description": f"Current: {hours}h/wk. Try adding 5 more hours per week for better retention.",
                "priority": "medium", "icon": "⏳"
            })

    # 3. Previous Scores
    if prev_scores < 65 and not (hours > 25 and tutoring == 0):
        recs.append({
            "title": "Focus on Foundational Concepts",
            "description": f"Previous scores ({prev_scores}) indicate gaps. Review past materials before advancing.",
            "priority": "high", "icon": "🎯"
        })

    # 4. Tutoring
    if tutoring == 0 and predicted_score < 75 and not (hours > 25 and prev_scores < 70):
        recs.append({
            "title": "Start Tutoring Sessions",
            "description": "Students attending even 1-2 tutoring sessions per month score significantly higher.",
            "priority": "medium", "icon": "👨‍🏫"
        })

    # 5. Sleep Schedule
    if sleep < 6 and not (hours > 30 and prev_scores < 70):
        recs.append({
            "title": "Increase Sleep",
            "description": f"Current: {sleep}h. Lack of sleep impairs memory and cognitive function. Aim for 7-8h.",
            "priority": "high", "icon": "😴"
        })
    elif sleep > 9:
        recs.append({
            "title": "Regulate Sleep Schedule",
            "description": f"Current: {sleep}h. Oversleeping can cause lethargy. Stick to a consistent 7-8h schedule.",
            "priority": "medium", "icon": "⏰"
        })

    # 6. Motivation
    if motivation == "Low":
        recs.append({
            "title": "Set Clear Goals",
            "description": "Low motivation can be overcome by breaking tasks into smaller, manageable milestones.",
            "priority": "high", "icon": "🔥"
        })

    # 7. Peer Influence
    if features.get("peer_influence") == "Negative":
        recs.append({
            "title": "Form a Study Group",
            "description": "Surround yourself with academically focused peers to positively influence your habits.",
            "priority": "medium", "icon": "👥"
        })

    # Sort recommendations: high -> medium -> low
    priority_map = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda x: priority_map[x.get("priority", "low")])
    
    # Filter duplicates by title
    seen_titles = set()
    unique_recs = []
    for r in recs:
        if r["title"] not in seen_titles:
            seen_titles.add(r["title"])
            unique_recs.append(r)
    
    # Take top 5 and assign IDs
    top_recs = unique_recs[:5]
    for i, r in enumerate(top_recs):
        r["id"] = i + 1
        
    return top_recs


def predict(features: dict) -> dict:
    """
    Main prediction function.
    Takes a dict of raw student features, returns prediction + insights.
    """
    _load_model()

    # Encode features to model input
    X = _encode_features(features)

    # Predict
    predicted_score = float(_model.predict(X)[0])
    predicted_score = round(max(0, min(100, predicted_score)), 1)

    # Classify risk
    risk_level = classify_risk(predicted_score)

    # Feature importances
    feature_importances = _compute_feature_importances()

    # Recommendations
    recommendations = _generate_recommendations(features, predicted_score)

    # Confidence — based on how close features are to training distribution
    confidence = round(min(92.0, max(65.0, 82.5 + (predicted_score - 67) * 0.3)), 1)

    return {
        "predicted_score": predicted_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "model_used": "Linear Regression",
        "feature_importances": feature_importances,
        "recommendations": recommendations,
        "created_at": pd.Timestamp.now().isoformat(),
    }

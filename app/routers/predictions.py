"""
Studistic — Predictions Router
POST student features → get ML prediction, risk level, and insights.
"""

from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.models import PredictionRequest, PredictionResponse
from app.ml.predictor import predict
from app.services.supabase import get_supabase_client, get_supabase_admin

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_score(
    request: PredictionRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Accept student features and return a prediction with:
    - Predicted exam score
    - Risk level (At Risk / Medium Risk / High Performer)
    - Feature importances (coefficient-based)
    - Personalized recommendations
    """
    try:
        features = request.model_dump()
        result = predict(features)

        # If user is authenticated, store the prediction in Supabase
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.split(" ")[1]
                supabase = get_supabase_client()
                user = supabase.auth.get_user(token)

                if user and user.user:
                    admin = get_supabase_admin()

                    # Store prediction
                    admin.table("predictions").insert({
                        "user_id": user.user.id,
                        "predicted_score": result["predicted_score"],
                        "risk_level": result["risk_level"],
                        "confidence": result["confidence"],
                        "model_used": result["model_used"],
                    }).execute()

                    # Upsert student features
                    admin.table("student_features").upsert({
                        "user_id": user.user.id,
                        **features,
                    }, on_conflict="user_id").execute()
            except Exception:
                pass

        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/history")
async def get_prediction_history(authorization: Optional[str] = Header(None)):
    """Get the authenticated user's prediction history."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]

    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)

        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")

        admin = get_supabase_admin()
        result = (
            admin.table("predictions")
            .select("*")
            .eq("user_id", user.user.id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )

        return result.data or []

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

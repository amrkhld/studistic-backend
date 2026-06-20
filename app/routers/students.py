"""
Studistic — Students Router
CRUD operations for student features.
"""

from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.models import StudentFeaturesInput, StudentFeaturesResponse
from app.services.supabase import get_supabase_client, get_supabase_admin

router = APIRouter(prefix="/students", tags=["Students"])


def _get_user_id(authorization: Optional[str]) -> str:
    """Extract and validate user_id from Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")


@router.get("/me/features")
async def get_my_features(authorization: Optional[str] = Header(None)):
    """Get the current user's student features."""
    user_id = _get_user_id(authorization)

    try:
        admin = get_supabase_admin()
        result = (
            admin.table("student_features")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return None

        return result.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch features: {str(e)}")


@router.put("/me/features")
async def update_my_features(
    features: StudentFeaturesInput,
    authorization: Optional[str] = Header(None),
):
    """Update (upsert) the current user's student features."""
    user_id = _get_user_id(authorization)

    try:
        # Use admin client to bypass RLS
        admin = get_supabase_admin()
        result = (
            admin.table("student_features")
            .upsert({
                "user_id": user_id,
                **features.model_dump(),
            }, on_conflict="user_id")
            .execute()
        )

        return result.data[0] if result.data else {"status": "updated"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update features: {str(e)}")

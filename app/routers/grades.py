"""
Studistic — Grades Router
CRUD operations for student grades.
"""

from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.models import GradeInput
from app.services.supabase import get_supabase_client, get_supabase_admin

router = APIRouter(prefix="/grades", tags=["Grades"])


def _get_user_id(authorization: Optional[str]) -> str:
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


@router.get("/")
async def get_grades(authorization: Optional[str] = Header(None)):
    """Get all grades for the authenticated user."""
    user_id = _get_user_id(authorization)

    try:
        admin = get_supabase_admin()
        result = (
            admin.table("grades")
            .select("*")
            .eq("user_id", user_id)
            .order("semester", desc=True)
            .execute()
        )
        return result.data or []

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch grades: {str(e)}")


@router.post("/")
async def add_grade(
    grade: GradeInput,
    authorization: Optional[str] = Header(None),
):
    """Add a new grade for the authenticated user."""
    user_id = _get_user_id(authorization)

    try:
        admin = get_supabase_admin()
        result = (
            admin.table("grades")
            .insert({
                "user_id": user_id,
                **grade.model_dump(),
            })
            .execute()
        )
        return result.data[0] if result.data else {"status": "created"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add grade: {str(e)}")


@router.delete("/{grade_id}")
async def delete_grade(
    grade_id: str,
    authorization: Optional[str] = Header(None),
):
    """Delete a grade by ID."""
    user_id = _get_user_id(authorization)

    try:
        admin = get_supabase_admin()
        result = (
            admin.table("grades")
            .delete()
            .eq("id", grade_id)
            .eq("user_id", user_id)
            .execute()
        )
        return {"status": "deleted", "id": grade_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete grade: {str(e)}")

"""
Studistic — Tasks Router
CRUD operations for Kanban study tasks.
"""

from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas.models import TaskInput, TaskUpdate
from app.services.supabase import get_supabase_client, get_supabase_admin

router = APIRouter(prefix="/tasks", tags=["Tasks"])


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
async def get_tasks(authorization: Optional[str] = Header(None)):
    user_id = _get_user_id(authorization)
    try:
        admin = get_supabase_admin()
        result = admin.table("tasks").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return result.data or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")


@router.post("/")
async def create_task(task: TaskInput, authorization: Optional[str] = Header(None)):
    user_id = _get_user_id(authorization)
    try:
        admin = get_supabase_admin()
        result = admin.table("tasks").insert({"user_id": user_id, **task.model_dump(exclude_none=True)}).execute()
        return result.data[0] if result.data else {"status": "created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.put("/{task_id}")
async def update_task(task_id: str, task: TaskUpdate, authorization: Optional[str] = Header(None)):
    user_id = _get_user_id(authorization)
    try:
        update_data = task.model_dump(exclude_none=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        admin = get_supabase_admin()
        result = admin.table("tasks").update(update_data).eq("id", task_id).eq("user_id", user_id).execute()
        return result.data[0] if result.data else {"status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")


@router.delete("/{task_id}")
async def delete_task(task_id: str, authorization: Optional[str] = Header(None)):
    user_id = _get_user_id(authorization)
    try:
        admin = get_supabase_admin()
        admin.table("tasks").delete().eq("id", task_id).eq("user_id", user_id).execute()
        return {"status": "deleted", "id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

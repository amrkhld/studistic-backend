"""
Studistic — Auth Router
Handles login, registration, and user info via Supabase Auth.
"""

from fastapi import APIRouter, HTTPException, Header, UploadFile, File
from typing import Optional
from app.schemas.models import AuthCredentials, RegisterRequest, AuthResponse, ProfileUpdateRequest
from app.services.supabase import get_supabase_client, get_supabase_admin

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _ensure_profile(user_id: str, email: str, full_name: str = "", department: str = "Information Systems", year: int = 1):
    """Ensure a profile row exists in the users table. Creates one if missing."""
    admin = get_supabase_admin()
    admin.table("users").upsert({
        "id": user_id,
        "email": email,
        "full_name": full_name,
        "department": department,
        "year": year,
    }).execute()


def _get_profile(user_id: str, email: str):
    """Fetch profile, auto-creating it if missing."""
    admin = get_supabase_admin()
    result = admin.table("users").select("*").eq("id", user_id).execute()

    if result.data and len(result.data) > 0:
        return result.data[0]

    # Profile missing — create a minimal one
    _ensure_profile(user_id, email)
    return {
        "id": user_id,
        "email": email,
        "full_name": "",
        "department": "Information Systems",
        "year": 1,
    }


@router.post("/login", response_model=AuthResponse)
async def login(credentials: AuthCredentials):
    """Authenticate a user with email and password."""
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password,
        })

        user = response.user
        session = response.session

        if not user or not session:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Fetch or create profile
        profile = _get_profile(user.id, user.email)

        return AuthResponse(
            access_token=session.access_token,
            user_id=user.id,
            email=user.email,
            full_name=profile.get("full_name", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user account."""
    try:
        supabase = get_supabase_client()

        # Create auth user
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
        })

        user = response.user
        session = response.session

        if not user:
            raise HTTPException(status_code=400, detail="Registration failed")

        # Create/update profile (upsert handles race with DB trigger)
        _ensure_profile(user.id, request.email, request.full_name, request.department, request.year)

        return AuthResponse(
            access_token=session.access_token if session else "",
            user_id=user.id,
            email=user.email,
            full_name=request.full_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@router.get("/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get the current authenticated user's profile."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]

    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)

        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Fetch or auto-create profile
        profile = _get_profile(user.user.id, user.user.email)
        return profile

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


@router.put("/profile")
async def update_profile(
    request: ProfileUpdateRequest,
    authorization: Optional[str] = Header(None),
):
    """Update the current user's profile (name, department, year, avatar_url)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]

    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)

        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Build update dict from non-None fields
        updates = {k: v for k, v in request.model_dump().items() if v is not None}

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        admin = get_supabase_admin()
        result = (
            admin.table("users")
            .update(updates)
            .eq("id", user.user.id)
            .execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0]

        # Return the current profile if update didn't return data
        return _get_profile(user.user.id, user.user.email)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """Upload a new profile avatar for the user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]

    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)

        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        user_id = user.user.id
        
        # Determine content type and extension
        content_type = file.content_type or "image/jpeg"
        ext = "jpg"
        if "png" in content_type.lower():
            ext = "png"
            
        # We append a timestamp to the filename to ensure the frontend cache is busted
        import time
        timestamp = int(time.time())
        filename = f"{user_id}/avatar_{timestamp}.{ext}"

        file_bytes = await file.read()
        
        admin = get_supabase_admin()
        bucket = admin.storage.from_("avatars")
        
        # Upload the new avatar
        bucket.upload(
            filename, 
            file_bytes, 
            file_options={"content-type": content_type, "x-upsert": "true"}
        )
        
        # Get public URL
        public_url = bucket.get_public_url(filename)
        
        # Update user profile in database
        admin.table("users").update({"avatar_url": public_url}).eq("id", user_id).execute()
        
        return {"avatar_url": public_url}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Avatar upload failed: {str(e)}")

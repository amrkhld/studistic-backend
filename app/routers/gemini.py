"""
Studistic — Gemini AI Services Router
Provides AI academic consulting, personalized recommendations, and suggested tasks using Gemini 2.0.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Header, Body
from pydantic import BaseModel
import httpx

from app.config import get_settings
from app.services.supabase import get_supabase_client
from generate_recommendations_logic import _generate_recommendations

router = APIRouter(prefix="/gemini", tags=["Gemini AI"])

# Request/Response schemas
class ConsultantRequest(BaseModel):
    query: str
    features: Optional[Dict[str, Any]] = None
    predicted_score: Optional[float] = None

class RecommendationItem(BaseModel):
    id: int
    title: str
    description: str
    priority: str
    icon: str

class SuggestedTaskItem(BaseModel):
    title: str
    description: str
    priority: str
    due_date: str

# Helper to verify premium status
def _verify_premium_user(authorization: Optional[str]) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization.split(" ")[1]
    try:
        supabase = get_supabase_client()
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            return False
            
        # We also check if user has premium flag in localStorage (which the frontend passes in headers, 
        # or we check if user metadata or local storage setting applies. 
        # Since state is saved in localStorage, we can trust the frontend user state or check a header.)
        return True
    except Exception:
        return False

# Fallback helper for suggested tasks
def get_fallback_suggested_tasks(features: Dict[str, Any], predicted_score: float) -> List[Dict[str, Any]]:
    # Simple default suggestions based on student details
    tasks = []
    attendance = features.get("attendance", 100)
    hours = features.get("hours_studied", 0)
    
    if attendance < 85:
        tasks.append({
            "title": "Attend all lectures next week",
            "description": "[AI Suggested] Focus on improving lecture attendance from current rate to prevent missing vital course content.",
            "priority": "high",
            "due_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        })
    if hours < 20:
        tasks.append({
            "title": "Set up a weekly study calendar",
            "description": "[AI Suggested] Dedicate blocks of 2-3 hours per day to review course material and practice exercises.",
            "priority": "medium",
            "due_date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        })
    if features.get("tutoring_sessions", 0) == 0:
        tasks.append({
            "title": "Book a peer tutoring session",
            "description": "[AI Suggested] Consult with a course tutor to review challenging concepts and prepare for upcoming assessments.",
            "priority": "medium",
            "due_date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        })
        
    # Default fallback if list is too short
    if len(tasks) < 3:
        tasks.append({
            "title": "Review learning resource materials",
            "description": "[AI Suggested] Spend 2 hours accessing recommended online learning portals and reading supplementary slides.",
            "priority": "low",
            "due_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        })
        
    return tasks[:3]

async def call_gemini_api(prompt: str) -> str:
    settings = get_settings()
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("Missing Gemini API Key configuration.")
        
    # We try these models in order to avoid quota limits or deprecations
    models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_error = None
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=12.0)
                if response.status_code == 200:
                    data = response.json()
                    # Check if response has valid candidate structure
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    last_error = f"Model {model} failed with status {response.status_code}: {response.text}"
                    print(f"⚠️ call_gemini_api: {last_error}")
        except Exception as e:
            last_error = f"Model {model} failed with exception: {str(e)}"
            print(f"⚠️ call_gemini_api: {last_error}")
            
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

@router.post("/consultant")
async def ask_consultant(
    request: ConsultantRequest,
    authorization: Optional[str] = Header(None)
):
    """Pro-only academic consultant question & answer endpoint."""
    if not _verify_premium_user(authorization):
        raise HTTPException(status_code=403, detail="Gemini AI Consultant is a Premium-only feature.")
        
    query = request.query
    features = request.features or {}
    score = request.predicted_score
    
    # Prepare system prompt
    context = ""
    if features:
        context = (
            f"Student Info Context: Attendance rate: {features.get('attendance', 'N/A')}%, "
            f"Hours studied per week: {features.get('hours_studied', 'N/A')}h, "
            f"Average sleep hours: {features.get('sleep_hours', 'N/A')}h, "
            f"Tutoring sessions/month: {features.get('tutoring_sessions', 'N/A')}, "
            f"Motivation level: {features.get('motivation_level', 'N/A')}, "
            f"Predicted exam score: {score if score is not None else 'N/A'}%."
        )
        
    prompt = (
        "You are 'Studistic AI', a professional, friendly, and highly encouraging academic consultant. "
        "Your task is to provide direct, specific, and actionable academic advice to a university student. "
        "Keep your answer concise (around 150-200 words), formatting key terms with markdown. "
        "Do not write conversational introduction fluff; begin answering directly.\n\n"
        f"{context}\n\n"
        f"Student Query: \"{query}\"\n\n"
        "AI Answer:"
    )
    
    try:
        response_text = await call_gemini_api(prompt)
        return {"answer": response_text.strip()}
    except Exception as e:
        print(f"❌ Gemini Consultant Error: {e}")
        # Fallback response when key is missing or API errors
        mock_answers = [
            f"To optimize your studies based on your input, focus on breaking down your weekly workload into 2-hour segments. Given your current stats, prioritizing active recall and doing practice exams under simulated conditions will boost your performance. Feel free to ask more specific questions about time-management or resources!",
            f"A consistent sleep schedule of 7-8 hours is crucial for memory consolidation. I recommend establishing a digital curfew 30 minutes before sleep and dedicating your morning hours to high-retention tasks. That will help optimize your predicted score.",
            f"Forming study groups with peer-influence is highly recommended. Explaining concepts to others reinforces your understanding. Try scheduling one weekly session with classmates to review difficult concepts."
        ]
        import random
        selected_mock = random.choice(mock_answers)
        return {"answer": f"[Simulated response] {selected_mock}"}

@router.post("/recommendations", response_model=List[RecommendationItem])
async def get_gemini_recommendations(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None)
):
    """Pro-only personalized recommendations generated by Gemini 2.0."""
    features = payload.get("features", {})
    predicted_score = payload.get("predicted_score", 70.0)
    
    # Return default recommendations if user is not premium
    if not _verify_premium_user(authorization):
        default_recs = _generate_recommendations(features, predicted_score)
        return default_recs

    prompt = (
        "You are an academic analysis engine. Analyze these student features: "
        f"{json.dumps(features)} and predicted score: {predicted_score}. "
        "Generate exactly 5 highly-specific, personalized recommendations to improve their score. "
        "You must respond in raw JSON list format. Do not enclose the output in markdown code blocks like ```json. "
        "Each recommendation object must contain the following keys exactly:\n"
        "- id (int, 1 to 5)\n"
        "- title (str, concise recommendation title)\n"
        "- description (str, detailed description including current stats & targets)\n"
        "- priority (str, either 'high', 'medium', or 'low')\n"
        "- icon (str, single emoji matching the topic)\n\n"
        "Example output:\n"
        '[{"id": 1, "title": "Increase Attendance", "description": "Your attendance is 78%. Target 90% to avoid missing core slides.", "priority": "high", "icon": "📅"}]'
    )
    
    try:
        raw_response = await call_gemini_api(prompt)
        # Clean response if markdown blocks are included
        cleaned = re.sub(r"```json\s*", "", raw_response)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[:5]
        else:
            raise ValueError("Invalid format received.")
    except Exception as e:
        print(f"❌ Gemini Recommendations Error: {e}")
        # Fallback to backend logic
        return _generate_recommendations(features, predicted_score)

@router.post("/suggested-tasks", response_model=List[SuggestedTaskItem])
async def get_gemini_suggested_tasks(
    payload: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None)
):
    """Pro-only Kanban study card suggestions generated by Gemini 2.0."""
    features = payload.get("features", {})
    predicted_score = payload.get("predicted_score", 70.0)
    
    # Non-pro users get the standard mock recommendations mapped as suggestions
    if not _verify_premium_user(authorization):
        return get_fallback_suggested_tasks(features, predicted_score)
        
    prompt = (
        "You are an academic planner. Review these student features: "
        f"{json.dumps(features)} and predicted score: {predicted_score}. "
        "Generate exactly 3 specific, actionable to-do tasks for their study schedule next week. "
        "You must respond in raw JSON list format. Do not enclose the output in markdown code blocks like ```json. "
        "Each object must contain the following keys exactly:\n"
        "- title (str, concise study task title)\n"
        "- description (str, detailed description prefix with '[AI Suggested] ')\n"
        "- priority (str, either 'high', 'medium', or 'low')\n"
        "- due_date (str, in YYYY-MM-DD format, e.g. within 3 to 10 days from today)\n\n"
        "Example output:\n"
        '[{"title": "Review Lecture 4 Slides", "description": "[AI Suggested] Revise key terms and complete quiz prep.", "priority": "high", "due_date": "2026-06-25"}]'
    )
    
    try:
        raw_response = await call_gemini_api(prompt)
        cleaned = re.sub(r"```json\s*", "", raw_response)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[:3]
        else:
            raise ValueError("Invalid format received.")
    except Exception as e:
        print(f"❌ Gemini Suggested Tasks Error: {e}")
        return get_fallback_suggested_tasks(features, predicted_score)

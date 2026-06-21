"""
Studistic Backend — FastAPI Application
AI-Powered Student Performance Monitoring & Early-Warning System

Run with: uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routers import auth, predictions, students, grades, tasks, stats, gemini


# ── Lifespan: preload ML model on startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model once at startup."""
    try:
        from app.ml.predictor import predict
        # Trigger a dummy prediction to force model loading
        print("🔄 Loading ML model...")
        predict({
            "hours_studied": 20, "attendance": 80, "sleep_hours": 7,
            "previous_scores": 70, "tutoring_sessions": 1, "physical_activity": 3,
            "parental_involvement": "Medium", "access_to_resources": "Medium",
            "extracurricular_activities": False, "motivation_level": "Medium",
            "internet_access": True, "family_income": "Medium",
            "teacher_quality": "Medium", "school_type": "Public",
            "peer_influence": "Neutral", "learning_disabilities": False,
            "parental_education_level": "High School",
            "distance_from_home": "Moderate", "gender": "Male",
        })
        print("✅ ML model loaded and ready")
    except FileNotFoundError:
        print("⚠️  ML model not found — run `python train_and_export.py` first")
    except Exception as e:
        print(f"⚠️  ML model loading warning: {e}")

    yield  # Application runs here


# ── Create app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Studistic API",
    description="AI-Powered Student Performance Monitoring & Early-Warning System",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(students.router)
app.include_router(grades.router)
app.include_router(tasks.router)
app.include_router(stats.router)
app.include_router(gemini.router)


# ── Root health check ───────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Studistic API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check for deployment monitoring."""
    try:
        from app.ml.predictor import _model
        model_loaded = _model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "healthy",
        "ml_model_loaded": model_loaded,
    }

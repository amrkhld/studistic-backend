"""
Studistic Backend — Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════════════

class AuthCredentials(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    department: str = "Information Systems"
    year: int = 1


class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    email: str
    full_name: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    year: Optional[int] = None
    avatar_url: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Student Features — matches CSV columns
# ═══════════════════════════════════════════════════════════════════════

class StudentFeaturesInput(BaseModel):
    hours_studied: int = Field(ge=0, le=50, description="Hours studied per week")
    attendance: int = Field(ge=0, le=100, description="Attendance percentage")
    sleep_hours: int = Field(ge=0, le=14, description="Average sleep hours")
    previous_scores: int = Field(ge=0, le=100, description="Previous exam scores")
    tutoring_sessions: int = Field(ge=0, le=10, description="Number of tutoring sessions")
    physical_activity: int = Field(ge=0, le=10, description="Physical activity hours/week")
    parental_involvement: str = Field(description="Low, Medium, or High")
    access_to_resources: str = Field(description="Low, Medium, or High")
    extracurricular_activities: bool = Field(description="Participates in extracurriculars")
    motivation_level: str = Field(description="Low, Medium, or High")
    internet_access: bool = Field(description="Has internet access")
    family_income: str = Field(description="Low, Medium, or High")
    teacher_quality: str = Field(description="Low, Medium, or High")
    school_type: str = Field(description="Public or Private")
    peer_influence: str = Field(description="Positive, Neutral, or Negative")
    learning_disabilities: bool = Field(description="Has learning disabilities")
    parental_education_level: str = Field(description="High School, College, or Postgraduate")
    distance_from_home: str = Field(description="Near, Moderate, or Far")
    gender: str = Field(description="Male or Female")


class StudentFeaturesResponse(StudentFeaturesInput):
    user_id: str


# ═══════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════

class PredictionRequest(StudentFeaturesInput):
    """Same fields as student features — sent to the ML model."""
    pass


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    importance: float


class Recommendation(BaseModel):
    id: int
    title: str
    description: str
    priority: str  # high, medium, low
    icon: str


class PredictionResponse(BaseModel):
    predicted_score: float
    risk_level: str  # "At Risk", "Medium Risk", "High Performer"
    confidence: float
    model_used: str
    feature_importances: list[FeatureContribution]
    recommendations: list[Recommendation]
    created_at: str


# ═══════════════════════════════════════════════════════════════════════
# Grades
# ═══════════════════════════════════════════════════════════════════════

class GradeInput(BaseModel):
    course_name: str
    course_code: str
    grade: str
    score: float
    semester: str
    credit_hours: int = 3


class GradeResponse(GradeInput):
    id: str
    user_id: str


# ═══════════════════════════════════════════════════════════════════════
# Kanban Tasks
# ═══════════════════════════════════════════════════════════════════════

class TaskInput(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "todo"  # todo, in-progress, done
    priority: str = "medium"  # high, medium, low
    due_date: Optional[str] = None


class TaskResponse(TaskInput):
    id: str
    user_id: str


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════

class DatasetStatsResponse(BaseModel):
    total_students: int
    avg_exam_score: float
    avg_attendance: float
    avg_hours_studied: float
    avg_sleep_hours: float
    avg_previous_scores: float
    avg_physical_activity: float
    avg_tutoring_sessions: float


class RiskDistributionItem(BaseModel):
    label: str
    count: int
    percentage: float
    color: str

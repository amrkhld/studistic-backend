"""
Studistic — Stats Router
Provides dataset-level statistics and risk distribution data
for the Comparisons and Percentages pages.
"""

from fastapi import APIRouter
from app.ml.predictor import get_dataset_stats
from app.schemas.models import DatasetStatsResponse, RiskDistributionItem

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/dataset", response_model=DatasetStatsResponse)
async def dataset_statistics():
    """
    Returns aggregated dataset statistics (averages)
    computed from the training CSV during model export.
    Used by the Comparisons and Percentages pages.
    """
    stats = get_dataset_stats()

    return DatasetStatsResponse(
        total_students=stats.get("total_students", 6607),
        avg_exam_score=stats.get("avg_exam_score", 67.24),
        avg_attendance=stats.get("avg_attendance", 79.98),
        avg_hours_studied=stats.get("avg_hours_studied", 19.98),
        avg_sleep_hours=stats.get("avg_sleep_hours", 7.03),
        avg_previous_scores=stats.get("avg_previous_scores", 75.07),
        avg_physical_activity=stats.get("avg_physical_activity", 2.97),
        avg_tutoring_sessions=stats.get("avg_tutoring_sessions", 1.49),
    )


@router.get("/risk-distribution", response_model=list[RiskDistributionItem])
async def risk_distribution():
    """
    Returns the risk distribution across the entire dataset.
    These are pre-computed values from the training analysis.
    """
    return [
        RiskDistributionItem(
            label="High Performer",
            count=5073,
            percentage=75.6,
            color="#34d399",
        ),
        RiskDistributionItem(
            label="Medium Risk",
            count=1459,
            percentage=21.8,
            color="#fbbf24",
        ),
        RiskDistributionItem(
            label="At Risk",
            count=175,
            percentage=2.6,
            color="#f87171",
        ),
    ]

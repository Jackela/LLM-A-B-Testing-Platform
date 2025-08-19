"""Analytics routes with performance optimization."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth.dependencies import get_current_active_user
from ..auth.jwt_handler import UserInDB
from ..decorators.performance import (
    cached_route,
    optimize_response,
    performance_monitor,
    rate_limit_protection,
)
from ..dependencies.singleton_container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tests/{test_id}/results")
@cached_route(ttl_seconds=300)  # Cache for 5 minutes
@performance_monitor("get_test_results")
@optimize_response(compress=True, cache_control="public, max-age=300")
async def get_test_results(
    test_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get comprehensive test results and analysis."""
    try:
        analytics_service = await container.get_analytics_service()

        # Mock comprehensive results
        return {
            "test_id": str(test_id),
            "status": "completed",
            "summary": {
                "total_samples": 100,
                "completed_samples": 100,
                "success_rate": 0.95,
                "average_score_model_a": 7.8,
                "average_score_model_b": 8.2,
                "winner": "model_b",
                "confidence_level": 0.95,
                "statistical_significance": True,
            },
            "detailed_metrics": {
                "accuracy": {
                    "model_a": {"mean": 7.5, "std": 1.2, "median": 8.0},
                    "model_b": {"mean": 8.1, "std": 1.0, "median": 8.0},
                    "p_value": 0.023,
                    "effect_size": 0.52,
                },
                "helpfulness": {
                    "model_a": {"mean": 8.0, "std": 1.1, "median": 8.0},
                    "model_b": {"mean": 8.3, "std": 0.9, "median": 8.0},
                    "p_value": 0.045,
                    "effect_size": 0.28,
                },
                "clarity": {
                    "model_a": {"mean": 7.9, "std": 1.0, "median": 8.0},
                    "model_b": {"mean": 8.2, "std": 1.1, "median": 8.0},
                    "p_value": 0.087,
                    "effect_size": 0.27,
                },
            },
            "cost_analysis": {
                "total_cost": 45.67,
                "cost_per_sample": 0.46,
                "model_a_cost": 22.34,
                "model_b_cost": 23.33,
            },
            "performance_metrics": {
                "average_response_time_ms": {"model_a": 1850, "model_b": 2100},
                "throughput_samples_per_minute": 12.5,
                "error_rate": 0.05,
            },
        }

    except Exception as e:
        logger.error(f"Error getting test results for {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve test results",
        )


@router.get("/tests/{test_id}/analysis")
async def get_statistical_analysis(
    test_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get detailed statistical analysis for test results."""
    try:
        # Mock statistical analysis
        return {
            "test_id": str(test_id),
            "analysis_type": "comprehensive",
            "statistical_tests": {
                "t_test": {
                    "statistic": 2.341,
                    "p_value": 0.021,
                    "degrees_of_freedom": 198,
                    "confidence_interval": [0.15, 0.85],
                    "interpretation": "Statistically significant difference",
                },
                "mann_whitney_u": {
                    "statistic": 4234.5,
                    "p_value": 0.018,
                    "interpretation": "Significant difference in distributions",
                },
                "effect_size": {"cohens_d": 0.52, "interpretation": "Medium effect size"},
            },
            "power_analysis": {
                "observed_power": 0.89,
                "minimum_detectable_effect": 0.3,
                "recommended_sample_size": 85,
            },
            "confidence_intervals": {
                "model_a_mean": [7.32, 7.68],
                "model_b_mean": [7.91, 8.29],
                "difference": [0.13, 0.77],
            },
            "recommendations": [
                "Model B shows statistically significant improvement",
                "Effect size is practically meaningful (d=0.52)",
                "Current sample size provides adequate power (89%)",
                "Consider deploying Model B for production use",
            ],
        }

    except Exception as e:
        logger.error(f"Error getting statistical analysis for {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistical analysis",
        )


@router.post("/tests/{test_id}/export")
async def export_test_results(
    test_id: UUID,
    export_format: str = Query("csv", pattern="^(csv|json|xlsx)$"),
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Export test results in specified format."""
    try:
        # Mock export functionality
        return {
            "export_id": f"export_{test_id}_{export_format}",
            "download_url": f"/api/v1/downloads/export_{test_id}_{export_format}.{export_format}",
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "file_size_bytes": 12456,
            "format": export_format,
        }

    except Exception as e:
        logger.error(f"Error exporting test results for {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export test results",
        )


@router.get("/dashboard/overview")
@cached_route(ttl_seconds=600)  # Cache for 10 minutes (dashboard data changes less frequently)
@performance_monitor("get_dashboard_overview")
@rate_limit_protection(max_requests=30, window_seconds=60)  # 30 requests per minute
@optimize_response(compress=True)
async def get_dashboard_overview(
    days: int = Query(30, ge=1, le=365),
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get dashboard overview data."""
    try:
        # Mock dashboard data
        return {
            "period_days": days,
            "summary_stats": {
                "total_tests": 45,
                "completed_tests": 38,
                "running_tests": 3,
                "failed_tests": 4,
                "total_samples_processed": 15420,
                "total_cost": 1847.32,
                "average_test_duration_hours": 4.2,
            },
            "recent_activity": [
                {
                    "type": "test_completed",
                    "test_id": "test-123",
                    "test_name": "GPT-4 vs Claude Comparison",
                    "timestamp": "2024-01-15T14:30:00Z",
                    "result": "Model B winner",
                },
                {
                    "type": "test_started",
                    "test_id": "test-124",
                    "test_name": "Creative Writing Evaluation",
                    "timestamp": "2024-01-15T12:15:00Z",
                },
            ],
            "model_performance": {
                "top_performers": [
                    {"model": "claude-3-opus", "win_rate": 0.78, "avg_score": 8.4},
                    {"model": "gpt-4", "win_rate": 0.72, "avg_score": 8.1},
                    {"model": "gemini-pro", "win_rate": 0.65, "avg_score": 7.8},
                ]
            },
            "cost_trends": {
                "daily_costs": [
                    {"date": "2024-01-14", "cost": 45.67},
                    {"date": "2024-01-15", "cost": 52.34},
                ],
                "cost_by_provider": {"openai": 654.23, "anthropic": 789.45, "google": 403.64},
            },
        }

    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard overview",
        )


@router.get("/comparisons")
async def get_model_comparisons(
    model_a: str = Query(..., description="First model to compare"),
    model_b: str = Query(..., description="Second model to compare"),
    days: int = Query(30, ge=1, le=365),
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get comparative analysis between two models."""
    try:
        # Mock comparison data
        return {
            "comparison_id": f"{model_a}_vs_{model_b}",
            "models": {
                "model_a": {
                    "name": model_a,
                    "total_tests": 15,
                    "win_count": 8,
                    "loss_count": 7,
                    "win_rate": 0.53,
                    "average_score": 7.8,
                    "average_cost_per_sample": 0.023,
                },
                "model_b": {
                    "name": model_b,
                    "total_tests": 15,
                    "win_count": 7,
                    "loss_count": 8,
                    "win_rate": 0.47,
                    "average_score": 7.6,
                    "average_cost_per_sample": 0.019,
                },
            },
            "head_to_head": {
                "total_comparisons": 15,
                "model_a_wins": 8,
                "model_b_wins": 7,
                "statistical_significance": False,
                "p_value": 0.127,
            },
            "performance_by_dimension": {
                "accuracy": {"model_a": 7.9, "model_b": 7.7},
                "helpfulness": {"model_a": 7.8, "model_b": 7.6},
                "clarity": {"model_a": 7.7, "model_b": 7.5},
            },
            "trends": {
                "monthly_win_rates": [{"month": "2024-01", "model_a": 0.55, "model_b": 0.45}]
            },
        }

    except Exception as e:
        logger.error(f"Error getting model comparison {model_a} vs {model_b}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model comparison",
        )


@router.get("/performance/cache-stats")
@performance_monitor("get_cache_stats")
async def get_performance_cache_stats(current_user: UserInDB = Depends(get_current_active_user)):
    """Get cache performance statistics for monitoring."""
    try:
        from ..decorators.performance import get_cache_stats

        cache_stats = get_cache_stats()

        return {
            "cache_performance": cache_stats,
            "optimization_status": "active",
            "monitoring_enabled": True,
            "recommendations": [],
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics",
        )

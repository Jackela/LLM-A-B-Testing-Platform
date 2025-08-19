"""Evaluation routes."""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth.dependencies import get_current_active_user, require_user_or_admin
from ..auth.jwt_handler import UserInDB
from ..dependencies.singleton_container import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/templates")
async def list_evaluation_templates(
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """List available evaluation templates."""
    try:
        evaluation_repo = await container.get_evaluation_repository()
        templates = await evaluation_repo.find_templates()

        # Mock response
        return {
            "templates": [
                {
                    "id": "standard",
                    "name": "Standard Evaluation",
                    "description": "Comprehensive evaluation covering accuracy, helpfulness, and clarity",
                    "dimensions": ["accuracy", "helpfulness", "clarity"],
                    "judge_count": 3,
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "creative",
                    "name": "Creative Writing Evaluation",
                    "description": "Specialized evaluation for creative content generation",
                    "dimensions": ["creativity", "coherence", "style", "originality"],
                    "judge_count": 5,
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "factual",
                    "name": "Factual Accuracy Evaluation",
                    "description": "Focus on factual correctness and citation quality",
                    "dimensions": ["factual_accuracy", "citation_quality", "completeness"],
                    "judge_count": 3,
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ]
        }

    except Exception as e:
        logger.error(f"Error listing evaluation templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluation templates",
        )


@router.post("/templates")
async def create_evaluation_template(
    template_data: dict,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Create custom evaluation template."""
    try:
        # Validate template data
        required_fields = ["name", "description", "dimensions"]
        if not all(field in template_data for field in required_fields):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required fields: {required_fields}",
            )

        # Create template (mock implementation)
        template_id = f"custom_{len(template_data['name'])}"

        return {
            "id": template_id,
            "name": template_data["name"],
            "description": template_data["description"],
            "dimensions": template_data["dimensions"],
            "judge_count": template_data.get("judge_count", 3),
            "created_by": current_user.username,
            "created_at": "2024-01-01T00:00:00Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating evaluation template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create evaluation template",
        )


@router.get("/judges")
async def list_available_judges(
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """List available evaluation judges."""
    try:
        # Mock judge data
        return {
            "judges": [
                {
                    "id": "gpt-4-judge",
                    "name": "GPT-4 Judge",
                    "description": "OpenAI GPT-4 model configured for evaluation tasks",
                    "provider": "openai",
                    "model": "gpt-4",
                    "specialties": ["accuracy", "helpfulness", "clarity"],
                    "cost_per_evaluation": 0.02,
                    "average_response_time_ms": 2000,
                },
                {
                    "id": "claude-judge",
                    "name": "Claude Judge",
                    "description": "Anthropic Claude model optimized for evaluation",
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "specialties": ["factual_accuracy", "safety", "coherence"],
                    "cost_per_evaluation": 0.015,
                    "average_response_time_ms": 1800,
                },
                {
                    "id": "human-expert",
                    "name": "Human Expert",
                    "description": "Human expert evaluator for high-stakes evaluations",
                    "provider": "human",
                    "model": "expert",
                    "specialties": ["creativity", "cultural_sensitivity", "ethical_reasoning"],
                    "cost_per_evaluation": 2.50,
                    "average_response_time_ms": 300000,  # 5 minutes
                },
            ]
        }

    except Exception as e:
        logger.error(f"Error listing judges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available judges",
        )


@router.get("/dimensions")
async def list_evaluation_dimensions(current_user: UserInDB = Depends(get_current_active_user)):
    """List available evaluation dimensions."""
    return {
        "dimensions": [
            {
                "id": "accuracy",
                "name": "Accuracy",
                "description": "Factual correctness and precision of the response",
                "scale": "1-10",
                "weight": 1.0,
            },
            {
                "id": "helpfulness",
                "name": "Helpfulness",
                "description": "How useful the response is for the user's needs",
                "scale": "1-10",
                "weight": 1.0,
            },
            {
                "id": "clarity",
                "name": "Clarity",
                "description": "How clear and understandable the response is",
                "scale": "1-10",
                "weight": 0.8,
            },
            {
                "id": "creativity",
                "name": "Creativity",
                "description": "Originality and creative value of the response",
                "scale": "1-10",
                "weight": 1.2,
            },
            {
                "id": "safety",
                "name": "Safety",
                "description": "Safety and harmlessness of the response",
                "scale": "1-10",
                "weight": 1.5,
            },
            {
                "id": "coherence",
                "name": "Coherence",
                "description": "Logical consistency and flow of the response",
                "scale": "1-10",
                "weight": 1.0,
            },
        ]
    }

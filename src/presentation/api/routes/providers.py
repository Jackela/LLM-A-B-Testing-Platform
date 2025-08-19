"""Model provider routes."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..auth.dependencies import get_current_active_user, require_admin
from ..auth.jwt_handler import UserInDB
from ..dependencies.singleton_container import get_container
from ..models.provider_models import (
    ConnectionTestResponse,
    ProviderHealthResponse,
    ProviderListResponse,
    ProviderResponse,
    ProviderUsageResponse,
    TestConnectionRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ProviderListResponse)
async def list_providers(
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """List all model providers."""
    try:
        provider_repo = await container.get_provider_repository()
        providers = await provider_repo.find_all()

        provider_responses = []
        for provider in providers:
            # Convert domain entity to API model
            from ..models.provider_models import (
                ModelCategory,
                ModelInfo,
                ProviderResponse,
                ProviderStatus,
                ProviderType,
                RateLimits,
            )

            # Mock conversion - in reality would map from domain entities
            provider_response = ProviderResponse(
                id=str(provider.id),
                name=provider.name,
                provider_type=ProviderType(provider.provider_type.value),
                status=ProviderStatus.ACTIVE,  # Mock status
                api_endpoint=provider.api_endpoint,
                rate_limits=RateLimits(
                    requests_per_minute=100, tokens_per_minute=10000, concurrent_requests=5
                ),
                models=[
                    ModelInfo(
                        id="gpt-4",
                        name="GPT-4",
                        category=ModelCategory.CHAT,
                        description="OpenAI's most capable model",
                        max_tokens=8192,
                        supports_streaming=True,
                        cost_per_1k_tokens=0.03,
                    )
                ],
                created_at=provider.created_at,
                updated_at=provider.updated_at,
                last_health_check=None,
            )
            provider_responses.append(provider_response)

        return ProviderListResponse(providers=provider_responses, total=len(provider_responses))

    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve providers"
        )


@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(
    provider_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get provider by ID."""
    try:
        provider_repo = await container.get_provider_repository()
        provider = await provider_repo.find_by_id(provider_id)

        if not provider:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

        # Convert to API model (simplified)
        from ..models.provider_models import (
            ModelCategory,
            ModelInfo,
            ProviderResponse,
            ProviderStatus,
            ProviderType,
            RateLimits,
        )

        return ProviderResponse(
            id=str(provider.id),
            name=provider.name,
            provider_type=ProviderType(provider.provider_type.value),
            status=ProviderStatus.ACTIVE,
            api_endpoint=provider.api_endpoint,
            rate_limits=RateLimits(
                requests_per_minute=100, tokens_per_minute=10000, concurrent_requests=5
            ),
            models=[],  # Would populate from provider.models
            created_at=provider.created_at,
            updated_at=provider.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve provider"
        )


@router.get("/{provider_id}/models")
async def get_provider_models(
    provider_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get models available for a provider."""
    try:
        provider_service = await container.get_provider_service()
        models = await provider_service.get_available_models(str(provider_id))

        return {"models": models}

    except Exception as e:
        logger.error(f"Error getting models for provider {provider_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve provider models",
        )


@router.post("/{provider_id}/test", response_model=ConnectionTestResponse)
async def test_provider_connection(
    provider_id: UUID,
    request: TestConnectionRequest,
    current_user: UserInDB = Depends(require_admin),
    container=Depends(get_container),
):
    """Test connection to a model provider."""
    try:
        provider_service = await container.get_provider_service()

        # Test connection
        test_result = await provider_service.test_connection(
            provider_id=str(provider_id),
            api_key=request.api_key,
            endpoint=request.endpoint,
            timeout=request.timeout_seconds,
        )

        return ConnectionTestResponse(
            success=test_result.get("success", False),
            response_time_ms=test_result.get("response_time_ms", 0),
            available_models=test_result.get("available_models", []),
            error=test_result.get("error"),
            timestamp=test_result.get("timestamp"),
        )

    except Exception as e:
        logger.error(f"Error testing provider {provider_id} connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test provider connection",
        )


@router.get("/{provider_id}/health", response_model=ProviderHealthResponse)
async def get_provider_health(
    provider_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get provider health status."""
    try:
        provider_service = await container.get_provider_service()
        health = await provider_service.get_health_status(str(provider_id))

        from ..models.provider_models import ProviderStatus

        return ProviderHealthResponse(
            provider_id=str(provider_id),
            status=ProviderStatus(health.get("status", "unknown")),
            response_time_ms=health.get("response_time_ms", 0),
            available_models=health.get("available_models", 0),
            rate_limit_remaining=health.get("rate_limit_remaining", {}),
            last_error=health.get("last_error"),
            timestamp=health.get("timestamp"),
        )

    except Exception as e:
        logger.error(f"Error getting provider {provider_id} health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get provider health",
        )


@router.get("/{provider_id}/usage", response_model=ProviderUsageResponse)
async def get_provider_usage(
    provider_id: UUID,
    days: int = Query(7, ge=1, le=90),
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get provider usage statistics."""
    try:
        analytics_service = await container.get_analytics_service()
        usage_data = await analytics_service.get_provider_usage(
            provider_id=str(provider_id), days=days
        )

        from ..models.provider_models import ModelUsageStats

        model_stats = [
            ModelUsageStats(
                model_id=stats["model_id"],
                total_requests=stats["total_requests"],
                successful_requests=stats["successful_requests"],
                failed_requests=stats["failed_requests"],
                average_response_time_ms=stats["average_response_time_ms"],
                total_tokens_used=stats["total_tokens_used"],
                total_cost=stats["total_cost"],
                last_used=stats.get("last_used"),
            )
            for stats in usage_data.get("models", [])
        ]

        return ProviderUsageResponse(
            provider_id=str(provider_id),
            period_start=usage_data["period_start"],
            period_end=usage_data["period_end"],
            total_requests=usage_data["total_requests"],
            total_cost=usage_data["total_cost"],
            models=model_stats,
        )

    except Exception as e:
        logger.error(f"Error getting provider {provider_id} usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get provider usage"
        )

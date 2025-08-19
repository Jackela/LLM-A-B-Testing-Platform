"""Test management routes."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from ..auth.dependencies import get_current_active_user, require_user_or_admin
from ..auth.jwt_handler import UserInDB
from ..dependencies.container import Container
from ..dependencies.singleton_container import get_container
from ..models.test_models import (
    AddSamplesRequest,
    CreateTestRequest,
    DifficultyLevel,
    StartTestRequest,
    TestFilters,
    TestListResponse,
    TestProgress,
    TestResponse,
    TestStatus,
    UpdateTestRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=TestResponse, status_code=status.HTTP_201_CREATED)
async def create_test(
    request: CreateTestRequest,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Create a new A/B test."""
    try:
        # Convert API model to use case DTO
        from ....application.dto.test_configuration_dto import (
            CreateTestCommandDTO,
            EvaluationConfigurationDTO,
            ModelConfigurationDTO,
            TestConfigurationDTO,
            TestSampleDTO,
        )

        # Create model configurations
        model_configs = [
            ModelConfigurationDTO(
                model_id=request.model_a.model_id,
                provider_name=request.model_a.provider,
                parameters=request.model_a.parameters,
                weight=0.5,
            ),
            ModelConfigurationDTO(
                model_id=request.model_b.model_id,
                provider_name=request.model_b.provider,
                parameters=request.model_b.parameters,
                weight=0.5,
            ),
        ]

        # Create evaluation configuration
        evaluation_config = EvaluationConfigurationDTO(
            template_id=request.evaluation_template_id,
            judge_count=3,
            consensus_threshold=0.7,
            quality_threshold=0.8,
            dimensions=["accuracy", "helpfulness", "clarity"],
        )

        # Create test configuration
        test_config = TestConfigurationDTO(
            models=model_configs,
            evaluation=evaluation_config,
            max_cost=100.0,  # Default max cost
            description=request.description or "",
        )

        # Create command
        command = CreateTestCommandDTO(
            name=request.name,
            configuration=test_config,
            samples=[],  # Start with empty samples
            tags=request.tags,
            metadata=request.metadata,
        )

        # Execute use case
        create_test_use_case = await container.get_create_test_use_case()
        result = await create_test_use_case.execute(command)

        if not result.created_test:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Test creation failed: {result.errors}",
            )

        # Return API response
        return TestResponse(
            id=str(result.test_id),
            name=request.name,
            description=request.description,
            status=TestStatus.DRAFT,
            model_a=request.model_a,
            model_b=request.model_b,
            evaluation_template_id=request.evaluation_template_id,
            sample_size=request.sample_size,
            difficulty_level=request.difficulty_level,
            tags=request.tags,
            metadata=request.metadata,
            created_at=result.created_at or None,
            updated_at=result.created_at or None,
            created_by=current_user.username,
            progress=None,
        )

    except Exception as e:
        logger.error(f"Error creating test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create test"
        )


@router.get("/", response_model=TestListResponse)
async def list_tests(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[List[TestStatus]] = Query(None),
    difficulty_filter: Optional[List[DifficultyLevel]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    created_by: Optional[str] = Query(None),
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """List tests with filtering and pagination."""
    try:
        # Get test repository
        test_repo = await container.get_test_repository()

        # Apply filters
        filters = {}
        if status_filter:
            filters["status"] = [s.value for s in status_filter]
        if difficulty_filter:
            filters["difficulty_level"] = [d.value for d in difficulty_filter]
        if tags:
            filters["tags"] = tags
        if created_by:
            filters["created_by"] = created_by

        # Get tests with pagination
        tests = await test_repo.find_with_filters(
            filters=filters, offset=(page - 1) * page_size, limit=page_size
        )

        total_count = await test_repo.count_with_filters(filters)

        # Convert to API models
        test_responses = []
        for test in tests:
            # Mock API response for now
            test_response = TestResponse(
                id=str(test.id),
                name=test.name,
                description=test.configuration.description,
                status=TestStatus(test.status.value),
                model_a={"model_id": "gpt-4", "provider": "openai", "parameters": {}},
                model_b={"model_id": "claude-3", "provider": "anthropic", "parameters": {}},
                evaluation_template_id="standard",
                sample_size=len(test.samples),
                difficulty_level=DifficultyLevel.MEDIUM,
                tags=[],
                metadata={},
                created_at=test.created_at,
                updated_at=test.updated_at,
                created_by="user",
            )
            test_responses.append(test_response)

        return TestListResponse(
            tests=test_responses, total=total_count, page=page, page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error listing tests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve tests"
        )


@router.get("/{test_id}", response_model=TestResponse)
async def get_test(
    test_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get test by ID."""
    try:
        test_repo = await container.get_test_repository()
        test = await test_repo.find_by_id(test_id)

        if not test:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test not found")

        # Convert to API model
        return TestResponse(
            id=str(test.id),
            name=test.name,
            description=test.configuration.description,
            status=TestStatus(test.status.value),
            model_a={"model_id": "gpt-4", "provider": "openai", "parameters": {}},
            model_b={"model_id": "claude-3", "provider": "anthropic", "parameters": {}},
            evaluation_template_id="standard",
            sample_size=len(test.samples),
            difficulty_level=DifficultyLevel.MEDIUM,
            tags=[],
            metadata={},
            created_at=test.created_at,
            updated_at=test.updated_at,
            created_by="user",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve test"
        )


@router.put("/{test_id}", response_model=TestResponse)
async def update_test(
    test_id: UUID,
    request: UpdateTestRequest,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Update test configuration."""
    try:
        # Get update test use case
        update_test_use_case = await container.get_update_test_use_case()

        # Convert to use case DTO
        from ....application.dto.test_configuration_dto import UpdateTestCommandDTO

        command = UpdateTestCommandDTO(
            test_id=test_id,
            name=request.name,
            description=request.description,
            tags=request.tags,
            metadata=request.metadata,
            updated_by=current_user.username,
        )

        result = await update_test_use_case.execute(command)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Test update failed: {result.errors}",
            )

        # Get updated test
        test_repo = await container.get_test_repository()
        updated_test = await test_repo.find_by_id(test_id)

        return TestResponse(
            id=str(updated_test.id),
            name=updated_test.name,
            description=updated_test.configuration.description,
            status=TestStatus(updated_test.status.value),
            model_a=request.model_a
            or {"model_id": "gpt-4", "provider": "openai", "parameters": {}},
            model_b=request.model_b
            or {"model_id": "claude-3", "provider": "anthropic", "parameters": {}},
            evaluation_template_id=request.evaluation_template_id or "standard",
            sample_size=request.sample_size or len(updated_test.samples),
            difficulty_level=request.difficulty_level or DifficultyLevel.MEDIUM,
            tags=request.tags or [],
            metadata=request.metadata or {},
            created_at=updated_test.created_at,
            updated_at=updated_test.updated_at,
            created_by="user",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update test"
        )


@router.delete("/{test_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_test(
    test_id: UUID,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Delete test."""
    try:
        test_repo = await container.get_test_repository()
        test = await test_repo.find_by_id(test_id)

        if not test:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test not found")

        # Check if test can be deleted (not running)
        if test.status.value in ["running", "paused"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete running or paused test",
            )

        await test_repo.delete(test_id)
        logger.info(f"Test {test_id} deleted by {current_user.username}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete test"
        )


@router.post("/{test_id}/start", response_model=TestProgress)
async def start_test(
    test_id: UUID,
    request: StartTestRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Start test execution."""
    try:
        # Get start test use case
        start_test_use_case = await container.get_start_test_use_case()

        # Convert to use case DTO
        from ....application.dto.test_configuration_dto import StartTestCommandDTO

        command = StartTestCommandDTO(
            test_id=test_id,
            concurrent_workers=request.concurrent_workers,
            timeout_seconds=request.timeout_seconds,
            started_by=current_user.username,
        )

        result = await start_test_use_case.execute(command)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to start test: {result.errors}",
            )

        # Add background task for test monitoring
        background_tasks.add_task(monitor_test_execution, test_id, container)

        return TestProgress(
            total_samples=result.total_samples,
            completed_samples=0,
            failed_samples=0,
            success_rate=0.0,
            estimated_completion=result.estimated_completion,
            current_status=TestStatus.RUNNING,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start test"
        )


@router.post("/{test_id}/stop")
async def stop_test(
    test_id: UUID,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Stop test execution."""
    try:
        test_repo = await container.get_test_repository()
        test = await test_repo.find_by_id(test_id)

        if not test:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test not found")

        if test.status.value != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Test is not running"
            )

        # Stop test (simplified implementation)
        test.pause()
        await test_repo.save(test)

        logger.info(f"Test {test_id} stopped by {current_user.username}")
        return {"message": "Test stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to stop test"
        )


@router.get("/{test_id}/progress", response_model=TestProgress)
async def get_test_progress(
    test_id: UUID,
    current_user: UserInDB = Depends(get_current_active_user),
    container=Depends(get_container),
):
    """Get test execution progress."""
    try:
        # Get monitor test use case
        monitor_test_use_case = await container.get_monitor_test_use_case()

        result = await monitor_test_use_case.get_progress(test_id)

        return TestProgress(
            total_samples=result.total_samples,
            completed_samples=result.completed_samples,
            failed_samples=result.failed_samples,
            success_rate=result.success_rate,
            estimated_completion=result.estimated_completion,
            current_status=TestStatus(result.current_status),
        )

    except Exception as e:
        logger.error(f"Error getting test progress {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get test progress"
        )


@router.post("/{test_id}/samples")
async def add_test_samples(
    test_id: UUID,
    request: AddSamplesRequest,
    current_user: UserInDB = Depends(require_user_or_admin),
    container=Depends(get_container),
):
    """Add samples to test."""
    try:
        # Get add samples use case
        add_samples_use_case = await container.get_add_samples_use_case()

        # Convert to use case DTO
        from ....application.dto.test_configuration_dto import AddSamplesCommandDTO, TestSampleDTO

        sample_dtos = [
            TestSampleDTO(
                prompt=sample.prompt,
                expected_output=sample.expected_response,
                difficulty="medium",  # Default
                metadata=sample.metadata,
            )
            for sample in request.samples
        ]

        command = AddSamplesCommandDTO(
            test_id=test_id, samples=sample_dtos, added_by=current_user.username
        )

        result = await add_samples_use_case.execute(command)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to add samples: {result.errors}",
            )

        return {"message": f"Successfully added {len(request.samples)} samples"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding samples to test {test_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add samples"
        )


async def monitor_test_execution(test_id: UUID, container: Container):
    """Background task to monitor test execution."""
    try:
        monitor_use_case = await container.get_monitor_test_use_case()
        await monitor_use_case.monitor_execution(test_id)
    except Exception as e:
        logger.error(f"Error monitoring test {test_id}: {e}")

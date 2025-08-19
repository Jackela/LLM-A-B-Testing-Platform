"""Service for orchestrating complex test workflows and coordination."""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from uuid import UUID

from ...domain.test_management.value_objects.test_status import TestStatus
from ..dto.test_configuration_dto import (
    CreateTestCommandDTO,
    StartTestCommandDTO,
    TestMonitoringResultDTO,
)
from ..interfaces.domain_event_publisher import DomainEventPublisher
from ..interfaces.unit_of_work import UnitOfWork
from ..use_cases.test_management.complete_test import CompleteTestUseCase
from ..use_cases.test_management.create_test import CreateTestUseCase
from ..use_cases.test_management.monitor_test import MonitorTestUseCase
from ..use_cases.test_management.process_samples import ProcessSamplesUseCase
from ..use_cases.test_management.start_test import StartTestUseCase

logger = logging.getLogger(__name__)


class TestOrchestrationService:
    """Service for orchestrating complex test workflows and managing test lifecycle."""

    def __init__(
        self,
        uow: UnitOfWork,
        event_publisher: DomainEventPublisher,
        create_test_use_case: CreateTestUseCase,
        start_test_use_case: StartTestUseCase,
        monitor_test_use_case: MonitorTestUseCase,
        complete_test_use_case: CompleteTestUseCase,
        process_samples_use_case: ProcessSamplesUseCase,
    ):
        self.uow = uow
        self.event_publisher = event_publisher
        self.create_test_use_case = create_test_use_case
        self.start_test_use_case = start_test_use_case
        self.monitor_test_use_case = monitor_test_use_case
        self.complete_test_use_case = complete_test_use_case
        self.process_samples_use_case = process_samples_use_case

        # Track active processing tasks
        self._active_processing_tasks: Dict[UUID, asyncio.Task] = {}
        self._orchestration_lock = asyncio.Lock()

    async def create_and_start_test(self, command: CreateTestCommandDTO) -> Dict:
        """Create a test and immediately start it if creation succeeds."""
        logger.info(f"Creating and starting test: {command.name}")

        try:
            # Step 1: Create the test
            create_result = await self.create_test_use_case.execute(command)

            if not create_result.created_test:
                logger.warning(f"Test creation failed: {create_result.errors}")
                return {
                    "success": False,
                    "stage": "creation",
                    "test_id": None,
                    "errors": create_result.errors,
                }

            test_id = create_result.test_id
            logger.info(f"Test created successfully: {test_id}")

            # Step 2: Start the test
            start_command = StartTestCommandDTO(test_id=test_id)
            start_result = await self.start_test_use_case.execute(start_command)

            if start_result.errors:
                logger.warning(f"Test start failed: {start_result.errors}")
                return {
                    "success": False,
                    "stage": "starting",
                    "test_id": test_id,
                    "errors": start_result.errors,
                }

            logger.info(f"Test started successfully: {test_id}")

            # Step 3: Begin sample processing
            await self.schedule_sample_processing(test_id)

            return {
                "success": True,
                "test_id": test_id,
                "status": start_result.status,
                "estimated_cost": create_result.estimated_cost,
                "estimated_duration": create_result.estimated_duration,
            }

        except Exception as e:
            logger.error(f"Error in create_and_start_test: {e}", exc_info=True)
            return {
                "success": False,
                "stage": "system_error",
                "test_id": None,
                "errors": [f"System error: {str(e)}"],
            }

    async def schedule_sample_processing(self, test_id: UUID, batch_size: int = 10) -> bool:
        """Schedule asynchronous sample processing for a test."""
        async with self._orchestration_lock:
            if test_id in self._active_processing_tasks:
                logger.warning(f"Sample processing already active for test {test_id}")
                return False

            # Create processing task
            task = asyncio.create_task(
                self._process_samples_with_monitoring(test_id, batch_size),
                name=f"process_samples_{test_id}",
            )

            self._active_processing_tasks[test_id] = task
            logger.info(f"Scheduled sample processing for test {test_id}")
            return True

    async def _process_samples_with_monitoring(self, test_id: UUID, batch_size: int) -> None:
        """Process samples with continuous monitoring and error recovery."""
        logger.info(f"Starting monitored sample processing for test {test_id}")

        try:
            max_retries = 3
            retry_count = 0
            processing_complete = False

            while not processing_complete and retry_count < max_retries:
                try:
                    # Process a batch of samples
                    result = await self.process_samples_use_case.execute(test_id, batch_size)

                    if result["success"]:
                        # Check if processing is complete
                        monitor_result = await self.monitor_test_use_case.execute(test_id)

                        if monitor_result.progress >= 1.0:
                            # All samples processed - trigger completion
                            logger.info(f"Sample processing complete for test {test_id}")
                            completion_result = await self.complete_test_use_case.execute(test_id)

                            if completion_result.status == TestStatus.COMPLETED.value:
                                processing_complete = True
                                logger.info(f"Test {test_id} completed successfully")
                            else:
                                logger.warning(
                                    f"Test completion had issues: {completion_result.errors}"
                                )
                        else:
                            # More samples to process - continue after brief delay
                            await asyncio.sleep(1.0)
                    else:
                        # Processing failed - retry after delay
                        retry_count += 1
                        logger.warning(
                            f"Sample processing failed (attempt {retry_count}): {result.get('error')}"
                        )
                        await asyncio.sleep(2.0**retry_count)  # Exponential backoff

                except asyncio.CancelledError:
                    logger.info(f"Sample processing cancelled for test {test_id}")
                    break
                except Exception as e:
                    retry_count += 1
                    logger.error(
                        f"Error in sample processing (attempt {retry_count}): {e}", exc_info=True
                    )
                    if retry_count >= max_retries:
                        # Mark test as failed
                        await self._mark_test_as_failed(
                            test_id, f"Processing failed after {max_retries} retries: {e}"
                        )
                        break
                    await asyncio.sleep(2.0**retry_count)

        except Exception as e:
            logger.error(
                f"Critical error in sample processing for test {test_id}: {e}", exc_info=True
            )
            await self._mark_test_as_failed(test_id, f"Critical processing error: {e}")
        finally:
            # Remove from active tasks
            async with self._orchestration_lock:
                self._active_processing_tasks.pop(test_id, None)
            logger.info(f"Sample processing task completed for test {test_id}")

    async def _mark_test_as_failed(self, test_id: UUID, reason: str) -> None:
        """Mark a test as failed due to processing errors."""
        try:
            async with self.uow:
                test = await self.uow.tests.find_by_id(test_id)
                if test and not test.status.is_terminal():
                    test.fail(reason)
                    await self.uow.tests.save(test)
                    await self.uow.commit()

                    # Publish failure events
                    domain_events = test.get_domain_events()
                    await self.event_publisher.publish_all(domain_events)
                    test.clear_domain_events()

                    logger.error(f"Test {test_id} marked as failed: {reason}")
        except Exception as e:
            logger.error(f"Error marking test as failed: {e}", exc_info=True)

    async def cancel_test_processing(self, test_id: UUID, reason: Optional[str] = None) -> bool:
        """Cancel active processing for a test."""
        async with self._orchestration_lock:
            task = self._active_processing_tasks.get(test_id)
            if task:
                task.cancel()
                logger.info(f"Cancelled processing task for test {test_id}")

            # Mark test as cancelled
            try:
                async with self.uow:
                    test = await self.uow.tests.find_by_id(test_id)
                    if test and not test.status.is_terminal():
                        test.cancel(reason or "Processing cancelled by user")
                        await self.uow.tests.save(test)
                        await self.uow.commit()

                        # Publish cancellation events
                        domain_events = test.get_domain_events()
                        await self.event_publisher.publish_all(domain_events)
                        test.clear_domain_events()

                        return True
            except Exception as e:
                logger.error(f"Error cancelling test {test_id}: {e}", exc_info=True)
                return False

        return False

    async def get_all_active_tests(self) -> List[TestMonitoringResultDTO]:
        """Get monitoring results for all active tests."""
        async with self.uow:
            active_tests = await self.uow.tests.find_active_tests()

            monitoring_results = []
            for test in active_tests:
                try:
                    result = await self.monitor_test_use_case.execute(test.id)
                    monitoring_results.append(result)
                except Exception as e:
                    logger.error(f"Error monitoring test {test.id}: {e}")

        return monitoring_results

    async def get_processing_status(self) -> Dict:
        """Get overall processing status across all tests."""
        async with self._orchestration_lock:
            active_processing_count = len(self._active_processing_tasks)
            active_test_ids = list(self._active_processing_tasks.keys())

        # Get test statuses
        test_status_counts = await self._get_test_status_counts()

        return {
            "active_processing_tasks": active_processing_count,
            "active_test_ids": [str(tid) for tid in active_test_ids],
            "test_status_counts": test_status_counts,
            "system_healthy": active_processing_count < 10,  # Arbitrary health check
        }

    async def _get_test_status_counts(self) -> Dict[str, int]:
        """Get counts of tests by status."""
        status_counts = {}

        try:
            async with self.uow:
                for status in TestStatus:
                    count = await self.uow.tests.count_by_status(status)
                    status_counts[status.value] = count
        except Exception as e:
            logger.error(f"Error getting test status counts: {e}")

        return status_counts

    async def health_check_processing_tasks(self) -> Dict:
        """Perform health check on active processing tasks."""
        health_report = {
            "healthy_tasks": 0,
            "failed_tasks": 0,
            "stuck_tasks": 0,
            "task_details": [],
        }

        async with self._orchestration_lock:
            for test_id, task in self._active_processing_tasks.items():
                task_status = {
                    "test_id": str(test_id),
                    "task_name": task.get_name(),
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                }

                if task.done():
                    if task.cancelled():
                        health_report["failed_tasks"] += 1
                        task_status["status"] = "cancelled"
                    elif task.exception():
                        health_report["failed_tasks"] += 1
                        task_status["status"] = "failed"
                        task_status["exception"] = str(task.exception())
                    else:
                        health_report["healthy_tasks"] += 1
                        task_status["status"] = "completed"
                else:
                    # Check if task might be stuck (simplified check)
                    health_report["healthy_tasks"] += 1
                    task_status["status"] = "running"

                health_report["task_details"].append(task_status)

        return health_report

    async def cleanup_completed_tasks(self) -> int:
        """Clean up completed processing tasks."""
        cleaned_count = 0

        async with self._orchestration_lock:
            completed_tasks = [
                test_id for test_id, task in self._active_processing_tasks.items() if task.done()
            ]

            for test_id in completed_tasks:
                del self._active_processing_tasks[test_id]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} completed processing tasks")

        return cleaned_count

    async def force_complete_test(self, test_id: UUID, reason: str = "Force completed") -> Dict:
        """Force complete a test regardless of processing status."""
        logger.warning(f"Force completing test {test_id}: {reason}")

        try:
            # Cancel active processing
            await self.cancel_test_processing(test_id, f"Cancelled for force completion: {reason}")

            # Complete the test
            result = await self.complete_test_use_case.execute(test_id, force_completion=True)

            return {
                "success": result.status == TestStatus.COMPLETED.value,
                "test_id": test_id,
                "status": result.status,
                "errors": result.errors,
            }

        except Exception as e:
            logger.error(f"Error force completing test {test_id}: {e}", exc_info=True)
            return {
                "success": False,
                "test_id": test_id,
                "errors": [f"Force completion error: {str(e)}"],
            }

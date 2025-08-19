"""Statistical analysis service for comprehensive A/B test analysis."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from ....domain.analytics.entities.analysis_result import (
    AggregatedData,
    AnalysisResult,
    ModelPerformanceMetrics,
)
from ....domain.analytics.entities.statistical_test import (
    CorrectionMethod,
    StatisticalTest,
    TestType,
)
from ....domain.analytics.exceptions import InsufficientDataError, StatisticalError, ValidationError
from ....domain.analytics.repositories.analytics_repository import AnalyticsRepository
from ....domain.analytics.services.data_aggregator import DataAggregator
from ....domain.analytics.services.insight_generator import InsightGenerator
from ....domain.analytics.services.significance_tester import SignificanceTester
from ....domain.analytics.value_objects.confidence_interval import ConfidenceInterval
from ....domain.analytics.value_objects.cost_data import CostData
from ....domain.analytics.value_objects.test_result import TestResult
from ....domain.test_management.repositories.test_repository import TestRepository
from ...dto.analysis_request_dto import AnalysisRequestDTO
from .significance_analyzer import SignificanceAnalyzer

logger = logging.getLogger(__name__)


class ComprehensiveAnalysisResult:
    """Comprehensive analysis result containing all statistical analysis outputs."""

    def __init__(
        self,
        test_id: UUID,
        statistical_tests: Dict[str, TestResult],
        effect_analysis: Dict[str, Any],
        insights: List[Any],
        confidence_level: float,
        model_performances: Dict[str, ModelPerformanceMetrics],
        aggregated_data: Dict[str, List[AggregatedData]],
        processing_time_ms: int,
    ):
        self.test_id = test_id
        self.statistical_tests = statistical_tests
        self.effect_analysis = effect_analysis
        self.insights = insights
        self.confidence_level = confidence_level
        self.model_performances = model_performances
        self.aggregated_data = aggregated_data
        self.processing_time_ms = processing_time_ms
        self.created_at = datetime.utcnow()


class AnalysisConfig:
    """Configuration for statistical analysis."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        correction_method: CorrectionMethod = CorrectionMethod.BONFERRONI,
        include_effect_sizes: bool = True,
        include_power_analysis: bool = True,
        enable_dimension_analysis: bool = True,
        enable_cost_analysis: bool = True,
        minimum_sample_size: int = 30,
    ):
        self.confidence_level = confidence_level
        self.correction_method = correction_method
        self.include_effect_sizes = include_effect_sizes
        self.include_power_analysis = include_power_analysis
        self.enable_dimension_analysis = enable_dimension_analysis
        self.enable_cost_analysis = enable_cost_analysis
        self.minimum_sample_size = minimum_sample_size


class StatisticalAnalysisService:
    """Service for comprehensive statistical analysis of A/B tests."""

    def __init__(
        self,
        test_repository: TestRepository,
        analytics_repository: AnalyticsRepository,
        significance_tester: SignificanceTester,
        data_aggregator: DataAggregator,
        insight_generator: InsightGenerator,
        significance_analyzer: SignificanceAnalyzer,
    ):
        self.test_repository = test_repository
        self.analytics_repository = analytics_repository
        self.significance_tester = significance_tester
        self.data_aggregator = data_aggregator
        self.insight_generator = insight_generator
        self.significance_analyzer = significance_analyzer
        self._logger = logger.getChild(self.__class__.__name__)

    async def analyze_test_results(
        self, test_id: UUID, analysis_config: Optional[AnalysisConfig] = None
    ) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive statistical analysis on test results.

        Args:
            test_id: ID of the test to analyze
            analysis_config: Configuration for the analysis

        Returns:
            ComprehensiveAnalysisResult with all analysis outputs

        Raises:
            ValidationError: If test_id is invalid or test not found
            InsufficientDataError: If not enough data for analysis
            StatisticalError: If statistical analysis fails
        """
        start_time = datetime.utcnow()

        if analysis_config is None:
            analysis_config = AnalysisConfig()

        try:
            self._logger.info(f"Starting comprehensive analysis for test {test_id}")

            # 1. Load test and validation
            test = await self._load_and_validate_test(test_id)

            # 2. Load evaluation results
            evaluation_results = await self._load_evaluation_results(test_id)

            # 3. Validate sufficient data
            self._validate_sufficient_data(evaluation_results, analysis_config)

            # 4. Perform statistical tests in parallel
            statistical_results = await self._perform_statistical_tests(
                evaluation_results, analysis_config
            )

            # 5. Calculate effect sizes and confidence intervals
            effect_analysis = await self._analyze_effect_sizes(
                statistical_results, evaluation_results, analysis_config
            )

            # 6. Aggregate data by different dimensions
            aggregated_data = await self._aggregate_data_by_dimensions(
                evaluation_results, analysis_config
            )

            # 7. Calculate model performance metrics
            model_performances = await self._calculate_model_performances(
                evaluation_results, analysis_config
            )

            # 8. Generate insights
            insights = await self._generate_insights(
                statistical_results, effect_analysis, model_performances, analysis_config
            )

            # 9. Calculate processing time
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            result = ComprehensiveAnalysisResult(
                test_id=test_id,
                statistical_tests=statistical_results,
                effect_analysis=effect_analysis,
                insights=insights,
                confidence_level=analysis_config.confidence_level,
                model_performances=model_performances,
                aggregated_data=aggregated_data,
                processing_time_ms=processing_time_ms,
            )

            self._logger.info(f"Analysis completed in {processing_time_ms}ms for test {test_id}")
            return result

        except Exception as e:
            self._logger.error(f"Analysis failed for test {test_id}: {str(e)}")
            raise StatisticalError(f"Analysis failed: {str(e)}")

    async def analyze_model_comparison(
        self, test_id: UUID, model_ids: List[str], analysis_config: Optional[AnalysisConfig] = None
    ) -> Dict[str, TestResult]:
        """
        Analyze pairwise comparisons between specific models.

        Args:
            test_id: ID of the test containing the models
            model_ids: List of model IDs to compare
            analysis_config: Configuration for the analysis

        Returns:
            Dictionary of comparison results keyed by model pair
        """
        if len(model_ids) < 2:
            raise ValidationError("At least 2 models required for comparison")

        if analysis_config is None:
            analysis_config = AnalysisConfig()

        try:
            # Load evaluation results filtered by model IDs
            evaluation_results = await self._load_evaluation_results(test_id, model_ids)

            # Group results by model
            model_results = {}
            for result in evaluation_results:
                model_id = result.model_id
                if model_id not in model_results:
                    model_results[model_id] = []
                model_results[model_id].append(result)

            # Perform pairwise comparisons
            return await self.significance_tester.test_multiple_models(
                model_results,
                test_type=TestType.TTEST_INDEPENDENT,
                confidence_level=analysis_config.confidence_level,
                correction_method=analysis_config.correction_method,
            )

        except Exception as e:
            self._logger.error(f"Model comparison failed: {str(e)}")
            raise StatisticalError(f"Model comparison failed: {str(e)}")

    async def analyze_dimension_performance(
        self, test_id: UUID, dimensions: List[str], analysis_config: Optional[AnalysisConfig] = None
    ) -> Dict[str, TestResult]:
        """
        Analyze performance differences across evaluation dimensions.

        Args:
            test_id: ID of the test to analyze
            dimensions: List of dimensions to compare
            analysis_config: Configuration for the analysis

        Returns:
            Dictionary of dimension comparison results
        """
        if not dimensions:
            raise ValidationError("At least one dimension required")

        if analysis_config is None:
            analysis_config = AnalysisConfig()

        try:
            evaluation_results = await self._load_evaluation_results(test_id)

            return await self.significance_tester.test_dimension_differences(
                evaluation_results, dimensions, confidence_level=analysis_config.confidence_level
            )

        except Exception as e:
            self._logger.error(f"Dimension analysis failed: {str(e)}")
            raise StatisticalError(f"Dimension analysis failed: {str(e)}")

    async def calculate_required_sample_sizes(
        self, effect_sizes: Dict[str, float], power: float = 0.8, alpha: float = 0.05
    ) -> Dict[str, int]:
        """
        Calculate required sample sizes for detecting given effect sizes.

        Args:
            effect_sizes: Dictionary of effect sizes by test type
            power: Desired statistical power (default: 0.8)
            alpha: Significance level (default: 0.05)

        Returns:
            Dictionary of required sample sizes by test type
        """
        sample_sizes = {}

        for test_name, effect_size in effect_sizes.items():
            try:
                sample_size = await self.significance_tester.calculate_required_sample_size(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    test_type=TestType.TTEST_INDEPENDENT,
                )
                sample_sizes[test_name] = sample_size

            except Exception as e:
                self._logger.warning(f"Sample size calculation failed for {test_name}: {str(e)}")
                sample_sizes[test_name] = 50  # Conservative default

        return sample_sizes

    async def _load_and_validate_test(self, test_id: UUID):
        """Load and validate test exists."""
        test = await self.test_repository.find_by_id(test_id)
        if not test:
            raise ValidationError(f"Test {test_id} not found")
        return test

    async def _load_evaluation_results(self, test_id: UUID, model_ids: Optional[List[str]] = None):
        """Load evaluation results for the test."""
        evaluation_results = await self.analytics_repository.get_evaluation_results(test_id)

        if model_ids:
            evaluation_results = [
                result for result in evaluation_results if result.model_id in model_ids
            ]

        return evaluation_results

    def _validate_sufficient_data(self, evaluation_results, analysis_config: AnalysisConfig):
        """Validate sufficient data for analysis."""
        if not evaluation_results:
            raise InsufficientDataError("No evaluation results found for analysis")

        # Group by model and check sample sizes
        model_counts = {}
        for result in evaluation_results:
            model_id = result.model_id
            model_counts[model_id] = model_counts.get(model_id, 0) + 1

        insufficient_models = [
            model_id
            for model_id, count in model_counts.items()
            if count < analysis_config.minimum_sample_size
        ]

        if insufficient_models:
            raise InsufficientDataError(
                f"Insufficient data for models: {insufficient_models}. "
                f"Minimum {analysis_config.minimum_sample_size} samples required per model."
            )

    async def _perform_statistical_tests(
        self, evaluation_results, analysis_config: AnalysisConfig
    ) -> Dict[str, TestResult]:
        """Perform all statistical tests."""
        tasks = []

        # Group results by model
        model_results = {}
        for result in evaluation_results:
            model_id = result.model_id
            if model_id not in model_results:
                model_results[model_id] = []
            model_results[model_id].append(result)

        # Model comparison tests
        if len(model_results) >= 2:
            task = self.significance_tester.test_multiple_models(
                model_results,
                confidence_level=analysis_config.confidence_level,
                correction_method=analysis_config.correction_method,
            )
            tasks.append(("model_comparisons", task))

        # Dimension analysis if enabled
        if analysis_config.enable_dimension_analysis and evaluation_results:
            # Extract unique dimensions
            dimensions = set()
            for result in evaluation_results:
                dimensions.update(result.dimension_scores.keys())

            if len(dimensions) >= 2:
                task = self.significance_tester.test_dimension_differences(
                    evaluation_results,
                    list(dimensions),
                    confidence_level=analysis_config.confidence_level,
                )
                tasks.append(("dimension_comparisons", task))

        # Difficulty level analysis
        task = self.significance_tester.test_difficulty_level_effects(
            evaluation_results, confidence_level=analysis_config.confidence_level
        )
        tasks.append(("difficulty_analysis", task))

        # Execute all tests
        results = {}
        for test_name, task in tasks:
            try:
                if asyncio.iscoroutine(task):
                    result = await task
                else:
                    result = task

                if result:
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        results[test_name] = result

            except Exception as e:
                self._logger.warning(f"Statistical test {test_name} failed: {str(e)}")
                continue

        return results

    async def _analyze_effect_sizes(
        self,
        statistical_results: Dict[str, TestResult],
        evaluation_results,
        analysis_config: AnalysisConfig,
    ) -> Dict[str, Any]:
        """Analyze effect sizes and practical significance."""
        if not analysis_config.include_effect_sizes:
            return {}

        effect_analysis = {
            "effect_sizes": {},
            "practical_significance": {},
            "confidence_intervals": {},
        }

        for test_name, test_result in statistical_results.items():
            # Extract effect size
            effect_size = test_result.effect_size
            if effect_size is not None:
                effect_analysis["effect_sizes"][test_name] = {
                    "value": float(effect_size),
                    "interpretation": self._interpret_effect_size(effect_size),
                }

            # Extract confidence interval
            if test_result.confidence_interval:
                effect_analysis["confidence_intervals"][test_name] = {
                    "lower": float(test_result.confidence_interval.lower_bound),
                    "upper": float(test_result.confidence_interval.upper_bound),
                    "confidence_level": analysis_config.confidence_level,
                }

            # Assess practical significance
            effect_analysis["practical_significance"][test_name] = (
                self._assess_practical_significance(test_result, analysis_config)
            )

        return effect_analysis

    async def _aggregate_data_by_dimensions(
        self, evaluation_results, analysis_config: AnalysisConfig
    ) -> Dict[str, List[AggregatedData]]:
        """Aggregate data by different dimensions."""
        aggregated_data = {}

        try:
            # Aggregate by model
            model_aggregation = await self.data_aggregator.aggregate_by_model(evaluation_results)
            aggregated_data["by_model"] = model_aggregation

            # Aggregate by difficulty level
            difficulty_aggregation = await self.data_aggregator.aggregate_by_difficulty(
                evaluation_results
            )
            aggregated_data["by_difficulty"] = difficulty_aggregation

            # Aggregate by evaluation dimension
            dimension_aggregation = await self.data_aggregator.aggregate_by_dimension(
                evaluation_results
            )
            aggregated_data["by_dimension"] = dimension_aggregation

        except Exception as e:
            self._logger.warning(f"Data aggregation failed: {str(e)}")

        return aggregated_data

    async def _calculate_model_performances(
        self, evaluation_results, analysis_config: AnalysisConfig
    ) -> Dict[str, ModelPerformanceMetrics]:
        """Calculate comprehensive model performance metrics."""
        model_performances = {}

        # Group results by model
        model_results = {}
        for result in evaluation_results:
            model_id = result.model_id
            if model_id not in model_results:
                model_results[model_id] = []
            model_results[model_id].append(result)

        # Calculate metrics for each model
        for model_id, results in model_results.items():
            try:
                # Calculate overall performance
                overall_scores = [float(r.overall_score) for r in results]
                overall_score = Decimal(str(sum(overall_scores) / len(overall_scores)))

                # Calculate dimension scores
                dimension_scores = {}
                for result in results:
                    for dim, score in result.dimension_scores.items():
                        if dim not in dimension_scores:
                            dimension_scores[dim] = []
                        dimension_scores[dim].append(float(score))

                # Average dimension scores
                avg_dimension_scores = {
                    dim: Decimal(str(sum(scores) / len(scores)))
                    for dim, scores in dimension_scores.items()
                }

                # Calculate confidence score (based on consistency)
                score_variance = sum((s - float(overall_score)) ** 2 for s in overall_scores) / len(
                    overall_scores
                )
                confidence_score = max(Decimal("0"), Decimal("1") - Decimal(str(score_variance)))

                # Cost metrics (if available)
                cost_metrics = None
                if analysis_config.enable_cost_analysis:
                    cost_metrics = await self._calculate_cost_metrics(model_id, results)

                # Get model name
                model_name = results[0].metadata.get("model_name", model_id)

                performance = ModelPerformanceMetrics(
                    model_id=model_id,
                    model_name=model_name,
                    overall_score=overall_score,
                    dimension_scores=avg_dimension_scores,
                    sample_count=len(results),
                    confidence_score=confidence_score,
                    cost_metrics=cost_metrics,
                    quality_indicators={
                        "score_variance": float(score_variance),
                        "completion_rate": len([r for r in results if r.is_completed()])
                        / len(results),
                        "error_rate": len([r for r in results if r.has_error()]) / len(results),
                    },
                )

                model_performances[model_id] = performance

            except Exception as e:
                self._logger.warning(
                    f"Performance calculation failed for model {model_id}: {str(e)}"
                )
                continue

        return model_performances

    async def _generate_insights(
        self,
        statistical_results: Dict[str, TestResult],
        effect_analysis: Dict[str, Any],
        model_performances: Dict[str, ModelPerformanceMetrics],
        analysis_config: AnalysisConfig,
    ):
        """Generate actionable insights from analysis results."""
        try:
            return await self.insight_generator.generate_comprehensive_insights(
                statistical_results=statistical_results,
                effect_analysis=effect_analysis,
                model_performances=model_performances,
                confidence_level=analysis_config.confidence_level,
            )
        except Exception as e:
            self._logger.warning(f"Insight generation failed: {str(e)}")
            return []

    async def _calculate_cost_metrics(self, model_id: str, results) -> Optional[CostData]:
        """Calculate cost metrics for a model."""
        try:
            # Extract cost information from results metadata
            total_cost = Decimal("0")
            total_tokens = 0

            for result in results:
                if "cost" in result.metadata:
                    total_cost += Decimal(str(result.metadata["cost"]))
                if "tokens" in result.metadata:
                    total_tokens += result.metadata["tokens"]

            if total_cost > 0 or total_tokens > 0:
                from ....domain.model_provider.value_objects.money import Money

                return CostData(
                    total_cost=Money(amount=total_cost, currency="USD"),
                    cost_per_sample=(
                        Money(amount=total_cost / len(results), currency="USD")
                        if len(results) > 0
                        else Money(amount=Decimal("0"), currency="USD")
                    ),
                    total_tokens=total_tokens,
                    average_tokens_per_sample=(
                        total_tokens / len(results) if len(results) > 0 else 0
                    ),
                )

        except Exception as e:
            self._logger.warning(f"Cost calculation failed for model {model_id}: {str(e)}")

        return None

    def _interpret_effect_size(self, effect_size: Decimal) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(float(effect_size))

        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def _assess_practical_significance(
        self, test_result: TestResult, analysis_config: AnalysisConfig
    ) -> Dict[str, Any]:
        """Assess practical significance of test result."""
        # Conservative thresholds for practical significance
        minimum_effect_size = 0.2  # Cohen's small effect
        minimum_improvement = 0.05  # 5% improvement threshold

        is_statistically_significant = test_result.is_significant()
        effect_size = float(test_result.effect_size) if test_result.effect_size else 0

        is_practically_significant = (
            abs(effect_size) >= minimum_effect_size or abs(effect_size) >= minimum_improvement
        )

        return {
            "is_statistically_significant": is_statistically_significant,
            "is_practically_significant": is_practically_significant,
            "effect_size_magnitude": self._interpret_effect_size(
                test_result.effect_size or Decimal("0")
            ),
            "recommendation": self._get_practical_recommendation(
                is_statistically_significant, is_practically_significant, effect_size
            ),
        }

    def _get_practical_recommendation(
        self, statistically_significant: bool, practically_significant: bool, effect_size: float
    ) -> str:
        """Get practical recommendation based on significance."""
        if statistically_significant and practically_significant:
            return "Strong evidence for difference - recommend action"
        elif statistically_significant and not practically_significant:
            return "Statistically significant but small effect - consider context"
        elif not statistically_significant and practically_significant:
            return "Large effect but not statistically significant - collect more data"
        else:
            return "No evidence for meaningful difference - no action needed"

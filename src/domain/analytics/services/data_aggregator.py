"""Data aggregation service for analytics domain."""

import math
import statistics
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

from ...evaluation.entities.evaluation_result import EvaluationResult
from ...model_provider.entities.model_response import ModelResponse
from ..entities.aggregation_rule import AggregationRule, AggregationType
from ..entities.analysis_result import AggregatedData, ModelPerformanceMetrics
from ..exceptions import AggregationError, InvalidAggregationRule, MissingDataError, ValidationError
from ..value_objects.confidence_interval import ConfidenceInterval
from ..value_objects.cost_data import CostData


class DataAggregator:
    """Domain service for data aggregation and processing."""

    def __init__(self):
        """Initialize data aggregator."""
        self._confidence_level = Decimal("0.95")

    def aggregate_evaluation_results(
        self, evaluation_results: List[EvaluationResult], aggregation_rules: List[AggregationRule]
    ) -> Dict[str, List[AggregatedData]]:
        """Aggregate evaluation results according to specified rules."""

        if not evaluation_results:
            raise MissingDataError("Evaluation results cannot be empty")

        if not aggregation_rules:
            raise InvalidAggregationRule("At least one aggregation rule required")

        # Validate evaluation results
        self._validate_evaluation_results(evaluation_results)

        # Convert evaluation results to data rows
        data_rows = self._convert_evaluation_results_to_rows(evaluation_results)

        # Apply each aggregation rule
        aggregated_results = {}

        for rule in aggregation_rules:
            if not rule.is_active:
                continue

            try:
                aggregated_data = self._apply_aggregation_rule(rule, data_rows)
                aggregated_results[rule.name] = aggregated_data
            except Exception as e:
                raise AggregationError(f"Failed to apply rule '{rule.name}': {str(e)}")

        return aggregated_results

    def calculate_model_performance(
        self,
        model_id: str,
        model_name: str,
        evaluation_results: List[EvaluationResult],
        model_responses: Optional[List[ModelResponse]] = None,
    ) -> ModelPerformanceMetrics:
        """Calculate comprehensive model performance metrics."""

        if not evaluation_results:
            raise MissingDataError("Evaluation results cannot be empty")

        # Validate results belong to the same model
        model_results = [r for r in evaluation_results if r.judge_id == model_id]
        if not model_results:
            model_results = evaluation_results  # Assume all results are for this model

        # Calculate overall performance score
        overall_score = self._calculate_weighted_overall_score(model_results)

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(model_results)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(model_results)

        # Calculate cost metrics if model responses provided
        cost_metrics = None
        if model_responses:
            cost_metrics = self._calculate_cost_metrics(model_responses)

        # Calculate quality indicators
        quality_indicators = self._calculate_quality_indicators(model_results)

        return ModelPerformanceMetrics(
            model_id=model_id,
            model_name=model_name,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            sample_count=len(model_results),
            confidence_score=confidence_score,
            cost_metrics=cost_metrics,
            quality_indicators=quality_indicators,
        )

    def aggregate_by_difficulty_level(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, AggregatedData]:
        """Aggregate results by difficulty level."""

        # Group by difficulty level
        difficulty_groups = defaultdict(list)

        for result in evaluation_results:
            difficulty = result.metadata.get("difficulty_level", "unknown")
            difficulty_groups[difficulty].append(float(result.overall_score))

        # Calculate aggregated data for each difficulty level
        aggregated_data = {}

        for difficulty, scores in difficulty_groups.items():
            if not scores:
                continue

            mean_score = Decimal(str(statistics.mean(scores))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Calculate confidence interval
            ci = self._calculate_confidence_interval(scores)

            aggregated_data[difficulty] = AggregatedData(
                group_key=(difficulty,),
                group_labels={"difficulty_level": difficulty},
                sample_count=len(scores),
                aggregated_value=mean_score,
                confidence_interval=(ci.lower_bound, ci.upper_bound) if ci else None,
                metadata={
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "median_score": statistics.median(scores),
                },
            )

        return aggregated_data

    def aggregate_by_category(
        self, evaluation_results: List[EvaluationResult], category_field: str = "category"
    ) -> Dict[str, AggregatedData]:
        """Aggregate results by category."""

        # Group by category
        category_groups = defaultdict(list)

        for result in evaluation_results:
            category = result.metadata.get(category_field, "unknown")
            category_groups[category].append(float(result.overall_score))

        # Calculate aggregated data for each category
        aggregated_data = {}

        for category, scores in category_groups.items():
            if not scores:
                continue

            mean_score = Decimal(str(statistics.mean(scores))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            # Calculate confidence interval
            ci = self._calculate_confidence_interval(scores)

            aggregated_data[category] = AggregatedData(
                group_key=(category,),
                group_labels={category_field: category},
                sample_count=len(scores),
                aggregated_value=mean_score,
                confidence_interval=(ci.lower_bound, ci.upper_bound) if ci else None,
                metadata={
                    "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "quartiles": self._calculate_quartiles(scores),
                },
            )

        return aggregated_data

    def aggregate_temporal_data(
        self,
        evaluation_results: List[EvaluationResult],
        time_window: str = "hour",  # hour, day, week
    ) -> Dict[str, AggregatedData]:
        """Aggregate results by time windows."""

        # Group by time windows
        time_groups = defaultdict(list)

        for result in evaluation_results:
            time_key = self._get_time_window_key(result.created_at, time_window)
            time_groups[time_key].append(float(result.overall_score))

        # Calculate aggregated data for each time window
        aggregated_data = {}

        for time_key, scores in time_groups.items():
            if not scores:
                continue

            mean_score = Decimal(str(statistics.mean(scores))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            aggregated_data[time_key] = AggregatedData(
                group_key=(time_key,),
                group_labels={"time_window": time_key},
                sample_count=len(scores),
                aggregated_value=mean_score,
                confidence_interval=None,  # Skip CI for temporal data
                metadata={"trend": self._calculate_trend(scores) if len(scores) > 2 else "stable"},
            )

        return aggregated_data

    def _validate_evaluation_results(self, evaluation_results: List[EvaluationResult]) -> None:
        """Validate evaluation results for aggregation."""

        incomplete_results = [r for r in evaluation_results if not r.is_completed()]
        if incomplete_results:
            raise ValidationError(f"{len(incomplete_results)} incomplete evaluation results")

        failed_results = [r for r in evaluation_results if r.has_error()]
        if failed_results:
            raise ValidationError(f"{len(failed_results)} failed evaluation results")

    def _convert_evaluation_results_to_rows(
        self, evaluation_results: List[EvaluationResult]
    ) -> List[Dict[str, Any]]:
        """Convert evaluation results to data rows for aggregation."""

        data_rows = []

        for result in evaluation_results:
            # Extract base data
            row = {
                "result_id": str(result.result_id),
                "judge_id": result.judge_id,
                "template_id": str(result.template_id),
                "overall_score": float(result.overall_score),
                "confidence_score": float(result.confidence_score),
                "evaluation_time_ms": result.evaluation_time_ms,
                "created_at": result.created_at,
                "model": result.judge_id,  # Alias for grouping
            }

            # Add dimension scores
            for dimension, score in result.dimension_scores.items():
                row[f"dimension_{dimension}"] = score

            # Add metadata
            row.update(result.metadata)

            # Add quality indicators
            if result.quality_report:
                row["quality_score"] = float(result.quality_report.quality_score)
                row["quality_level"] = result.quality_report.overall_quality.value
                row["has_quality_issues"] = not result.quality_report.is_passing()

            data_rows.append(row)

        return data_rows

    def _apply_aggregation_rule(
        self, rule: AggregationRule, data_rows: List[Dict[str, Any]]
    ) -> List[AggregatedData]:
        """Apply single aggregation rule to data rows."""

        # Filter data according to rule conditions
        filtered_rows = [row for row in data_rows if rule.applies_to_data(row)]

        if not filtered_rows:
            return []  # No data matches rule conditions

        # Group data by grouping fields
        groups = defaultdict(list)
        group_labels_map = {}

        for row in filtered_rows:
            group_key = rule.get_grouping_key(row)
            target_value = rule.get_target_value(row)

            if target_value is not None:
                groups[group_key].append(target_value)

                # Store group labels for first occurrence
                if group_key not in group_labels_map:
                    labels = {}
                    for i, field in enumerate(rule.group_by_fields):
                        labels[field.value] = str(group_key[i]) if i < len(group_key) else "unknown"
                    group_labels_map[group_key] = labels

        # Calculate aggregated values for each group
        aggregated_data = []

        for group_key, values in groups.items():
            if not values:
                continue

            try:
                # Calculate aggregated value based on aggregation type
                aggregated_value = self._calculate_aggregated_value(
                    values, rule.aggregation_type, rule.parameters, rule.weighting_rule
                )

                # Calculate confidence interval if appropriate
                confidence_interval = None
                if rule.aggregation_type in [AggregationType.MEAN, AggregationType.WEIGHTED_MEAN]:
                    ci = self._calculate_confidence_interval(values)
                    confidence_interval = (ci.lower_bound, ci.upper_bound) if ci else None

                aggregated_data.append(
                    AggregatedData(
                        group_key=group_key,
                        group_labels=group_labels_map.get(group_key, {}),
                        sample_count=len(values),
                        aggregated_value=aggregated_value,
                        confidence_interval=confidence_interval,
                        metadata={
                            "aggregation_type": rule.aggregation_type.value,
                            "rule_name": rule.name,
                            "target_field": rule.target_field,
                        },
                    )
                )

            except Exception as e:
                # Skip group if aggregation fails
                continue

        return aggregated_data

    def _calculate_aggregated_value(
        self,
        values: List[float],
        aggregation_type: AggregationType,
        parameters: Dict[str, Any],
        weighting_rule: Optional[Any] = None,
    ) -> Decimal:
        """Calculate aggregated value based on aggregation type."""

        if not values:
            return Decimal("0")

        try:
            if aggregation_type == AggregationType.MEAN:
                result = statistics.mean(values)
            elif aggregation_type == AggregationType.MEDIAN:
                result = statistics.median(values)
            elif aggregation_type == AggregationType.MODE:
                result = statistics.mode(values)
            elif aggregation_type == AggregationType.SUM:
                result = sum(values)
            elif aggregation_type == AggregationType.COUNT:
                result = len(values)
            elif aggregation_type == AggregationType.MIN:
                result = min(values)
            elif aggregation_type == AggregationType.MAX:
                result = max(values)
            elif aggregation_type == AggregationType.STD:
                result = statistics.stdev(values) if len(values) > 1 else 0.0
            elif aggregation_type == AggregationType.VAR:
                result = statistics.variance(values) if len(values) > 1 else 0.0
            elif aggregation_type == AggregationType.RANGE:
                result = max(values) - min(values) if len(values) > 1 else 0.0
            elif aggregation_type == AggregationType.IQR:
                q75, q25 = (
                    self._calculate_quartiles(values)[2],
                    self._calculate_quartiles(values)[0],
                )
                result = q75 - q25
            elif aggregation_type == AggregationType.PERCENTILE:
                percentile = parameters.get("percentile", 50)
                result = self._calculate_percentile(values, percentile)
            elif aggregation_type == AggregationType.QUARTILE:
                quartile = parameters.get("quartile", 2)  # Default to median
                quartiles = self._calculate_quartiles(values)
                result = quartiles[quartile - 1]
            elif aggregation_type == AggregationType.GEOMETRIC_MEAN:
                if all(v > 0 for v in values):
                    result = statistics.geometric_mean(values)
                else:
                    result = 0.0  # Cannot calculate geometric mean with non-positive values
            elif aggregation_type == AggregationType.HARMONIC_MEAN:
                if all(v > 0 for v in values):
                    result = statistics.harmonic_mean(values)
                else:
                    result = 0.0  # Cannot calculate harmonic mean with non-positive values
            elif aggregation_type == AggregationType.WEIGHTED_MEAN:
                # For weighted mean, we would need weights from the weighting rule
                # For now, fall back to regular mean
                result = statistics.mean(values)
            else:
                raise AggregationError(f"Unsupported aggregation type: {aggregation_type}")

            return Decimal(str(result)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        except Exception as e:
            raise AggregationError(f"Failed to calculate {aggregation_type.value}: {str(e)}")

    def _calculate_weighted_overall_score(self, results: List[EvaluationResult]) -> Decimal:
        """Calculate weighted overall score from evaluation results."""

        if not results:
            return Decimal("0")

        # Weight by confidence score
        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for result in results:
            weight = result.confidence_score
            score = result.overall_score

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return Decimal(str(statistics.mean([float(r.overall_score) for r in results])))

        return (weighted_sum / total_weight).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_dimension_scores(self, results: List[EvaluationResult]) -> Dict[str, Decimal]:
        """Calculate average scores for each dimension."""

        dimension_scores = defaultdict(list)

        for result in results:
            for dimension, score in result.dimension_scores.items():
                dimension_scores[dimension].append(score)

        # Calculate average for each dimension
        averaged_scores = {}
        for dimension, scores in dimension_scores.items():
            if scores:
                avg_score = statistics.mean(scores)
                averaged_scores[dimension] = Decimal(str(avg_score)).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )

        return averaged_scores

    def _calculate_confidence_score(self, results: List[EvaluationResult]) -> Decimal:
        """Calculate aggregate confidence score."""

        if not results:
            return Decimal("0")

        confidence_scores = [float(r.confidence_score) for r in results]

        # Use geometric mean for confidence aggregation
        if all(c > 0 for c in confidence_scores):
            geometric_mean = statistics.geometric_mean(confidence_scores)
            return Decimal(str(geometric_mean)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        else:
            # Fall back to arithmetic mean
            arithmetic_mean = statistics.mean(confidence_scores)
            return Decimal(str(arithmetic_mean)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_cost_metrics(self, model_responses: List[ModelResponse]) -> Optional[CostData]:
        """Calculate cost metrics from model responses."""

        if not model_responses:
            return None

        # This would need integration with the actual ModelResponse entity
        # For now, return None as placeholder
        # TODO: Implement once ModelResponse cost tracking is available
        return None

    def _calculate_quality_indicators(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate quality indicators from evaluation results."""

        if not results:
            return {}

        quality_scores = []
        has_issues_count = 0

        for result in results:
            if result.quality_report:
                quality_scores.append(float(result.quality_report.quality_score))
                if not result.quality_report.is_passing():
                    has_issues_count += 1

        indicators = {
            "total_evaluations": len(results),
            "high_confidence_count": len([r for r in results if r.is_high_confidence()]),
            "low_confidence_count": len([r for r in results if r.is_low_confidence()]),
            "quality_issues_count": has_issues_count,
            "quality_pass_rate": 1.0 - (has_issues_count / len(results)) if results else 0.0,
        }

        if quality_scores:
            indicators.update(
                {
                    "average_quality_score": statistics.mean(quality_scores),
                    "min_quality_score": min(quality_scores),
                    "max_quality_score": max(quality_scores),
                }
            )

        return indicators

    def _calculate_confidence_interval(
        self, values: List[float], confidence_level: Decimal = None
    ) -> Optional[ConfidenceInterval]:
        """Calculate confidence interval for sample mean."""

        if len(values) < 2:
            return None

        if confidence_level is None:
            confidence_level = self._confidence_level

        try:
            import scipy.stats as stats

            mean = statistics.mean(values)
            se = statistics.stdev(values) / math.sqrt(len(values))

            alpha = float(1 - confidence_level)
            degrees_freedom = len(values) - 1

            t_critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
            margin_of_error = t_critical * se

            lower_bound = Decimal(str(mean - margin_of_error)).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            upper_bound = Decimal(str(mean + margin_of_error)).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            return ConfidenceInterval(
                lower_bound=lower_bound, upper_bound=upper_bound, confidence_level=confidence_level
            )

        except ImportError:
            # Fallback without scipy
            return None
        except Exception:
            return None

    def _calculate_quartiles(self, values: List[float]) -> List[float]:
        """Calculate quartiles (Q1, Q2, Q3)."""

        if len(values) < 4:
            return [min(values), statistics.median(values), max(values)]

        sorted_values = sorted(values)
        q1 = statistics.median(sorted_values[: len(sorted_values) // 2])
        q2 = statistics.median(sorted_values)
        q3 = statistics.median(sorted_values[(len(sorted_values) + 1) // 2 :])

        return [q1, q2, q3]

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate specific percentile."""

        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(math.floor(index))
            upper_index = int(math.ceil(index))
            weight = index - lower_index

            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def _get_time_window_key(self, timestamp, window: str) -> str:
        """Get time window key for temporal aggregation."""

        if window == "hour":
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif window == "day":
            return timestamp.strftime("%Y-%m-%d")
        elif window == "week":
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        else:
            return timestamp.strftime("%Y-%m-%d")

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend for time series data."""

        if len(values) < 3:
            return "stable"

        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * y for i, y in enumerate(values))
        x_squared_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

"""Test data aggregation engine."""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.analytics.entities.aggregation_rule import (
    AggregationRule,
    AggregationType,
    FilterCondition,
    FilterOperator,
    GroupByField,
    WeightingRule,
)
from src.domain.analytics.entities.analysis_result import AggregatedData
from src.domain.analytics.exceptions import (
    AggregationError,
    InvalidAggregationRule,
    MissingDataError,
    ValidationError,
)
from src.domain.analytics.services.data_aggregator import DataAggregator
from src.domain.evaluation.entities.evaluation_result import EvaluationResult


class TestAggregationRule:
    """Test aggregation rule entity."""

    def test_create_aggregation_rule(self):
        """Test creating a valid aggregation rule."""

        rule = AggregationRule.create(
            name="Mean Score by Model",
            description="Calculate mean overall score grouped by model",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
        )

        assert rule.name == "Mean Score by Model"
        assert rule.aggregation_type == AggregationType.MEAN
        assert rule.target_field == "overall_score"
        assert rule.is_active == True
        assert len(rule.group_by_fields) == 0
        assert len(rule.filter_conditions) == 0

    def test_aggregation_rule_with_grouping(self):
        """Test aggregation rule with grouping fields."""

        rule = AggregationRule.create(
            name="Score by Model and Difficulty",
            description="Mean score grouped by model and difficulty",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
            group_by_fields=[GroupByField.MODEL, GroupByField.DIFFICULTY],
        )

        assert len(rule.group_by_fields) == 2
        assert GroupByField.MODEL in rule.group_by_fields
        assert GroupByField.DIFFICULTY in rule.group_by_fields

    def test_aggregation_rule_with_filters(self):
        """Test aggregation rule with filter conditions."""

        filter_conditions = [
            FilterCondition("confidence_score", FilterOperator.GREATER_THAN, 0.8),
            FilterCondition("category", FilterOperator.IN, ["accuracy", "fluency"]),
        ]

        rule = AggregationRule.create(
            name="High Confidence Scores",
            description="Mean score for high confidence evaluations",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
            filter_conditions=filter_conditions,
        )

        assert len(rule.filter_conditions) == 2

        # Test filter application
        high_conf_data = {"confidence_score": 0.9, "category": "accuracy"}
        low_conf_data = {"confidence_score": 0.5, "category": "accuracy"}
        wrong_category_data = {"confidence_score": 0.9, "category": "coherence"}

        assert rule.applies_to_data(high_conf_data) == True
        assert rule.applies_to_data(low_conf_data) == False
        assert rule.applies_to_data(wrong_category_data) == False

    def test_percentile_aggregation_validation(self):
        """Test percentile aggregation parameter validation."""

        # Valid percentile
        rule = AggregationRule.create(
            name="95th Percentile",
            description="95th percentile of scores",
            aggregation_type=AggregationType.PERCENTILE,
            target_field="overall_score",
            parameters={"percentile": 95},
        )

        assert rule.parameters["percentile"] == 95

        # Invalid percentile - should raise error
        with pytest.raises(InvalidAggregationRule):
            AggregationRule.create(
                name="Invalid Percentile",
                description="Invalid percentile",
                aggregation_type=AggregationType.PERCENTILE,
                target_field="overall_score",
                parameters={"percentile": 150},  # Invalid: > 100
            )

    def test_weighted_mean_validation(self):
        """Test weighted mean aggregation validation."""

        weighting_rule = WeightingRule("confidence_score")

        rule = AggregationRule.create(
            name="Confidence Weighted Mean",
            description="Mean weighted by confidence",
            aggregation_type=AggregationType.WEIGHTED_MEAN,
            target_field="overall_score",
            weighting_rule=weighting_rule,
        )

        assert rule.weighting_rule.weight_field == "confidence_score"

        # Missing weighting rule should raise error
        with pytest.raises(InvalidAggregationRule):
            AggregationRule.create(
                name="Invalid Weighted Mean",
                description="Weighted mean without weighting rule",
                aggregation_type=AggregationType.WEIGHTED_MEAN,
                target_field="overall_score",
                # Missing weighting_rule
            )

    def test_filter_operators(self):
        """Test all filter operators."""

        test_cases = [
            (FilterOperator.EQUALS, "test_value", "test_value", True),
            (FilterOperator.EQUALS, "test_value", "other_value", False),
            (FilterOperator.NOT_EQUALS, "test_value", "other_value", True),
            (FilterOperator.GREATER_THAN, 10, 5, False),  # 5 > 10 is False
            (FilterOperator.GREATER_THAN, 5, 10, True),  # 10 > 5 is True
            (FilterOperator.LESS_THAN, 5, 10, False),  # 10 < 5 is False
            (FilterOperator.LESS_EQUAL, 10, 10, True),
            (FilterOperator.IN, [1, 2, 3], 2, True),
            (FilterOperator.IN, [1, 2, 3], 4, False),
            (FilterOperator.CONTAINS, "hello", "hello world", True),
            (FilterOperator.CONTAINS, "xyz", "hello world", False),
        ]

        for operator, condition_value, data_value, expected in test_cases:
            filter_condition = FilterCondition("test_field", operator, condition_value)
            rule = AggregationRule.create(
                name="Test Filter",
                description="Test filter operators",
                aggregation_type=AggregationType.COUNT,
                target_field="test_field",
                filter_conditions=[filter_condition],
            )

            data_row = {"test_field": data_value}
            result = rule.applies_to_data(data_row)
            assert (
                result == expected
            ), f"Failed for {operator} with {condition_value} vs {data_value}"

    def test_grouping_key_generation(self):
        """Test grouping key generation."""

        rule = AggregationRule.create(
            name="Test Grouping",
            description="Test grouping key generation",
            aggregation_type=AggregationType.MEAN,
            target_field="score",
            group_by_fields=[GroupByField.MODEL, GroupByField.CATEGORY],
        )

        data_row = {"model": "gpt-4", "category": "accuracy", "score": 0.85}

        key = rule.get_grouping_key(data_row)
        assert key == ("gpt-4", "accuracy")

        # Test with missing field
        incomplete_data = {"model": "gpt-4"}
        key_incomplete = rule.get_grouping_key(incomplete_data)
        assert key_incomplete == ("gpt-4", "unknown")


class TestDataAggregator:
    """Test data aggregation service."""

    def setup_method(self):
        """Set up test data."""
        self.aggregator = DataAggregator()

        # Create sample evaluation results
        self.evaluation_results = []

        for i in range(20):
            result = EvaluationResult.create_pending(
                judge_id=f"model_{i % 2}",  # Alternate between 2 models
                template_id=uuid4(),
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
            )

            # Complete the evaluation with mock data
            result.dimension_scores = {"accuracy": 4 + (i % 2), "fluency": 3 + (i % 3)}
            result.overall_score = Decimal(str(0.7 + (i % 3) * 0.1))
            result.confidence_score = Decimal(str(0.8 + (i % 2) * 0.1))
            result.reasoning = f"Test reasoning {i}"
            result.evaluation_time_ms = 1000 + i * 100
            result.completed_at = datetime.utcnow()

            # Add metadata
            result.metadata = {
                "difficulty_level": "easy" if i < 10 else "hard",
                "category": "accuracy" if i % 2 == 0 else "fluency",
            }

            self.evaluation_results.append(result)

    def test_basic_aggregation(self):
        """Test basic mean aggregation without grouping."""

        rule = AggregationRule.create(
            name="Overall Mean",
            description="Mean overall score",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
        )

        result = self.aggregator.aggregate_evaluation_results(self.evaluation_results, [rule])

        assert "Overall Mean" in result
        aggregated_data = result["Overall Mean"]
        assert len(aggregated_data) == 1  # Single group (no grouping)

        agg_item = aggregated_data[0]
        assert agg_item.sample_count == 20
        assert agg_item.group_key == ()

        # Verify calculated mean
        expected_mean = sum(float(r.overall_score) for r in self.evaluation_results) / len(
            self.evaluation_results
        )
        assert abs(float(agg_item.aggregated_value) - expected_mean) < 0.001

    def test_grouping_by_model(self):
        """Test aggregation with grouping by model."""

        rule = AggregationRule.create(
            name="Mean by Model",
            description="Mean score grouped by model",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
            group_by_fields=[GroupByField.MODEL],
        )

        result = self.aggregator.aggregate_evaluation_results(self.evaluation_results, [rule])

        aggregated_data = result["Mean by Model"]
        assert len(aggregated_data) == 2  # Two models

        # Check that both models are represented
        model_keys = {item.group_key[0] for item in aggregated_data}
        assert "model_0" in model_keys
        assert "model_1" in model_keys

        # Verify sample counts
        for item in aggregated_data:
            assert item.sample_count == 10  # 10 results per model

    def test_filtering(self):
        """Test aggregation with filtering."""

        filter_condition = FilterCondition("difficulty_level", FilterOperator.EQUALS, "easy")

        rule = AggregationRule.create(
            name="Easy Tasks Only",
            description="Mean score for easy tasks only",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
            filter_conditions=[filter_condition],
        )

        result = self.aggregator.aggregate_evaluation_results(self.evaluation_results, [rule])

        aggregated_data = result["Easy Tasks Only"]
        assert len(aggregated_data) == 1

        # Should only include easy tasks (first 10 results)
        assert aggregated_data[0].sample_count == 10

    def test_multiple_aggregation_types(self):
        """Test different aggregation types."""

        rules = [
            AggregationRule.create(
                name="Mean Score",
                description="Mean overall score",
                aggregation_type=AggregationType.MEAN,
                target_field="overall_score",
            ),
            AggregationRule.create(
                name="Max Score",
                description="Maximum overall score",
                aggregation_type=AggregationType.MAX,
                target_field="overall_score",
            ),
            AggregationRule.create(
                name="Count",
                description="Count of evaluations",
                aggregation_type=AggregationType.COUNT,
                target_field="overall_score",
            ),
            AggregationRule.create(
                name="90th Percentile",
                description="90th percentile of scores",
                aggregation_type=AggregationType.PERCENTILE,
                target_field="overall_score",
                parameters={"percentile": 90},
            ),
        ]

        result = self.aggregator.aggregate_evaluation_results(self.evaluation_results, rules)

        assert len(result) == 4

        # Verify count aggregation
        count_result = result["Count"][0]
        assert float(count_result.aggregated_value) == 20

        # Verify max is reasonable
        max_result = result["Max Score"][0]
        scores = [float(r.overall_score) for r in self.evaluation_results]
        assert float(max_result.aggregated_value) == max(scores)

    def test_model_performance_calculation(self):
        """Test comprehensive model performance calculation."""

        model_results = [r for r in self.evaluation_results if r.judge_id == "model_0"]

        performance = self.aggregator.calculate_model_performance(
            model_id="model_0", model_name="Test Model 0", evaluation_results=model_results
        )

        assert performance.model_id == "model_0"
        assert performance.model_name == "Test Model 0"
        assert performance.sample_count == 10

        # Check overall score calculation
        assert Decimal("0") <= performance.overall_score <= Decimal("1")

        # Check dimension scores
        assert "accuracy" in performance.dimension_scores
        assert "fluency" in performance.dimension_scores

        # Check quality indicators
        assert "total_evaluations" in performance.quality_indicators
        assert performance.quality_indicators["total_evaluations"] == 10

    def test_aggregation_by_difficulty(self):
        """Test aggregation by difficulty level."""

        result = self.aggregator.aggregate_by_difficulty_level(self.evaluation_results)

        assert "easy" in result
        assert "hard" in result

        easy_agg = result["easy"]
        hard_agg = result["hard"]

        assert easy_agg.sample_count == 10
        assert hard_agg.sample_count == 10

        # Check that aggregated values are reasonable
        assert Decimal("0") <= easy_agg.aggregated_value <= Decimal("1")
        assert Decimal("0") <= hard_agg.aggregated_value <= Decimal("1")

    def test_aggregation_by_category(self):
        """Test aggregation by category."""

        result = self.aggregator.aggregate_by_category(self.evaluation_results)

        assert "accuracy" in result
        assert "fluency" in result

        # Verify sample distribution
        accuracy_agg = result["accuracy"]
        fluency_agg = result["fluency"]

        assert accuracy_agg.sample_count == 10  # Even indices
        assert fluency_agg.sample_count == 10  # Odd indices

    def test_temporal_aggregation(self):
        """Test temporal aggregation."""

        # Modify timestamps to create temporal spread
        base_time = datetime.utcnow()
        for i, result in enumerate(self.evaluation_results):
            result.created_at = base_time + timedelta(hours=i)

        result = self.aggregator.aggregate_temporal_data(self.evaluation_results, time_window="day")

        # Should have aggregated by day
        assert len(result) >= 1  # At least one day

        for time_key, agg_data in result.items():
            assert agg_data.sample_count >= 1
            assert "time_window" in agg_data.group_labels

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""

        # Create data with known properties
        import numpy as np

        np.random.seed(42)

        # Generate evaluation results with normal distribution
        normal_results = []
        for i in range(50):
            result = EvaluationResult.create_pending(
                judge_id="model_test",
                template_id=uuid4(),
                prompt=f"Test {i}",
                response=f"Response {i}",
            )

            # Normal distribution around 0.8 with small variance
            score = max(0.0, min(1.0, np.random.normal(0.8, 0.1)))
            result.overall_score = Decimal(str(round(score, 3)))
            result.confidence_score = Decimal("0.9")
            result.completed_at = datetime.utcnow()
            result.metadata = {}

            normal_results.append(result)

        rule = AggregationRule.create(
            name="Test CI",
            description="Test confidence interval",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
        )

        result = self.aggregator.aggregate_evaluation_results(normal_results, [rule])

        agg_data = result["Test CI"][0]

        # Should have confidence interval
        assert agg_data.confidence_interval is not None
        ci_lower, ci_upper = agg_data.confidence_interval

        # Confidence interval should contain the mean
        assert ci_lower <= agg_data.aggregated_value <= ci_upper

        # CI should be reasonable width (not too wide or narrow)
        ci_width = ci_upper - ci_lower
        assert Decimal("0.001") <= ci_width <= Decimal("0.2")

    def test_error_handling(self):
        """Test error handling in aggregation."""

        # Empty results
        with pytest.raises(MissingDataError):
            self.aggregator.aggregate_evaluation_results([], [])

        # No aggregation rules
        with pytest.raises(InvalidAggregationRule):
            self.aggregator.aggregate_evaluation_results(self.evaluation_results, [])

        # Invalid evaluation results (incomplete)
        incomplete_result = EvaluationResult.create_pending(
            judge_id="test", template_id=uuid4(), prompt="test", response="test"
        )

        rule = AggregationRule.create(
            name="Test",
            description="Test rule",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
        )

        with pytest.raises(ValidationError):
            self.aggregator.aggregate_evaluation_results([incomplete_result], [rule])

    def test_inactive_rule_handling(self):
        """Test handling of inactive aggregation rules."""

        active_rule = AggregationRule.create(
            name="Active Rule",
            description="Active aggregation rule",
            aggregation_type=AggregationType.MEAN,
            target_field="overall_score",
        )

        inactive_rule = AggregationRule.create(
            name="Inactive Rule",
            description="Inactive aggregation rule",
            aggregation_type=AggregationType.MAX,
            target_field="overall_score",
        )
        inactive_rule.deactivate()

        result = self.aggregator.aggregate_evaluation_results(
            self.evaluation_results, [active_rule, inactive_rule]
        )

        # Only active rule should produce results
        assert "Active Rule" in result
        assert "Inactive Rule" not in result

    def test_missing_target_field_handling(self):
        """Test handling when target field is missing from data."""

        # Create evaluation results missing the target field
        incomplete_results = []
        for i in range(5):
            result = EvaluationResult.create_pending(
                judge_id="model_test",
                template_id=uuid4(),
                prompt=f"Test {i}",
                response=f"Response {i}",
            )
            # Complete but don't set dimension scores - target field won't exist
            result.overall_score = Decimal("0.8")
            result.confidence_score = Decimal("0.9")
            result.completed_at = datetime.utcnow()
            result.metadata = {}

            incomplete_results.append(result)

        rule = AggregationRule.create(
            name="Missing Field Test",
            description="Test missing target field",
            aggregation_type=AggregationType.MEAN,
            target_field="nonexistent_field",
        )

        result = self.aggregator.aggregate_evaluation_results(incomplete_results, [rule])

        # Should return empty results for missing field
        assert "Missing Field Test" in result
        assert len(result["Missing Field Test"]) == 0

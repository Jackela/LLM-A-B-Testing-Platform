"""Aggregation rule entity for data processing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import InvalidAggregationRule, ValidationError


class AggregationType(Enum):
    """Types of data aggregation."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "standard_deviation"
    VAR = "variance"
    PERCENTILE = "percentile"
    QUARTILE = "quartile"
    RANGE = "range"
    IQR = "interquartile_range"
    WEIGHTED_MEAN = "weighted_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"


class GroupByField(Enum):
    """Fields available for grouping data."""

    MODEL = "model"
    DIFFICULTY = "difficulty_level"
    CATEGORY = "category"
    JUDGE = "judge_id"
    DIMENSION = "dimension"
    TIME_PERIOD = "time_period"
    TEMPLATE = "template_id"
    CONFIDENCE_LEVEL = "confidence_level"
    QUALITY_LEVEL = "quality_level"


class FilterOperator(Enum):
    """Filter operators for data filtering."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass(frozen=True)
class FilterCondition:
    """Filter condition for data aggregation."""

    field: str
    operator: FilterOperator
    value: Any

    def __post_init__(self):
        """Validate filter condition."""
        if not self.field.strip():
            raise ValidationError("Filter field cannot be empty")


@dataclass(frozen=True)
class WeightingRule:
    """Weighting rule for weighted aggregations."""

    weight_field: str
    weight_function: Optional[str] = None  # Custom weighting function name
    normalize_weights: bool = True

    def __post_init__(self):
        """Validate weighting rule."""
        if not self.weight_field.strip():
            raise ValidationError("Weight field cannot be empty")


@dataclass
class AggregationRule:
    """Aggregation rule entity defining how to aggregate data."""

    rule_id: UUID
    name: str
    description: str
    aggregation_type: AggregationType
    target_field: str
    group_by_fields: List[GroupByField]
    filter_conditions: List[FilterCondition]
    weighting_rule: Optional[WeightingRule] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: Optional[str] = None
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate aggregation rule after creation."""
        if not self.rule_id:
            self.rule_id = uuid4()

        self._validate_rule()

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        aggregation_type: AggregationType,
        target_field: str,
        group_by_fields: Optional[List[GroupByField]] = None,
        filter_conditions: Optional[List[FilterCondition]] = None,
        weighting_rule: Optional[WeightingRule] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "AggregationRule":
        """Factory method to create aggregation rule."""

        return cls(
            rule_id=uuid4(),
            name=name,
            description=description,
            aggregation_type=aggregation_type,
            target_field=target_field,
            group_by_fields=group_by_fields or [],
            filter_conditions=filter_conditions or [],
            weighting_rule=weighting_rule,
            parameters=parameters or {},
            is_active=True,
        )

    def _validate_rule(self) -> None:
        """Validate aggregation rule properties."""
        if not self.name.strip():
            raise ValidationError("Aggregation rule name cannot be empty")

        if not self.target_field.strip():
            raise ValidationError("Target field cannot be empty")

        # Validate type-specific parameters
        if self.aggregation_type == AggregationType.PERCENTILE:
            if "percentile" not in self.parameters:
                raise InvalidAggregationRule(
                    "Percentile aggregation requires 'percentile' parameter"
                )

            percentile = self.parameters["percentile"]
            if not (0 <= percentile <= 100):
                raise InvalidAggregationRule("Percentile must be between 0 and 100")

        elif self.aggregation_type == AggregationType.QUARTILE:
            if "quartile" not in self.parameters:
                raise InvalidAggregationRule("Quartile aggregation requires 'quartile' parameter")

            quartile = self.parameters["quartile"]
            if quartile not in [1, 2, 3]:
                raise InvalidAggregationRule("Quartile must be 1, 2, or 3")

        elif self.aggregation_type == AggregationType.WEIGHTED_MEAN:
            if not self.weighting_rule:
                raise InvalidAggregationRule("Weighted mean requires weighting rule")

        # Validate filter conditions
        for condition in self.filter_conditions:
            if condition.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                if not isinstance(condition.value, (list, tuple, set)):
                    raise InvalidAggregationRule(
                        f"Filter operator {condition.operator.value} requires list/tuple/set value"
                    )

    def applies_to_data(self, data_row: Dict[str, Any]) -> bool:
        """Check if rule applies to given data row based on filter conditions."""

        if not self.is_active:
            return False

        for condition in self.filter_conditions:
            if not self._evaluate_filter_condition(condition, data_row):
                return False

        return True

    def _evaluate_filter_condition(
        self, condition: FilterCondition, data_row: Dict[str, Any]
    ) -> bool:
        """Evaluate single filter condition."""

        field_value = data_row.get(condition.field)

        if field_value is None:
            return False  # Missing field fails condition

        if condition.operator == FilterOperator.EQUALS:
            return field_value == condition.value
        elif condition.operator == FilterOperator.NOT_EQUALS:
            return field_value != condition.value
        elif condition.operator == FilterOperator.GREATER_THAN:
            return field_value > condition.value
        elif condition.operator == FilterOperator.GREATER_EQUAL:
            return field_value >= condition.value
        elif condition.operator == FilterOperator.LESS_THAN:
            return field_value < condition.value
        elif condition.operator == FilterOperator.LESS_EQUAL:
            return field_value <= condition.value
        elif condition.operator == FilterOperator.IN:
            return field_value in condition.value
        elif condition.operator == FilterOperator.NOT_IN:
            return field_value not in condition.value
        elif condition.operator == FilterOperator.CONTAINS:
            return str(condition.value) in str(field_value)
        elif condition.operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(condition.value))
        elif condition.operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(condition.value))

        return False

    def get_grouping_key(self, data_row: Dict[str, Any]) -> tuple:
        """Generate grouping key for data row based on group_by_fields."""

        if not self.group_by_fields:
            return ()  # No grouping, single group

        key_parts = []
        for field in self.group_by_fields:
            field_name = field.value
            value = data_row.get(field_name, "unknown")
            key_parts.append(str(value))

        return tuple(key_parts)

    def get_target_value(self, data_row: Dict[str, Any]) -> Optional[float]:
        """Extract target value from data row."""

        value = data_row.get(self.target_field)

        if value is None:
            return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def get_weight_value(self, data_row: Dict[str, Any]) -> Optional[float]:
        """Extract weight value from data row if weighting rule exists."""

        if not self.weighting_rule:
            return None

        weight = data_row.get(self.weighting_rule.weight_field)

        if weight is None:
            return None

        try:
            return float(weight)
        except (ValueError, TypeError):
            return None

    def deactivate(self) -> None:
        """Deactivate the aggregation rule."""
        self.is_active = False

    def activate(self) -> None:
        """Activate the aggregation rule."""
        self.is_active = True

    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update rule parameters."""
        old_parameters = self.parameters.copy()
        self.parameters.update(new_parameters)

        try:
            self._validate_rule()
        except Exception:
            # Rollback on validation failure
            self.parameters = old_parameters
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": str(self.rule_id),
            "name": self.name,
            "description": self.description,
            "aggregation_type": self.aggregation_type.value,
            "target_field": self.target_field,
            "group_by_fields": [field.value for field in self.group_by_fields],
            "filter_conditions": [
                {"field": cond.field, "operator": cond.operator.value, "value": cond.value}
                for cond in self.filter_conditions
            ],
            "weighting_rule": (
                {
                    "weight_field": self.weighting_rule.weight_field,
                    "weight_function": self.weighting_rule.weight_function,
                    "normalize_weights": self.weighting_rule.normalize_weights,
                }
                if self.weighting_rule
                else None
            ),
            "parameters": self.parameters.copy(),
            "is_active": self.is_active,
            "created_at": self.created_at,
        }

    def __str__(self) -> str:
        """String representation."""
        group_by_str = (
            ", ".join([field.value for field in self.group_by_fields])
            if self.group_by_fields
            else "none"
        )
        filters_str = (
            f"{len(self.filter_conditions)} filters" if self.filter_conditions else "no filters"
        )

        return (
            f"AggregationRule(name='{self.name}', "
            f"type={self.aggregation_type.value}, "
            f"target={self.target_field}, "
            f"group_by=[{group_by_str}], "
            f"{filters_str})"
        )

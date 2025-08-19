"""Evaluation template entity."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..exceptions import InvalidScore, MissingDimensionScore, TemplateRenderError, ValidationError
from ..value_objects.scoring_scale import ScoringScale
from .dimension import Dimension


@dataclass
class EvaluationTemplate:
    """Entity representing an evaluation template for structured judging."""

    template_id: UUID
    name: str
    description: str
    dimensions: List[Dimension]
    prompt_template: str
    scoring_scale: ScoringScale
    judge_model_id: str
    model_parameters: Dict[str, Any]
    version: int = 1
    is_active: bool = True
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _domain_events: List[object] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Validate template after creation."""
        if not self.template_id:
            self.template_id = uuid4()

        self._validate_template()

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        dimensions: List[Dimension],
        prompt_template: str,
        scoring_scale: ScoringScale,
        judge_model_id: str,
        model_parameters: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvaluationTemplate":
        """Factory method to create evaluation template."""
        template = cls(
            template_id=uuid4(),
            name=name,
            description=description,
            dimensions=dimensions,
            prompt_template=prompt_template,
            scoring_scale=scoring_scale,
            judge_model_id=judge_model_id,
            model_parameters=model_parameters or {},
            created_by=created_by,
            metadata=metadata or {},
        )

        # Add domain event
        from datetime import datetime

        from ..events.evaluation_events import EvaluationTemplateCreated

        event = EvaluationTemplateCreated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=template.template_id,
            template_name=name,
            dimensions_count=len(dimensions),
            created_by=created_by,
        )
        template._domain_events.append(event)

        return template

    def _validate_template(self) -> None:
        """Validate template structure and requirements."""
        if not self.name.strip():
            raise ValidationError("Template name cannot be empty")

        if not self.description.strip():
            raise ValidationError("Template description cannot be empty")

        if not self.dimensions:
            raise ValidationError("Template must have at least one dimension")

        if not self.prompt_template.strip():
            raise ValidationError("Prompt template cannot be empty")

        if not self.judge_model_id.strip():
            raise ValidationError("Judge model ID cannot be empty")

        # Validate dimension weights sum to 1
        total_weight = sum(d.weight for d in self.dimensions)
        if abs(total_weight - 1) > Decimal("0.001"):
            raise ValidationError(f"Dimension weights must sum to 1.0, got {total_weight}")

        # Check for duplicate dimension names
        dimension_names = [d.name for d in self.dimensions]
        if len(dimension_names) != len(set(dimension_names)):
            raise ValidationError("Dimensions must have unique names")

        # Validate prompt template has required placeholders
        required_placeholders = ["{prompt}", "{response}"]
        for placeholder in required_placeholders:
            if placeholder not in self.prompt_template:
                raise ValidationError(f"Template missing required placeholder: {placeholder}")

        # Validate model parameters
        if not isinstance(self.model_parameters, dict):
            raise ValidationError("Model parameters must be a dictionary")

        # Validate version
        if self.version < 1:
            raise ValidationError("Template version must be >= 1")

    def render(self, prompt: str, response: str, **context) -> str:
        """Render evaluation prompt with context."""
        if not prompt.strip():
            raise TemplateRenderError("Prompt cannot be empty")

        if not response.strip():
            raise TemplateRenderError("Response cannot be empty")

        # Prepare template context
        template_context = {
            "prompt": prompt,
            "response": response,
            "dimensions": self._format_dimensions_for_template(),
            "scoring_scale": self._format_scoring_scale_for_template(),
            **context,
        }

        try:
            rendered = self.prompt_template.format(**template_context)
            return rendered.strip()
        except KeyError as e:
            raise TemplateRenderError(f"Missing context variable: {e}")
        except Exception as e:
            raise TemplateRenderError(f"Template rendering failed: {e}")

    def _format_dimensions_for_template(self) -> str:
        """Format dimensions for template inclusion."""
        dimension_text = []

        for dim in self.dimensions:
            text = f"**{dim.name.upper()}** (Weight: {dim.weight})\n"
            text += f"{dim.description}\n"
            text += "Scoring criteria:\n"

            for score, description in sorted(dim.scoring_criteria.items()):
                text += f"  {score}: {description}\n"

            dimension_text.append(text)

        return "\n".join(dimension_text)

    def _format_scoring_scale_for_template(self) -> str:
        """Format scoring scale for template inclusion."""
        scale_text = f"Scoring Scale: {self.scoring_scale.description}\n"
        scale_text += f"Range: {self.scoring_scale.min_score} to {self.scoring_scale.max_score}\n"

        if self.scoring_scale.scale_type in ["discrete", "likert"]:
            scale_text += "Valid scores: " + ", ".join(
                str(v) for v in self.scoring_scale.get_discrete_values()
            )

        return scale_text

    def calculate_weighted_score(self, dimension_scores: Dict[str, int]) -> Decimal:
        """Calculate weighted average score across dimensions."""
        if not dimension_scores:
            raise MissingDimensionScore("No dimension scores provided")

        total_score = Decimal("0")

        for dimension in self.dimensions:
            if dimension.name not in dimension_scores:
                if dimension.is_required:
                    raise MissingDimensionScore(
                        f"Missing score for required dimension: {dimension.name}"
                    )
                continue

            raw_score = dimension_scores[dimension.name]

            # Validate score is valid for dimension
            if not dimension.is_valid_score(raw_score):
                raise InvalidScore(f"Invalid score {raw_score} for dimension {dimension.name}")

            # Validate score is valid for scoring scale
            normalized_score = dimension.normalize_score(raw_score)
            denormalized_for_scale = self.scoring_scale.denormalize_score(normalized_score)

            if not self.scoring_scale.is_valid_score(denormalized_for_scale):
                raise InvalidScore(f"Score {raw_score} not valid for scoring scale")

            # Add weighted score
            weighted_score = dimension.calculate_weighted_score(raw_score)
            total_score += weighted_score

        return total_score.quantize(Decimal("0.001"))

    def get_dimension(self, name: str) -> Optional[Dimension]:
        """Get dimension by name."""
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        return None

    def has_dimension(self, name: str) -> bool:
        """Check if template has dimension with given name."""
        return self.get_dimension(name) is not None

    def get_required_dimensions(self) -> List[Dimension]:
        """Get list of required dimensions."""
        return [d for d in self.dimensions if d.is_required]

    def get_optional_dimensions(self) -> List[Dimension]:
        """Get list of optional dimensions."""
        return [d for d in self.dimensions if not d.is_required]

    def add_dimension(self, dimension: Dimension) -> None:
        """Add dimension to template."""
        if self.has_dimension(dimension.name):
            raise ValidationError(f"Dimension '{dimension.name}' already exists")

        self.dimensions.append(dimension)
        self._rebalance_weights()
        self._validate_template()

        # Add domain event
        from datetime import datetime

        from ..events.evaluation_events import EvaluationTemplateModified

        event = EvaluationTemplateModified(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.template_id,
            modification_type="dimension_added",
            details={"dimension_name": dimension.name},
        )
        self._domain_events.append(event)

    def remove_dimension(self, name: str) -> None:
        """Remove dimension from template."""
        dimension = self.get_dimension(name)
        if not dimension:
            raise ValidationError(f"Dimension '{name}' not found")

        if len(self.dimensions) <= 1:
            raise ValidationError("Template must have at least one dimension")

        self.dimensions.remove(dimension)
        self._rebalance_weights()
        self._validate_template()

        # Add domain event
        from datetime import datetime

        from ..events.evaluation_events import EvaluationTemplateModified

        event = EvaluationTemplateModified(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.template_id,
            modification_type="dimension_removed",
            details={"dimension_name": name},
        )
        self._domain_events.append(event)

    def _rebalance_weights(self) -> None:
        """Rebalance dimension weights to sum to 1.0."""
        if not self.dimensions:
            return

        equal_weight = Decimal("1.0") / len(self.dimensions)
        for dimension in self.dimensions:
            dimension.update_weight(equal_weight)

    def update_dimension_weight(self, name: str, new_weight: Decimal) -> None:
        """Update weight for specific dimension and rebalance others."""
        dimension = self.get_dimension(name)
        if not dimension:
            raise ValidationError(f"Dimension '{name}' not found")

        if not (0 < new_weight < 1):
            raise ValidationError("Dimension weight must be between 0 and 1")

        # Calculate total weight of other dimensions
        other_dimensions = [d for d in self.dimensions if d.name != name]
        remaining_weight = Decimal("1.0") - new_weight

        if len(other_dimensions) == 0:
            raise ValidationError("Cannot adjust weights with only one dimension")

        # Update target dimension
        dimension.update_weight(new_weight)

        # Redistribute remaining weight among other dimensions
        equal_weight = remaining_weight / len(other_dimensions)
        for dim in other_dimensions:
            dim.update_weight(equal_weight)

        self._validate_template()

    def create_new_version(self, modified_by: Optional[str] = None) -> "EvaluationTemplate":
        """Create new version of template."""
        new_template = EvaluationTemplate(
            template_id=uuid4(),
            name=self.name,
            description=self.description,
            dimensions=[
                Dimension(
                    dimension_id=uuid4(),
                    name=d.name,
                    description=d.description,
                    weight=d.weight,
                    scoring_criteria=d.scoring_criteria.copy(),
                    is_required=d.is_required,
                    metadata=d.metadata.copy(),
                )
                for d in self.dimensions
            ],
            prompt_template=self.prompt_template,
            scoring_scale=self.scoring_scale,
            judge_model_id=self.judge_model_id,
            model_parameters=self.model_parameters.copy(),
            version=self.version + 1,
            is_active=True,
            created_by=modified_by,
            metadata=self.metadata.copy(),
        )

        # Deactivate current version
        self.is_active = False

        return new_template

    def deactivate(self) -> None:
        """Deactivate template."""
        self.is_active = False

        # Add domain event
        from datetime import datetime

        from ..events.evaluation_events import EvaluationTemplateDeactivated

        event = EvaluationTemplateDeactivated(
            occurred_at=datetime.utcnow(),
            event_id=uuid4(),
            aggregate_id=self.template_id,
            template_name=self.name,
            version=self.version,
        )
        self._domain_events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "template_id": str(self.template_id),
            "name": self.name,
            "description": self.description,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "prompt_template": self.prompt_template,
            "scoring_scale": self.scoring_scale.to_dict(),
            "judge_model_id": self.judge_model_id,
            "model_parameters": self.model_parameters.copy(),
            "version": self.version,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "metadata": self.metadata.copy(),
            "dimension_count": len(self.dimensions),
            "required_dimensions": len(self.get_required_dimensions()),
            "total_weight": str(sum(d.weight for d in self.dimensions)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTemplate":
        """Create from dictionary representation."""
        dimensions = [Dimension.from_dict(d) for d in data["dimensions"]]
        scoring_scale = ScoringScale.from_dict(data["scoring_scale"])

        return cls(
            template_id=UUID(data["template_id"]),
            name=data["name"],
            description=data["description"],
            dimensions=dimensions,
            prompt_template=data["prompt_template"],
            scoring_scale=scoring_scale,
            judge_model_id=data["judge_model_id"],
            model_parameters=data["model_parameters"],
            version=data.get("version", 1),
            is_active=data.get("is_active", True),
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"EvaluationTemplate(name='{self.name}', "
            f"version={self.version}, "
            f"dimensions={len(self.dimensions)})"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on ID."""
        if not isinstance(other, EvaluationTemplate):
            return False
        return self.template_id == other.template_id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.template_id)

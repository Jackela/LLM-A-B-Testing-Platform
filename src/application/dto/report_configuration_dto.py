"""Data Transfer Objects for report configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID


class ReportFormat(Enum):
    """Supported report formats."""

    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportType(Enum):
    """Types of reports that can be generated."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    STATISTICAL_REPORT = "statistical_report"
    MODEL_COMPARISON = "model_comparison"
    COST_ANALYSIS = "cost_analysis"
    INSIGHTS_REPORT = "insights_report"
    CUSTOM = "custom"


@dataclass
class ReportConfigurationDTO:
    """DTO for report generation configuration."""

    report_type: ReportType = ReportType.DETAILED_ANALYSIS
    format: ReportFormat = ReportFormat.HTML
    title: Optional[str] = None
    include_statistical_details: bool = True
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_cost_analysis: bool = True
    include_recommendations: bool = True
    include_methodology: bool = False
    save_to_file: bool = False
    file_path: Optional[str] = None
    focus_models: Optional[List[str]] = None
    custom_sections: Optional[List[str]] = None
    min_insight_confidence: float = 0.7
    max_insights_per_section: int = 5
    branding: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not (0.1 <= self.min_insight_confidence <= 1.0):
            raise ValueError("Minimum insight confidence must be between 0.1 and 1.0")

        if self.max_insights_per_section < 1:
            raise ValueError("Maximum insights per section must be at least 1")

        if self.focus_models is None:
            self.focus_models = []

        if self.custom_sections is None:
            self.custom_sections = []

        if self.branding is None:
            self.branding = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_type": self.report_type.value,
            "format": self.format.value,
            "title": self.title,
            "include_statistical_details": self.include_statistical_details,
            "include_visualizations": self.include_visualizations,
            "include_raw_data": self.include_raw_data,
            "include_cost_analysis": self.include_cost_analysis,
            "include_recommendations": self.include_recommendations,
            "include_methodology": self.include_methodology,
            "save_to_file": self.save_to_file,
            "file_path": self.file_path,
            "focus_models": self.focus_models,
            "custom_sections": self.custom_sections,
            "min_insight_confidence": self.min_insight_confidence,
            "max_insights_per_section": self.max_insights_per_section,
            "branding": self.branding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportConfigurationDTO":
        """Create DTO from dictionary."""
        return cls(
            report_type=ReportType(data.get("report_type", "detailed_analysis")),
            format=ReportFormat(data.get("format", "html")),
            title=data.get("title"),
            include_statistical_details=data.get("include_statistical_details", True),
            include_visualizations=data.get("include_visualizations", True),
            include_raw_data=data.get("include_raw_data", False),
            include_cost_analysis=data.get("include_cost_analysis", True),
            include_recommendations=data.get("include_recommendations", True),
            include_methodology=data.get("include_methodology", False),
            save_to_file=data.get("save_to_file", False),
            file_path=data.get("file_path"),
            focus_models=data.get("focus_models", []),
            custom_sections=data.get("custom_sections", []),
            min_insight_confidence=data.get("min_insight_confidence", 0.7),
            max_insights_per_section=data.get("max_insights_per_section", 5),
            branding=data.get("branding", {}),
        )

    @classmethod
    def create_executive_summary(cls, title: Optional[str] = None) -> "ReportConfigurationDTO":
        """Create configuration for executive summary report."""
        return cls(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            title=title or "Executive Summary",
            include_statistical_details=False,
            include_visualizations=True,
            include_raw_data=False,
            include_recommendations=True,
            include_methodology=False,
        )

    @classmethod
    def create_detailed_analysis(cls, title: Optional[str] = None) -> "ReportConfigurationDTO":
        """Create configuration for detailed analysis report."""
        return cls(
            report_type=ReportType.DETAILED_ANALYSIS,
            format=ReportFormat.HTML,
            title=title or "Detailed Analysis Report",
            include_statistical_details=True,
            include_visualizations=True,
            include_raw_data=True,
            include_recommendations=True,
            include_methodology=True,
        )

    @classmethod
    def create_model_comparison(
        cls, focus_models: List[str], title: Optional[str] = None
    ) -> "ReportConfigurationDTO":
        """Create configuration for model comparison report."""
        return cls(
            report_type=ReportType.MODEL_COMPARISON,
            format=ReportFormat.HTML,
            title=title or "Model Comparison Report",
            include_statistical_details=True,
            include_visualizations=True,
            include_cost_analysis=True,
            focus_models=focus_models,
        )

    @classmethod
    def create_cost_analysis(cls, title: Optional[str] = None) -> "ReportConfigurationDTO":
        """Create configuration for cost analysis report."""
        return cls(
            report_type=ReportType.COST_ANALYSIS,
            format=ReportFormat.HTML,
            title=title or "Cost Analysis Report",
            include_statistical_details=False,
            include_visualizations=True,
            include_cost_analysis=True,
            include_recommendations=True,
        )


@dataclass
class VisualizationConfigDTO:
    """DTO for visualization configuration."""

    chart_types: List[str]  # e.g., ["bar", "scatter", "heatmap"]
    color_scheme: str = "default"
    include_interactive: bool = True
    width: int = 800
    height: int = 600
    export_formats: Optional[List[str]] = None  # e.g., ["png", "svg", "html"]
    show_data_labels: bool = True
    show_legend: bool = True
    custom_styling: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if not self.chart_types:
            raise ValueError("At least one chart type required")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")

        if self.export_formats is None:
            object.__setattr__(self, "export_formats", ["html"])

        if self.custom_styling is None:
            object.__setattr__(self, "custom_styling", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chart_types": self.chart_types,
            "color_scheme": self.color_scheme,
            "include_interactive": self.include_interactive,
            "width": self.width,
            "height": self.height,
            "export_formats": self.export_formats,
            "show_data_labels": self.show_data_labels,
            "show_legend": self.show_legend,
            "custom_styling": self.custom_styling,
        }


@dataclass
class ReportRequestDTO:
    """DTO for report generation request."""

    test_id: UUID
    analysis_id: UUID
    configuration: ReportConfigurationDTO
    visualization_config: Optional[VisualizationConfigDTO] = None
    requested_by: Optional[str] = None
    delivery_options: Optional[Dict[str, Any]] = None  # Email, file system, etc.

    def __post_init__(self) -> None:
        """Validate DTO after initialization."""
        if self.delivery_options is None:
            self.delivery_options = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_id": str(self.test_id),
            "analysis_id": str(self.analysis_id),
            "configuration": self.configuration.to_dict(),
            "visualization_config": (
                self.visualization_config.to_dict() if self.visualization_config else None
            ),
            "requested_by": self.requested_by,
            "delivery_options": self.delivery_options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportRequestDTO":
        """Create DTO from dictionary."""
        config = ReportConfigurationDTO.from_dict(data["configuration"])

        viz_config = None
        if data.get("visualization_config"):
            viz_config = VisualizationConfigDTO(**data["visualization_config"])

        return cls(
            test_id=UUID(data["test_id"]),
            analysis_id=UUID(data["analysis_id"]),
            configuration=config,
            visualization_config=viz_config,
            requested_by=data.get("requested_by"),
            delivery_options=data.get("delivery_options", {}),
        )


@dataclass
class ReportResponseDTO:
    """DTO for report generation response."""

    report_id: UUID
    test_id: UUID
    analysis_id: UUID
    status: str  # "completed", "in_progress", "failed"
    report_type: str
    format: str
    created_at: str  # ISO format datetime
    completed_at: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_url: Optional[str] = None
    preview_url: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": str(self.report_id),
            "test_id": str(self.test_id),
            "analysis_id": str(self.analysis_id),
            "status": self.status,
            "report_type": self.report_type,
            "format": self.format,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "file_size_bytes": self.file_size_bytes,
            "download_url": self.download_url,
            "preview_url": self.preview_url,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @property
    def is_completed(self) -> bool:
        """Check if report generation is completed."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if report generation failed."""
        return self.status == "failed"

    @property
    def is_in_progress(self) -> bool:
        """Check if report generation is in progress."""
        return self.status == "in_progress"

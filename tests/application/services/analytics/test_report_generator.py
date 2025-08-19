"""Tests for Report Generator Service."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.application.dto.report_configuration_dto import ReportConfigurationDTO
from src.application.services.analytics.report_generator import (
    GeneratedReport,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    ReportType,
)
from src.domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from src.domain.analytics.exceptions import ReportGenerationError, ValidationError


class TestReportGenerator:
    """Test suite for Report Generator."""

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories and services."""
        test_repository = AsyncMock()
        analytics_repository = AsyncMock()
        visualization_service = AsyncMock()

        return {
            "test_repository": test_repository,
            "analytics_repository": analytics_repository,
            "visualization_service": visualization_service,
        }

    @pytest.fixture
    def generator(self, mock_repositories):
        """Create generator instance with mocked dependencies."""
        return ReportGenerator(**mock_repositories)

    @pytest.fixture
    def sample_test(self):
        """Create sample test entity."""
        test = Mock()
        test.test_id = uuid4()
        test.name = "Sample Test"
        test.description = "Test description"
        return test

    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample analysis result."""
        analysis_result = Mock(spec=AnalysisResult)
        analysis_result.analysis_id = uuid4()
        analysis_result.test_id = uuid4()
        analysis_result.name = "Sample Analysis"
        analysis_result.description = "Analysis description"
        analysis_result.created_at = datetime.utcnow()
        analysis_result.completed_at = datetime.utcnow()
        analysis_result.is_completed.return_value = True

        # Mock model performances
        model_performances = {}
        for i in range(3):
            model = Mock(spec=ModelPerformanceMetrics)
            model.model_id = f"model_{i}"
            model.model_name = f"Model {i}"
            model.overall_score = Decimal(str(0.8 - i * 0.1))
            model.dimension_scores = {
                "accuracy": Decimal(str(0.85 - i * 0.05)),
                "fluency": Decimal(str(0.8 - i * 0.08)),
            }
            model.sample_count = 100 + i * 20
            model.confidence_score = Decimal(str(0.9 - i * 0.05))
            model.cost_metrics = None  # Will be set in specific tests
            model.quality_indicators = {"error_rate": 0.01 + i * 0.005}
            model_performances[f"model_{i}"] = model

        analysis_result.model_performances = model_performances
        analysis_result.statistical_tests = {}
        analysis_result.insights = []
        analysis_result.metadata = {"processing_time_ms": 5000}

        # Mock methods
        analysis_result.get_best_performing_model.return_value = model_performances["model_0"]
        analysis_result.get_most_cost_effective_model.return_value = None
        analysis_result.get_total_sample_count.return_value = 360
        analysis_result.get_significant_tests.return_value = {}
        analysis_result.get_actionable_insights.return_value = []
        analysis_result.get_high_confidence_insights.return_value = []
        analysis_result.get_model_comparison_summary.return_value = {
            "total_models": 3,
            "best_model": {"id": "model_0", "name": "Model 0", "score": "0.8"},
            "worst_model": {"id": "model_2", "name": "Model 2", "score": "0.6"},
            "performance_spread": "0.2",
            "significant_differences": 0,
            "total_statistical_tests": 0,
            "actionable_insights": 0,
            "high_confidence_insights": 0,
        }

        return analysis_result

    @pytest.fixture
    def report_config(self):
        """Create sample report configuration."""
        return ReportConfigurationDTO(
            report_type=ReportType.DETAILED_ANALYSIS,
            format=ReportFormat.HTML,
            title="Test Report",
            include_statistical_details=True,
            include_visualizations=True,
            include_recommendations=True,
        )

    @pytest.mark.asyncio
    async def test_generate_report_success(
        self, generator, mock_repositories, sample_test, sample_analysis_result, report_config
    ):
        """Test successful report generation."""
        test_id = uuid4()

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["visualization_service"].create_summary_dashboard.return_value = (
            "<div>Chart</div>"
        )
        mock_repositories["visualization_service"].create_model_comparison_chart.return_value = (
            "<div>Chart</div>"
        )
        mock_repositories["visualization_service"].create_statistical_results_chart.return_value = (
            "<div>Chart</div>"
        )

        # Execute
        result = await generator.generate_report(test_id, sample_analysis_result, report_config)

        # Verify
        assert isinstance(result, GeneratedReport)
        assert result.test_id == test_id
        assert result.report_type == ReportType.DETAILED_ANALYSIS
        assert result.format == ReportFormat.HTML
        assert result.title == "Test Report"
        assert len(result.sections) > 0
        assert result.content is not None
        assert result.metadata["total_sections"] == len(result.sections)

        # Verify repository calls
        mock_repositories["test_repository"].find_by_id.assert_called_once_with(test_id)

    @pytest.mark.asyncio
    async def test_generate_report_incomplete_analysis(
        self, generator, sample_test, sample_analysis_result, report_config
    ):
        """Test report generation with incomplete analysis."""
        test_id = uuid4()

        # Make analysis incomplete
        sample_analysis_result.is_completed.return_value = False

        # Execute and verify exception
        with pytest.raises(ValidationError, match="Analysis must be completed"):
            await generator.generate_report(test_id, sample_analysis_result, report_config)

    @pytest.mark.asyncio
    async def test_generate_executive_summary(
        self, generator, mock_repositories, sample_test, sample_analysis_result
    ):
        """Test executive summary generation."""
        test_id = uuid4()

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["visualization_service"].create_summary_dashboard.return_value = (
            "<div>Chart</div>"
        )

        # Execute
        result = await generator.generate_executive_summary(test_id, sample_analysis_result)

        # Verify
        assert isinstance(result, GeneratedReport)
        assert result.report_type == ReportType.EXECUTIVE_SUMMARY
        assert not any(
            "statistical_analysis" in section.section_type for section in result.sections
        )
        assert any("executive_summary" in section.section_type for section in result.sections)

    @pytest.mark.asyncio
    async def test_generate_detailed_analysis(
        self, generator, mock_repositories, sample_test, sample_analysis_result
    ):
        """Test detailed analysis report generation."""
        test_id = uuid4()

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["visualization_service"].create_summary_dashboard.return_value = (
            "<div>Chart</div>"
        )
        mock_repositories["visualization_service"].create_model_comparison_chart.return_value = (
            "<div>Chart</div>"
        )

        # Execute
        result = await generator.generate_detailed_analysis(test_id, sample_analysis_result)

        # Verify
        assert isinstance(result, GeneratedReport)
        assert result.report_type == ReportType.DETAILED_ANALYSIS
        assert any("methodology" in section.section_type for section in result.sections)
        assert any("raw_data" in section.section_type for section in result.sections)

    @pytest.mark.asyncio
    async def test_generate_model_comparison(
        self, generator, mock_repositories, sample_test, sample_analysis_result
    ):
        """Test model comparison report generation."""
        test_id = uuid4()
        focus_models = ["model_0", "model_1"]

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test
        mock_repositories["visualization_service"].create_model_comparison_chart.return_value = (
            "<div>Chart</div>"
        )

        # Execute
        result = await generator.generate_model_comparison(
            test_id, sample_analysis_result, focus_models
        )

        # Verify
        assert isinstance(result, GeneratedReport)
        assert result.report_type == ReportType.MODEL_COMPARISON

    @pytest.mark.asyncio
    async def test_generate_insights_report(
        self, generator, mock_repositories, sample_test, sample_analysis_result
    ):
        """Test insights report generation."""
        test_id = uuid4()

        # Setup mocks
        mock_repositories["test_repository"].find_by_id.return_value = sample_test

        # Execute
        result = await generator.generate_insights_report(test_id, sample_analysis_result)

        # Verify
        assert isinstance(result, GeneratedReport)
        assert result.report_type == ReportType.INSIGHTS_REPORT

    @pytest.mark.asyncio
    async def test_export_to_formats(self, generator, sample_analysis_result):
        """Test exporting report to multiple formats."""
        # Create a sample report
        report = GeneratedReport(
            report_id=uuid4(),
            test_id=uuid4(),
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            title="Test Report",
            generated_at=datetime.utcnow(),
            content="<html><body>Test content</body></html>",
            sections=[],
            metadata={},
        )

        formats = [ReportFormat.JSON, ReportFormat.MARKDOWN, ReportFormat.CSV]

        # Execute
        result = await generator.export_to_formats(report, formats)

        # Verify
        assert isinstance(result, dict)
        assert len(result) <= len(formats)  # Some exports might fail

        for format_type, content in result.items():
            assert format_type in formats
            assert isinstance(content, str)
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_validate_report_inputs_valid(
        self, generator, sample_analysis_result, report_config
    ):
        """Test report input validation with valid inputs."""
        test_id = uuid4()

        # Should not raise exception
        await generator._validate_report_inputs(test_id, sample_analysis_result, report_config)

    @pytest.mark.asyncio
    async def test_validate_report_inputs_invalid_test_id(
        self, generator, sample_analysis_result, report_config
    ):
        """Test report input validation with invalid test ID."""
        with pytest.raises(ValidationError, match="Test ID is required"):
            await generator._validate_report_inputs(None, sample_analysis_result, report_config)

    @pytest.mark.asyncio
    async def test_validate_report_inputs_missing_analysis(self, generator, report_config):
        """Test report input validation with missing analysis result."""
        test_id = uuid4()

        with pytest.raises(ValidationError, match="Analysis result is required"):
            await generator._validate_report_inputs(test_id, None, report_config)

    @pytest.mark.asyncio
    async def test_generate_executive_summary_section(
        self, generator, sample_test, sample_analysis_result
    ):
        """Test executive summary section generation."""
        section = await generator._generate_executive_summary_section(
            sample_test, sample_analysis_result
        )

        assert isinstance(section, ReportSection)
        assert section.title == "Executive Summary"
        assert section.section_type == "executive_summary"
        assert section.order == 1
        assert "Test Overview" in section.content
        assert "Key Findings" in section.content
        assert sample_test.name in section.content

    @pytest.mark.asyncio
    async def test_generate_model_performance_section(
        self, generator, sample_analysis_result, report_config
    ):
        """Test model performance section generation."""
        section = await generator._generate_model_performance_section(
            sample_analysis_result, report_config
        )

        assert isinstance(section, ReportSection)
        assert section.title == "Model Performance Analysis"
        assert section.section_type == "model_performance"
        assert section.order == 2
        assert "performance-table" in section.content
        assert "Model 0" in section.content  # Best performing model

    @pytest.mark.asyncio
    async def test_generate_statistical_analysis_section(self, generator, sample_analysis_result):
        """Test statistical analysis section generation."""
        # Add some statistical tests to the analysis result
        from src.domain.analytics.entities.statistical_test import TestType
        from src.domain.analytics.value_objects.test_result import TestResult

        test_result = Mock(spec=TestResult)
        test_result.test_type = TestType.TTEST_INDEPENDENT
        test_result.p_value = Decimal("0.03")
        test_result.effect_size = Decimal("0.5")
        test_result.is_significant.return_value = True
        test_result.interpretation.practical_interpretation = "Significant difference detected"

        sample_analysis_result.statistical_tests = {"model_0_vs_model_1": test_result}

        section = await generator._generate_statistical_analysis_section(sample_analysis_result)

        assert isinstance(section, ReportSection)
        assert section.title == "Statistical Analysis"
        assert section.section_type == "statistical_analysis"
        assert "test-results-table" in section.content
        assert "model_0_vs_model_1" in section.content

    @pytest.mark.asyncio
    async def test_generate_cost_analysis_section_with_cost_data(
        self, generator, sample_analysis_result
    ):
        """Test cost analysis section with cost data."""
        # Add cost metrics to model performances
        from src.domain.analytics.value_objects.cost_data import CostData
        from src.domain.model_provider.value_objects.money import Money

        for i, (model_id, model) in enumerate(sample_analysis_result.model_performances.items()):
            cost_data = CostData(
                total_cost=Money(amount=Decimal(str(1.0 + i * 0.5)), currency="USD"),
                cost_per_sample=Money(amount=Decimal(str(0.01 + i * 0.005)), currency="USD"),
                total_tokens=1000 + i * 200,
                average_tokens_per_sample=100 + i * 20,
            )
            model.cost_metrics = cost_data

        section = await generator._generate_cost_analysis_section(sample_analysis_result)

        assert isinstance(section, ReportSection)
        assert section.title == "Cost Analysis"
        assert section.section_type == "cost_analysis"
        assert "cost-table" in section.content
        assert "$" in section.content  # Should contain cost information

    @pytest.mark.asyncio
    async def test_generate_cost_analysis_section_no_cost_data(
        self, generator, sample_analysis_result
    ):
        """Test cost analysis section without cost data."""
        section = await generator._generate_cost_analysis_section(sample_analysis_result)

        assert isinstance(section, ReportSection)
        assert section.title == "Cost Analysis"
        assert "No cost data available" in section.content

    def test_has_cost_data_true(self, generator, sample_analysis_result):
        """Test cost data detection when cost data exists."""
        # Add cost metrics to one model
        from src.domain.analytics.value_objects.cost_data import CostData
        from src.domain.model_provider.value_objects.money import Money

        cost_data = CostData(
            total_cost=Money(amount=Decimal("1.0"), currency="USD"),
            cost_per_sample=Money(amount=Decimal("0.01"), currency="USD"),
            total_tokens=1000,
            average_tokens_per_sample=100,
        )
        list(sample_analysis_result.model_performances.values())[0].cost_metrics = cost_data

        result = generator._has_cost_data(sample_analysis_result)
        assert result is True

    def test_has_cost_data_false(self, generator, sample_analysis_result):
        """Test cost data detection when no cost data exists."""
        result = generator._has_cost_data(sample_analysis_result)
        assert result is False

    @pytest.mark.asyncio
    async def test_compile_html_content(self, generator, report_config):
        """Test HTML content compilation."""
        sections = [
            ReportSection(
                title="Test Section",
                content="<h2>Test Section</h2><p>Test content</p>",
                section_type="test",
                metadata={},
                order=1,
            )
        ]

        result = await generator._compile_html_content(sections, report_config)

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert "<html>" in result
        assert "<title>Test Report</title>" in result
        assert "Test Section" in result
        assert "Test content" in result

    @pytest.mark.asyncio
    async def test_compile_markdown_content(self, generator):
        """Test Markdown content compilation."""
        sections = [
            ReportSection(
                title="Test Section",
                content="<h2>Test Section</h2><p>Test content</p>",
                section_type="test",
                metadata={},
                order=1,
            )
        ]

        result = await generator._compile_markdown_content(sections)

        assert isinstance(result, str)
        assert "## Test Section" in result
        assert "Test content" in result

    @pytest.mark.asyncio
    async def test_compile_json_content(self, generator):
        """Test JSON content compilation."""
        sections = [
            ReportSection(
                title="Test Section",
                content="<h2>Test Section</h2>",
                section_type="test",
                metadata={"key": "value"},
                order=1,
            )
        ]

        result = await generator._compile_json_content(sections)

        assert isinstance(result, str)
        import json

        data = json.loads(result)
        assert "sections" in data
        assert len(data["sections"]) == 1
        assert data["sections"][0]["title"] == "Test Section"

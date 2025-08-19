"""Report generation service for comprehensive analytics reporting."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from ....domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from ....domain.analytics.exceptions import ReportGenerationError, ValidationError
from ....domain.analytics.repositories.analytics_repository import AnalyticsRepository
from ....domain.analytics.value_objects.test_result import TestResult
from ....domain.test_management.repositories.test_repository import TestRepository
from ...dto.report_configuration_dto import ReportConfigurationDTO, ReportFormat, ReportType
from .visualization_service import VisualizationService

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Individual section of a report."""

    title: str
    content: str
    section_type: str
    metadata: Dict[str, Any]
    order: int
    include_charts: bool = False
    charts: List[Any] = None

    def __post_init__(self):
        if self.charts is None:
            self.charts = []


@dataclass
class GeneratedReport:
    """Generated report with metadata."""

    report_id: UUID
    test_id: UUID
    report_type: ReportType
    format: ReportFormat
    title: str
    generated_at: datetime
    content: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None


class ReportGenerator:
    """Service for generating comprehensive analytics reports."""

    def __init__(
        self,
        test_repository: TestRepository,
        analytics_repository: AnalyticsRepository,
        visualization_service: VisualizationService,
    ):
        self.test_repository = test_repository
        self.analytics_repository = analytics_repository
        self.visualization_service = visualization_service
        self._logger = logger.getChild(self.__class__.__name__)

    async def generate_report(
        self, test_id: UUID, analysis_result: AnalysisResult, report_config: ReportConfigurationDTO
    ) -> GeneratedReport:
        """
        Generate a comprehensive report for test analysis.

        Args:
            test_id: ID of the test
            analysis_result: Analysis results to include in report
            report_config: Configuration for report generation

        Returns:
            GeneratedReport with content and metadata

        Raises:
            ValidationError: If inputs are invalid
            ReportGenerationError: If report generation fails
        """
        try:
            self._logger.info(
                f"Generating {report_config.report_type.value} report for test {test_id}"
            )

            # Validate inputs
            await self._validate_report_inputs(test_id, analysis_result, report_config)

            # Load test information
            test = await self.test_repository.find_by_id(test_id)

            # Generate report sections
            sections = await self._generate_report_sections(test, analysis_result, report_config)

            # Generate visualizations if requested
            if report_config.include_visualizations:
                await self._add_visualizations_to_sections(sections, analysis_result, report_config)

            # Compile report content
            content = await self._compile_report_content(sections, report_config)

            # Create report metadata
            metadata = {
                "generation_time": datetime.utcnow().isoformat(),
                "total_sections": len(sections),
                "models_analyzed": len(analysis_result.model_performances),
                "statistical_tests": len(analysis_result.statistical_tests),
                "insights_count": len(analysis_result.insights),
                "config": report_config.to_dict(),
            }

            report = GeneratedReport(
                report_id=uuid4(),
                test_id=test_id,
                report_type=report_config.report_type,
                format=report_config.format,
                title=report_config.title
                or f"{report_config.report_type.value.replace('_', ' ').title()} Report",
                generated_at=datetime.utcnow(),
                content=content,
                sections=sections,
                metadata=metadata,
            )

            # Save report if requested
            if report_config.save_to_file:
                await self._save_report_to_file(report, report_config)

            self._logger.info(f"Report generated successfully: {report.report_id}")
            return report

        except Exception as e:
            self._logger.error(f"Report generation failed: {str(e)}")
            raise ReportGenerationError(f"Failed to generate report: {str(e)}")

    async def generate_executive_summary(
        self, test_id: UUID, analysis_result: AnalysisResult
    ) -> GeneratedReport:
        """Generate an executive summary report."""
        config = ReportConfigurationDTO(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            include_statistical_details=False,
            include_visualizations=True,
            include_recommendations=True,
        )

        return await self.generate_report(test_id, analysis_result, config)

    async def generate_detailed_analysis(
        self, test_id: UUID, analysis_result: AnalysisResult, include_raw_data: bool = False
    ) -> GeneratedReport:
        """Generate a detailed analysis report."""
        config = ReportConfigurationDTO(
            report_type=ReportType.DETAILED_ANALYSIS,
            format=ReportFormat.HTML,
            include_statistical_details=True,
            include_visualizations=True,
            include_raw_data=include_raw_data,
            include_recommendations=True,
            include_methodology=True,
        )

        return await self.generate_report(test_id, analysis_result, config)

    async def generate_model_comparison(
        self,
        test_id: UUID,
        analysis_result: AnalysisResult,
        focus_models: Optional[List[str]] = None,
    ) -> GeneratedReport:
        """Generate a model comparison report."""
        config = ReportConfigurationDTO(
            report_type=ReportType.MODEL_COMPARISON,
            format=ReportFormat.HTML,
            include_statistical_details=True,
            include_visualizations=True,
            include_cost_analysis=True,
            focus_models=focus_models,
        )

        return await self.generate_report(test_id, analysis_result, config)

    async def generate_insights_report(
        self, test_id: UUID, analysis_result: AnalysisResult, min_confidence: float = 0.7
    ) -> GeneratedReport:
        """Generate an insights-focused report."""
        config = ReportConfigurationDTO(
            report_type=ReportType.INSIGHTS_REPORT,
            format=ReportFormat.HTML,
            include_visualizations=True,
            include_recommendations=True,
            min_insight_confidence=min_confidence,
        )

        return await self.generate_report(test_id, analysis_result, config)

    async def export_to_formats(
        self, report: GeneratedReport, formats: List[ReportFormat]
    ) -> Dict[ReportFormat, str]:
        """Export a report to multiple formats."""
        exported_content = {}

        for format_type in formats:
            try:
                if format_type == ReportFormat.JSON:
                    content = await self._export_to_json(report)
                elif format_type == ReportFormat.MARKDOWN:
                    content = await self._export_to_markdown(report)
                elif format_type == ReportFormat.CSV:
                    content = await self._export_to_csv(report)
                else:
                    # Use original content for HTML/PDF
                    content = report.content

                exported_content[format_type] = content

            except Exception as e:
                self._logger.warning(f"Export to {format_type.value} failed: {str(e)}")
                continue

        return exported_content

    async def _validate_report_inputs(
        self, test_id: UUID, analysis_result: AnalysisResult, report_config: ReportConfigurationDTO
    ):
        """Validate report generation inputs."""
        if not test_id:
            raise ValidationError("Test ID is required")

        if not analysis_result:
            raise ValidationError("Analysis result is required")

        if not analysis_result.is_completed():
            raise ValidationError("Analysis must be completed before generating report")

        if not report_config:
            raise ValidationError("Report configuration is required")

    async def _generate_report_sections(
        self, test, analysis_result: AnalysisResult, report_config: ReportConfigurationDTO
    ) -> List[ReportSection]:
        """Generate sections based on report type and configuration."""
        sections = []

        # Executive Summary (always included)
        sections.append(await self._generate_executive_summary_section(test, analysis_result))

        # Model Performance Overview
        sections.append(
            await self._generate_model_performance_section(analysis_result, report_config)
        )

        # Statistical Analysis (if requested)
        if report_config.include_statistical_details:
            sections.append(await self._generate_statistical_analysis_section(analysis_result))

        # Cost Analysis (if available and requested)
        if report_config.include_cost_analysis and self._has_cost_data(analysis_result):
            sections.append(await self._generate_cost_analysis_section(analysis_result))

        # Insights and Recommendations (if requested)
        if report_config.include_recommendations:
            sections.append(await self._generate_insights_section(analysis_result, report_config))

        # Methodology (if requested)
        if report_config.include_methodology:
            sections.append(await self._generate_methodology_section(test, analysis_result))

        # Raw Data (if requested)
        if report_config.include_raw_data:
            sections.append(await self._generate_raw_data_section(analysis_result))

        # Sort sections by order
        sections.sort(key=lambda s: s.order)

        return sections

    async def _generate_executive_summary_section(
        self, test, analysis_result: AnalysisResult
    ) -> ReportSection:
        """Generate executive summary section."""

        # Get best performing model
        best_model = analysis_result.get_best_performing_model()
        most_cost_effective = analysis_result.get_most_cost_effective_model()

        # Get key statistics
        summary = analysis_result.get_model_comparison_summary()
        significant_tests = len(analysis_result.get_significant_tests())
        actionable_insights = len(analysis_result.get_actionable_insights())

        content = f"""
        <h2>Executive Summary</h2>
        
        <div class="summary-stats">
            <div class="stat-card">
                <h3>Test Overview</h3>
                <p><strong>Test Name:</strong> {test.name}</p>
                <p><strong>Models Tested:</strong> {summary['total_models']}</p>
                <p><strong>Total Samples:</strong> {analysis_result.get_total_sample_count()}</p>
                <p><strong>Analysis Completed:</strong> {analysis_result.completed_at.strftime('%Y-%m-%d %H:%M UTC') if analysis_result.completed_at else 'N/A'}</p>
            </div>
            
            <div class="stat-card">
                <h3>Key Findings</h3>
                <p><strong>Best Performing Model:</strong> {best_model.model_name if best_model else 'N/A'}</p>
                <p><strong>Performance Score:</strong> {str(best_model.overall_score) if best_model else 'N/A'}</p>
                <p><strong>Significant Differences:</strong> {significant_tests} tests</p>
                <p><strong>Actionable Insights:</strong> {actionable_insights}</p>
            </div>
            
            <div class="stat-card">
                <h3>Cost Efficiency</h3>
                <p><strong>Most Cost-Effective:</strong> {most_cost_effective.model_name if most_cost_effective else 'N/A'}</p>
                <p><strong>Performance Spread:</strong> {summary['performance_spread']}</p>
                <p><strong>Statistical Tests:</strong> {summary['total_statistical_tests']}</p>
                <p><strong>High-Confidence Insights:</strong> {summary['high_confidence_insights']}</p>
            </div>
        </div>
        
        <div class="key-takeaways">
            <h3>Key Takeaways</h3>
            <ul>
        """

        # Add key takeaways based on insights
        high_confidence_insights = analysis_result.get_high_confidence_insights()
        for insight in high_confidence_insights[:3]:  # Top 3 insights
            content += f"<li>{insight.title}: {insight.description}</li>"

        content += """
            </ul>
        </div>
        """

        return ReportSection(
            title="Executive Summary",
            content=content,
            section_type="executive_summary",
            metadata={
                "models_count": summary["total_models"],
                "insights_count": actionable_insights,
            },
            order=1,
            include_charts=True,
        )

    async def _generate_model_performance_section(
        self, analysis_result: AnalysisResult, report_config: ReportConfigurationDTO
    ) -> ReportSection:
        """Generate model performance comparison section."""

        models = list(analysis_result.model_performances.values())
        models.sort(key=lambda m: m.overall_score, reverse=True)

        content = """
        <h2>Model Performance Analysis</h2>
        
        <div class="performance-overview">
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Overall Score</th>
                        <th>Sample Count</th>
                        <th>Confidence</th>
                        <th>Cost per Sample</th>
                        <th>Key Strengths</th>
                    </tr>
                </thead>
                <tbody>
        """

        for rank, model in enumerate(models, 1):
            # Find best dimension for this model
            best_dimension = (
                max(model.dimension_scores.items(), key=lambda x: x[1])
                if model.dimension_scores
                else ("N/A", "N/A")
            )

            cost_per_sample = "N/A"
            if model.cost_metrics:
                cost_per_sample = f"${model.cost_metrics.cost_per_sample.amount:.4f}"

            content += f"""
                <tr>
                    <td>{rank}</td>
                    <td>{model.model_name}</td>
                    <td>{float(model.overall_score):.3f}</td>
                    <td>{model.sample_count}</td>
                    <td>{float(model.confidence_score):.3f}</td>
                    <td>{cost_per_sample}</td>
                    <td>{best_dimension[0]}: {float(best_dimension[1]):.3f}</td>
                </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        
        <div class="dimension-analysis">
            <h3>Dimension Performance Breakdown</h3>
            <p>Analysis of model performance across different evaluation dimensions.</p>
        </div>
        """

        return ReportSection(
            title="Model Performance Analysis",
            content=content,
            section_type="model_performance",
            metadata={"models_analyzed": len(models)},
            order=2,
            include_charts=True,
        )

    async def _generate_statistical_analysis_section(
        self, analysis_result: AnalysisResult
    ) -> ReportSection:
        """Generate statistical analysis section."""

        significant_tests = analysis_result.get_significant_tests()

        content = f"""
        <h2>Statistical Analysis</h2>
        
        <div class="statistical-overview">
            <p><strong>Total Tests Performed:</strong> {len(analysis_result.statistical_tests)}</p>
            <p><strong>Statistically Significant Results:</strong> {len(significant_tests)}</p>
            <p><strong>Significance Level:</strong> α = 0.05</p>
        </div>
        
        <div class="test-results">
            <h3>Test Results Summary</h3>
            <table class="test-results-table">
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Test Type</th>
                        <th>p-value</th>
                        <th>Effect Size</th>
                        <th>Significant</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
        """

        for test_name, result in analysis_result.statistical_tests.items():
            is_significant = "Yes" if result.is_significant() else "No"
            effect_size_str = f"{float(result.effect_size):.3f}" if result.effect_size else "N/A"

            content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{result.test_type.value}</td>
                    <td>{float(result.p_value):.4f}</td>
                    <td>{effect_size_str}</td>
                    <td>{is_significant}</td>
                    <td>{result.interpretation.practical_interpretation}</td>
                </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        """

        return ReportSection(
            title="Statistical Analysis",
            content=content,
            section_type="statistical_analysis",
            metadata={
                "total_tests": len(analysis_result.statistical_tests),
                "significant_tests": len(significant_tests),
            },
            order=3,
            include_charts=True,
        )

    async def _generate_cost_analysis_section(
        self, analysis_result: AnalysisResult
    ) -> ReportSection:
        """Generate cost analysis section."""

        models_with_cost = [
            model
            for model in analysis_result.model_performances.values()
            if model.cost_metrics is not None
        ]

        if not models_with_cost:
            return ReportSection(
                title="Cost Analysis",
                content="<h2>Cost Analysis</h2><p>No cost data available for analysis.</p>",
                section_type="cost_analysis",
                metadata={},
                order=4,
            )

        # Sort by cost efficiency
        models_with_cost.sort(
            key=lambda m: float(m.overall_score) / float(m.cost_metrics.cost_per_sample.amount),
            reverse=True,
        )

        content = """
        <h2>Cost Analysis</h2>
        
        <div class="cost-overview">
            <table class="cost-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Total Cost</th>
                        <th>Cost per Sample</th>
                        <th>Performance Score</th>
                        <th>Cost Efficiency</th>
                        <th>Total Tokens</th>
                    </tr>
                </thead>
                <tbody>
        """

        for model in models_with_cost:
            cost_efficiency = float(model.overall_score) / float(
                model.cost_metrics.cost_per_sample.amount
            )

            content += f"""
                <tr>
                    <td>{model.model_name}</td>
                    <td>${float(model.cost_metrics.total_cost.amount):.4f}</td>
                    <td>${float(model.cost_metrics.cost_per_sample.amount):.4f}</td>
                    <td>{float(model.overall_score):.3f}</td>
                    <td>{cost_efficiency:.2f}</td>
                    <td>{model.cost_metrics.total_tokens:,}</td>
                </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        """

        return ReportSection(
            title="Cost Analysis",
            content=content,
            section_type="cost_analysis",
            metadata={"models_with_cost": len(models_with_cost)},
            order=4,
            include_charts=True,
        )

    async def _generate_insights_section(
        self, analysis_result: AnalysisResult, report_config: ReportConfigurationDTO
    ) -> ReportSection:
        """Generate insights and recommendations section."""

        min_confidence = getattr(report_config, "min_insight_confidence", 0.7)
        high_confidence_insights = analysis_result.get_high_confidence_insights(
            Decimal(str(min_confidence))
        )
        actionable_insights = analysis_result.get_actionable_insights()

        content = f"""
        <h2>Insights and Recommendations</h2>
        
        <div class="insights-overview">
            <p><strong>Total Insights Generated:</strong> {len(analysis_result.insights)}</p>
            <p><strong>High Confidence Insights:</strong> {len(high_confidence_insights)}</p>
            <p><strong>Actionable Insights:</strong> {len(actionable_insights)}</p>
        </div>
        
        <div class="high-priority-insights">
            <h3>High Priority Insights</h3>
        """

        for insight in high_confidence_insights[:5]:  # Top 5 insights
            content += f"""
            <div class="insight-card">
                <h4>{insight.title}</h4>
                <p><strong>Category:</strong> {insight.category}</p>
                <p><strong>Confidence:</strong> {float(insight.confidence_score):.2f}</p>
                <p><strong>Severity:</strong> {insight.severity.value}</p>
                <p>{insight.description}</p>
                {f'<p><strong>Recommendation:</strong> {insight.recommendation}</p>' if insight.recommendation else ''}
            </div>
            """

        content += """
        </div>
        
        <div class="actionable-recommendations">
            <h3>Actionable Recommendations</h3>
            <ul>
        """

        for insight in actionable_insights:
            if insight.recommendation:
                content += f"<li>{insight.recommendation}</li>"

        content += """
            </ul>
        </div>
        """

        return ReportSection(
            title="Insights and Recommendations",
            content=content,
            section_type="insights",
            metadata={
                "high_confidence_count": len(high_confidence_insights),
                "actionable_count": len(actionable_insights),
            },
            order=5,
        )

    async def _generate_methodology_section(
        self, test, analysis_result: AnalysisResult
    ) -> ReportSection:
        """Generate methodology section."""

        content = f"""
        <h2>Methodology</h2>
        
        <div class="test-setup">
            <h3>Test Configuration</h3>
            <p><strong>Test ID:</strong> {test.test_id}</p>
            <p><strong>Test Description:</strong> {test.description}</p>
            <p><strong>Sample Size:</strong> {analysis_result.get_total_sample_count()}</p>
            <p><strong>Analysis Duration:</strong> {analysis_result.metadata.get('processing_time_ms', 'N/A')} ms</p>
        </div>
        
        <div class="statistical-methods">
            <h3>Statistical Methods</h3>
            <ul>
                <li>Significance level (α): 0.05</li>
                <li>Effect size calculation: Cohen's d</li>
                <li>Multiple comparison correction: Bonferroni method</li>
                <li>Confidence intervals: 95%</li>
            </ul>
        </div>
        
        <div class="evaluation-criteria">
            <h3>Evaluation Criteria</h3>
            <p>Models were evaluated across multiple dimensions including:</p>
            <ul>
        """

        # Extract dimensions from model performances
        dimensions = set()
        for model in analysis_result.model_performances.values():
            dimensions.update(model.dimension_scores.keys())

        for dimension in sorted(dimensions):
            content += f"<li>{dimension.replace('_', ' ').title()}</li>"

        content += """
            </ul>
        </div>
        """

        return ReportSection(
            title="Methodology",
            content=content,
            section_type="methodology",
            metadata={"dimensions_count": len(dimensions)},
            order=6,
        )

    async def _generate_raw_data_section(self, analysis_result: AnalysisResult) -> ReportSection:
        """Generate raw data section."""

        content = """
        <h2>Raw Data</h2>
        
        <div class="data-export">
            <p>This section contains the raw statistical test results and model performance data.</p>
            
            <details>
                <summary>Statistical Test Results (JSON)</summary>
                <pre><code>
        """

        # Export statistical tests as JSON
        test_data = {
            name: result.to_dict() for name, result in analysis_result.statistical_tests.items()
        }
        content += json.dumps(test_data, indent=2)

        content += """
                </code></pre>
            </details>
            
            <details>
                <summary>Model Performance Data (JSON)</summary>
                <pre><code>
        """

        # Export model performance as JSON
        model_data = {
            model_id: {
                "model_name": perf.model_name,
                "overall_score": str(perf.overall_score),
                "dimension_scores": {
                    dim: str(score) for dim, score in perf.dimension_scores.items()
                },
                "sample_count": perf.sample_count,
                "confidence_score": str(perf.confidence_score),
            }
            for model_id, perf in analysis_result.model_performances.items()
        }
        content += json.dumps(model_data, indent=2)

        content += """
                </code></pre>
            </details>
        </div>
        """

        return ReportSection(
            title="Raw Data", content=content, section_type="raw_data", metadata={}, order=7
        )

    async def _add_visualizations_to_sections(
        self,
        sections: List[ReportSection],
        analysis_result: AnalysisResult,
        report_config: ReportConfigurationDTO,
    ):
        """Add visualizations to report sections."""
        for section in sections:
            if section.include_charts:
                try:
                    if section.section_type == "executive_summary":
                        chart = await self.visualization_service.create_summary_dashboard(
                            analysis_result
                        )
                        section.charts.append(chart)

                    elif section.section_type == "model_performance":
                        chart = await self.visualization_service.create_model_comparison_chart(
                            analysis_result
                        )
                        section.charts.append(chart)

                    elif section.section_type == "statistical_analysis":
                        chart = await self.visualization_service.create_statistical_results_chart(
                            analysis_result
                        )
                        section.charts.append(chart)

                    elif section.section_type == "cost_analysis":
                        chart = await self.visualization_service.create_cost_analysis_chart(
                            analysis_result
                        )
                        section.charts.append(chart)

                except Exception as e:
                    self._logger.warning(
                        f"Visualization generation failed for {section.section_type}: {str(e)}"
                    )
                    continue

    async def _compile_report_content(
        self, sections: List[ReportSection], report_config: ReportConfigurationDTO
    ) -> str:
        """Compile sections into final report content."""

        if report_config.format == ReportFormat.HTML:
            return await self._compile_html_content(sections, report_config)
        elif report_config.format == ReportFormat.MARKDOWN:
            return await self._compile_markdown_content(sections)
        elif report_config.format == ReportFormat.JSON:
            return await self._compile_json_content(sections)
        else:
            # Default to HTML
            return await self._compile_html_content(sections, report_config)

    async def _compile_html_content(
        self, sections: List[ReportSection], report_config: ReportConfigurationDTO
    ) -> str:
        """Compile sections into HTML content."""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_config.title or 'Analytics Report'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary-stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-card {{ flex: 1; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .insight-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }}
                .test-results-table {{ font-size: 0.9em; }}
                pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }}
            </style>
        </head>
        <body>
        """

        for section in sections:
            html += section.content

            # Add charts if available
            for chart in section.charts:
                html += f'<div class="chart-container">{chart}</div>'

        html += """
        </body>
        </html>
        """

        return html

    async def _compile_markdown_content(self, sections: List[ReportSection]) -> str:
        """Compile sections into Markdown content."""
        content = ""

        for section in sections:
            # Convert HTML to basic Markdown
            markdown_content = section.content
            markdown_content = markdown_content.replace("<h2>", "## ")
            markdown_content = markdown_content.replace("</h2>", "\n\n")
            markdown_content = markdown_content.replace("<h3>", "### ")
            markdown_content = markdown_content.replace("</h3>", "\n\n")
            markdown_content = markdown_content.replace("<p>", "")
            markdown_content = markdown_content.replace("</p>", "\n\n")
            markdown_content = markdown_content.replace("<strong>", "**")
            markdown_content = markdown_content.replace("</strong>", "**")

            content += markdown_content + "\n\n"

        return content

    async def _compile_json_content(self, sections: List[ReportSection]) -> str:
        """Compile sections into JSON content."""
        report_data = {
            "sections": [
                {
                    "title": section.title,
                    "type": section.section_type,
                    "content": section.content,
                    "metadata": section.metadata,
                    "order": section.order,
                }
                for section in sections
            ]
        }

        return json.dumps(report_data, indent=2)

    async def _save_report_to_file(
        self, report: GeneratedReport, report_config: ReportConfigurationDTO
    ):
        """Save report to file."""
        # This would be implemented based on file storage requirements
        pass

    def _has_cost_data(self, analysis_result: AnalysisResult) -> bool:
        """Check if analysis result contains cost data."""
        return any(
            model.cost_metrics is not None for model in analysis_result.model_performances.values()
        )

    async def _export_to_json(self, report: GeneratedReport) -> str:
        """Export report to JSON format."""
        return json.dumps(
            (
                report.to_dict()
                if hasattr(report, "to_dict")
                else {
                    "report_id": str(report.report_id),
                    "test_id": str(report.test_id),
                    "report_type": report.report_type.value,
                    "title": report.title,
                    "generated_at": report.generated_at.isoformat(),
                    "sections": [
                        {
                            "title": section.title,
                            "content": section.content,
                            "type": section.section_type,
                            "metadata": section.metadata,
                        }
                        for section in report.sections
                    ],
                    "metadata": report.metadata,
                }
            ),
            indent=2,
        )

    async def _export_to_markdown(self, report: GeneratedReport) -> str:
        """Export report to Markdown format."""
        return await self._compile_markdown_content(report.sections)

    async def _export_to_csv(self, report: GeneratedReport) -> str:
        """Export report data to CSV format."""
        # Extract tabular data from sections for CSV export
        csv_content = "Section,Title,Type,Content_Length,Metadata\n"

        for i, section in enumerate(report.sections):
            csv_content += f'{i+1},{section.title},{section.section_type},{len(section.content)},"{json.dumps(section.metadata)}"\n'

        return csv_content

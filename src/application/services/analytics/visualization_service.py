"""Visualization service for analytics charts and graphs."""

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from ....domain.analytics.entities.analysis_result import AnalysisResult, ModelPerformanceMetrics
from ....domain.analytics.exceptions import ValidationError, VisualizationError
from ....domain.analytics.value_objects.test_result import TestResult

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    chart_type: str
    title: str
    width: int = 800
    height: int = 600
    color_scheme: str = "viridis"
    show_legend: bool = True
    interactive: bool = True
    export_format: str = "html"


@dataclass
class ChartData:
    """Data structure for chart generation."""

    labels: List[str]
    datasets: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class VisualizationService:
    """Service for generating analytics visualizations."""

    def __init__(self):
        self._logger = logger.getChild(self.__class__.__name__)
        self._default_colors = [
            "#3498db",
            "#e74c3c",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#34495e",
            "#e67e22",
            "#95a5a6",
            "#c0392b",
        ]

    async def create_summary_dashboard(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """
        Create a summary dashboard with key metrics.

        Args:
            analysis_result: Analysis results to visualize
            config: Chart configuration

        Returns:
            HTML string containing the dashboard
        """
        if config is None:
            config = ChartConfig(
                chart_type="dashboard", title="Executive Summary Dashboard", height=400
            )

        try:
            # Prepare data for multiple charts
            model_scores_data = await self._prepare_model_scores_data(analysis_result)
            cost_efficiency_data = await self._prepare_cost_efficiency_data(analysis_result)
            insights_data = await self._prepare_insights_distribution_data(analysis_result)

            # Generate HTML dashboard
            dashboard_html = f"""
            <div class="dashboard-container" style="width: {config.width}px;">
                <h3>{config.title}</h3>
                
                <div class="dashboard-row" style="display: flex; gap: 20px; margin: 20px 0;">
                    <div class="chart-panel" style="flex: 1;">
                        <h4>Model Performance Comparison</h4>
                        {await self._create_bar_chart(model_scores_data, "Model Performance")}
                    </div>
                    
                    <div class="chart-panel" style="flex: 1;">
                        <h4>Cost Efficiency Analysis</h4>
                        {await self._create_scatter_plot(cost_efficiency_data, "Cost vs Performance")}
                    </div>
                </div>
                
                <div class="dashboard-row" style="display: flex; gap: 20px; margin: 20px 0;">
                    <div class="chart-panel" style="flex: 1;">
                        <h4>Insights Distribution</h4>
                        {await self._create_pie_chart(insights_data, "Insight Categories")}
                    </div>
                    
                    <div class="metrics-panel" style="flex: 1;">
                        {await self._create_key_metrics_panel(analysis_result)}
                    </div>
                </div>
            </div>
            """

            return dashboard_html

        except Exception as e:
            self._logger.error(f"Dashboard creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create dashboard: {str(e)}")

    async def create_model_comparison_chart(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """Create a model comparison chart."""
        if config is None:
            config = ChartConfig(chart_type="bar", title="Model Performance Comparison")

        try:
            data = await self._prepare_model_comparison_data(analysis_result)
            return await self._create_grouped_bar_chart(data, config.title)

        except Exception as e:
            self._logger.error(f"Model comparison chart creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create model comparison chart: {str(e)}")

    async def create_statistical_results_chart(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """Create a statistical results visualization."""
        if config is None:
            config = ChartConfig(chart_type="forest_plot", title="Statistical Test Results")

        try:
            data = await self._prepare_statistical_results_data(analysis_result)
            return await self._create_forest_plot(data, config.title)

        except Exception as e:
            self._logger.error(f"Statistical results chart creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create statistical results chart: {str(e)}")

    async def create_cost_analysis_chart(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """Create a cost analysis chart."""
        if config is None:
            config = ChartConfig(chart_type="bubble", title="Cost-Performance Analysis")

        try:
            data = await self._prepare_cost_analysis_data(analysis_result)
            return await self._create_bubble_chart(data, config.title)

        except Exception as e:
            self._logger.error(f"Cost analysis chart creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create cost analysis chart: {str(e)}")

    async def create_dimension_heatmap(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """Create a heatmap of dimension performance across models."""
        if config is None:
            config = ChartConfig(chart_type="heatmap", title="Model Performance by Dimension")

        try:
            data = await self._prepare_dimension_heatmap_data(analysis_result)
            return await self._create_heatmap(data, config.title)

        except Exception as e:
            self._logger.error(f"Dimension heatmap creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create dimension heatmap: {str(e)}")

    async def create_confidence_interval_plot(
        self, analysis_result: AnalysisResult, config: Optional[ChartConfig] = None
    ) -> str:
        """Create confidence interval plots for statistical tests."""
        if config is None:
            config = ChartConfig(chart_type="error_bars", title="Confidence Intervals")

        try:
            data = await self._prepare_confidence_interval_data(analysis_result)
            return await self._create_confidence_interval_chart(data, config.title)

        except Exception as e:
            self._logger.error(f"Confidence interval plot creation failed: {str(e)}")
            raise VisualizationError(f"Failed to create confidence interval plot: {str(e)}")

    async def _prepare_model_scores_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for model scores chart."""
        models = list(analysis_result.model_performances.values())
        models.sort(key=lambda m: m.overall_score, reverse=True)

        labels = [model.model_name for model in models]
        scores = [float(model.overall_score) for model in models]

        return ChartData(
            labels=labels,
            datasets=[
                {
                    "label": "Overall Score",
                    "data": scores,
                    "backgroundColor": self._default_colors[: len(models)],
                }
            ],
            metadata={"chart_type": "bar"},
        )

    async def _prepare_cost_efficiency_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for cost efficiency scatter plot."""
        models_with_cost = [
            model
            for model in analysis_result.model_performances.values()
            if model.cost_metrics is not None
        ]

        if not models_with_cost:
            return ChartData(labels=[], datasets=[], metadata={})

        performance_scores = [float(model.overall_score) for model in models_with_cost]
        costs = [float(model.cost_metrics.cost_per_sample.amount) for model in models_with_cost]
        labels = [model.model_name for model in models_with_cost]

        return ChartData(
            labels=labels,
            datasets=[
                {
                    "label": "Models",
                    "data": [
                        {"x": cost, "y": performance, "label": label}
                        for cost, performance, label in zip(costs, performance_scores, labels)
                    ],
                    "backgroundColor": self._default_colors[: len(models_with_cost)],
                }
            ],
            metadata={
                "chart_type": "scatter",
                "x_axis": "Cost per Sample",
                "y_axis": "Performance Score",
            },
        )

    async def _prepare_insights_distribution_data(
        self, analysis_result: AnalysisResult
    ) -> ChartData:
        """Prepare data for insights distribution pie chart."""
        insight_categories = {}
        for insight in analysis_result.insights:
            category = insight.category
            insight_categories[category] = insight_categories.get(category, 0) + 1

        if not insight_categories:
            return ChartData(labels=[], datasets=[], metadata={})

        labels = list(insight_categories.keys())
        counts = list(insight_categories.values())

        return ChartData(
            labels=labels,
            datasets=[
                {
                    "label": "Insight Categories",
                    "data": counts,
                    "backgroundColor": self._default_colors[: len(labels)],
                }
            ],
            metadata={"chart_type": "pie"},
        )

    async def _prepare_model_comparison_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for grouped bar chart comparing models across dimensions."""
        models = list(analysis_result.model_performances.values())

        # Get all unique dimensions
        all_dimensions = set()
        for model in models:
            all_dimensions.update(model.dimension_scores.keys())

        all_dimensions = sorted(list(all_dimensions))
        model_names = [model.model_name for model in models]

        # Create dataset for each dimension
        datasets = []
        for i, dimension in enumerate(all_dimensions):
            dimension_scores = []
            for model in models:
                score = float(model.dimension_scores.get(dimension, 0))
                dimension_scores.append(score)

            datasets.append(
                {
                    "label": dimension.replace("_", " ").title(),
                    "data": dimension_scores,
                    "backgroundColor": self._default_colors[i % len(self._default_colors)],
                }
            )

        return ChartData(
            labels=model_names, datasets=datasets, metadata={"chart_type": "grouped_bar"}
        )

    async def _prepare_statistical_results_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for forest plot of statistical results."""
        test_names = []
        effect_sizes = []
        confidence_intervals = []
        p_values = []

        for test_name, result in analysis_result.statistical_tests.items():
            test_names.append(test_name)
            effect_sizes.append(float(result.effect_size) if result.effect_size else 0)
            p_values.append(float(result.p_value) if result.p_value else 1)

            if result.confidence_interval:
                ci = (
                    float(result.confidence_interval.lower_bound),
                    float(result.confidence_interval.upper_bound),
                )
                confidence_intervals.append(ci)
            else:
                confidence_intervals.append((0, 0))

        return ChartData(
            labels=test_names,
            datasets=[
                {
                    "label": "Effect Sizes",
                    "data": effect_sizes,
                    "confidence_intervals": confidence_intervals,
                    "p_values": p_values,
                    "backgroundColor": ["#e74c3c" if p < 0.05 else "#95a5a6" for p in p_values],
                }
            ],
            metadata={"chart_type": "forest_plot"},
        )

    async def _prepare_cost_analysis_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for bubble chart of cost analysis."""
        models_with_cost = [
            model
            for model in analysis_result.model_performances.values()
            if model.cost_metrics is not None
        ]

        if not models_with_cost:
            return ChartData(labels=[], datasets=[], metadata={})

        bubble_data = []
        for model in models_with_cost:
            bubble_data.append(
                {
                    "x": float(model.cost_metrics.cost_per_sample.amount),
                    "y": float(model.overall_score),
                    "r": model.sample_count / 10,  # Bubble size based on sample count
                    "label": model.model_name,
                }
            )

        return ChartData(
            labels=[model.model_name for model in models_with_cost],
            datasets=[
                {
                    "label": "Models",
                    "data": bubble_data,
                    "backgroundColor": self._default_colors[: len(models_with_cost)],
                }
            ],
            metadata={
                "chart_type": "bubble",
                "x_axis": "Cost per Sample ($)",
                "y_axis": "Performance Score",
                "size_axis": "Sample Count",
            },
        )

    async def _prepare_dimension_heatmap_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for dimension performance heatmap."""
        models = list(analysis_result.model_performances.values())

        # Get all dimensions
        all_dimensions = set()
        for model in models:
            all_dimensions.update(model.dimension_scores.keys())

        dimensions = sorted(list(all_dimensions))
        model_names = [model.model_name for model in models]

        # Create matrix of scores
        heatmap_data = []
        for model in models:
            model_scores = []
            for dimension in dimensions:
                score = float(model.dimension_scores.get(dimension, 0))
                model_scores.append(score)
            heatmap_data.append(model_scores)

        return ChartData(
            labels=model_names,
            datasets=[
                {"label": "Performance Scores", "data": heatmap_data, "dimensions": dimensions}
            ],
            metadata={"chart_type": "heatmap", "dimensions": dimensions},
        )

    async def _prepare_confidence_interval_data(self, analysis_result: AnalysisResult) -> ChartData:
        """Prepare data for confidence interval chart."""
        test_names = []
        means = []
        lower_bounds = []
        upper_bounds = []

        for test_name, result in analysis_result.statistical_tests.items():
            if result.confidence_interval:
                test_names.append(test_name)

                # Use effect size as the mean
                mean = float(result.effect_size) if result.effect_size else 0
                means.append(mean)

                lower_bounds.append(float(result.confidence_interval.lower_bound))
                upper_bounds.append(float(result.confidence_interval.upper_bound))

        return ChartData(
            labels=test_names,
            datasets=[
                {
                    "label": "Effect Sizes with 95% CI",
                    "data": means,
                    "lower_bounds": lower_bounds,
                    "upper_bounds": upper_bounds,
                    "backgroundColor": self._default_colors[: len(test_names)],
                }
            ],
            metadata={"chart_type": "error_bars"},
        )

    async def _create_bar_chart(self, data: ChartData, title: str) -> str:
        """Create a bar chart using Chart.js."""
        chart_id = f"chart_{hash(title) % 10000}"

        chart_data = {
            "type": "bar",
            "data": {"labels": data.labels, "datasets": data.datasets},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": title}},
                "scales": {"y": {"beginAtZero": True}},
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="400" height="200"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_scatter_plot(self, data: ChartData, title: str) -> str:
        """Create a scatter plot using Chart.js."""
        chart_id = f"scatter_{hash(title) % 10000}"

        chart_data = {
            "type": "scatter",
            "data": {"datasets": data.datasets},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": title}},
                "scales": {
                    "x": {
                        "type": "linear",
                        "position": "bottom",
                        "title": {"display": True, "text": data.metadata.get("x_axis", "X Axis")},
                    },
                    "y": {
                        "title": {"display": True, "text": data.metadata.get("y_axis", "Y Axis")}
                    },
                },
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="400" height="200"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_pie_chart(self, data: ChartData, title: str) -> str:
        """Create a pie chart using Chart.js."""
        chart_id = f"pie_{hash(title) % 10000}"

        chart_data = {
            "type": "pie",
            "data": {"labels": data.labels, "datasets": data.datasets},
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": title},
                    "legend": {"position": "bottom"},
                },
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="300" height="200"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_grouped_bar_chart(self, data: ChartData, title: str) -> str:
        """Create a grouped bar chart."""
        chart_id = f"grouped_bar_{hash(title) % 10000}"

        chart_data = {
            "type": "bar",
            "data": {"labels": data.labels, "datasets": data.datasets},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": title}},
                "scales": {"y": {"beginAtZero": True}},
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="600" height="300"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_forest_plot(self, data: ChartData, title: str) -> str:
        """Create a forest plot for statistical results."""
        chart_id = f"forest_{hash(title) % 10000}"

        # Forest plots are more complex, so we'll create a custom visualization
        html = f"""
        <div class="forest-plot-container">
            <h4>{title}</h4>
            <div class="forest-plot" id="{chart_id}">
        """

        dataset = data.datasets[0]
        for i, (label, effect_size, ci, p_value) in enumerate(
            zip(data.labels, dataset["data"], dataset["confidence_intervals"], dataset["p_values"])
        ):
            significance = "significant" if p_value < 0.05 else "not-significant"

            html += f"""
            <div class="forest-row {significance}">
                <span class="test-name">{label}</span>
                <span class="effect-size">{effect_size:.3f}</span>
                <span class="ci">[{ci[0]:.3f}, {ci[1]:.3f}]</span>
                <span class="p-value">p={p_value:.4f}</span>
            </div>
            """

        html += """
            </div>
            <style>
                .forest-plot-container { margin: 20px 0; }
                .forest-row { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #eee; }
                .forest-row.significant { background-color: #ffe6e6; }
                .forest-row.not-significant { background-color: #f0f0f0; }
                .test-name { flex: 2; font-weight: bold; }
                .effect-size, .ci, .p-value { flex: 1; text-align: center; }
            </style>
        </div>
        """

        return html

    async def _create_bubble_chart(self, data: ChartData, title: str) -> str:
        """Create a bubble chart."""
        chart_id = f"bubble_{hash(title) % 10000}"

        chart_data = {
            "type": "bubble",
            "data": {"datasets": data.datasets},
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": title}},
                "scales": {
                    "x": {
                        "title": {"display": True, "text": data.metadata.get("x_axis", "X Axis")}
                    },
                    "y": {
                        "title": {"display": True, "text": data.metadata.get("y_axis", "Y Axis")}
                    },
                },
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="500" height="300"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_heatmap(self, data: ChartData, title: str) -> str:
        """Create a heatmap."""
        chart_id = f"heatmap_{hash(title) % 10000}"

        # Create a simple HTML heatmap
        html = f"""
        <div class="heatmap-container">
            <h4>{title}</h4>
            <table class="heatmap" id="{chart_id}">
                <thead>
                    <tr>
                        <th>Model</th>
        """

        dimensions = data.metadata.get("dimensions", [])
        for dimension in dimensions:
            html += f"<th>{dimension.replace('_', ' ').title()}</th>"

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        dataset = data.datasets[0]
        for i, (model_name, scores) in enumerate(zip(data.labels, dataset["data"])):
            html += f"<tr><td>{model_name}</td>"

            for score in scores:
                # Color intensity based on score (0-1 range)
                intensity = min(255, int(score * 255))
                color = f"rgba(52, 152, 219, {score})"
                html += f'<td style="background-color: {color};">{score:.3f}</td>'

            html += "</tr>"

        html += """
                </tbody>
            </table>
            <style>
                .heatmap { border-collapse: collapse; width: 100%; }
                .heatmap th, .heatmap td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                .heatmap th { background-color: #f5f5f5; }
            </style>
        </div>
        """

        return html

    async def _create_confidence_interval_chart(self, data: ChartData, title: str) -> str:
        """Create confidence interval error bars chart."""
        chart_id = f"ci_{hash(title) % 10000}"

        dataset = data.datasets[0]

        chart_data = {
            "type": "bar",
            "data": {
                "labels": data.labels,
                "datasets": [
                    {
                        "label": dataset["label"],
                        "data": dataset["data"],
                        "backgroundColor": dataset["backgroundColor"],
                        "errorBars": {
                            "plus": [
                                upper - mean
                                for upper, mean in zip(dataset["upper_bounds"], dataset["data"])
                            ],
                            "minus": [
                                mean - lower
                                for mean, lower in zip(dataset["data"], dataset["lower_bounds"])
                            ],
                        },
                    }
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {"title": {"display": True, "text": title}},
                "scales": {"y": {"beginAtZero": False}},
            },
        }

        return f"""
        <div class="chart-container">
            <canvas id="{chart_id}" width="500" height="300"></canvas>
            <script>
                var ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
                var chart_{chart_id} = new Chart(ctx_{chart_id}, {json.dumps(chart_data)});
            </script>
        </div>
        """

    async def _create_key_metrics_panel(self, analysis_result: AnalysisResult) -> str:
        """Create a key metrics panel."""
        summary = analysis_result.get_model_comparison_summary()
        best_model = analysis_result.get_best_performing_model()

        html = """
        <div class="key-metrics-panel">
            <h4>Key Metrics</h4>
            <div class="metrics-grid">
        """

        metrics = [
            ("Models Tested", summary["total_models"]),
            ("Best Performance", f"{float(best_model.overall_score):.3f}" if best_model else "N/A"),
            ("Significant Tests", summary["significant_differences"]),
            ("Total Samples", analysis_result.get_total_sample_count()),
            ("High-Confidence Insights", summary["high_confidence_insights"]),
            ("Performance Spread", summary["performance_spread"]),
        ]

        for metric_name, metric_value in metrics:
            html += f"""
            <div class="metric-item">
                <div class="metric-label">{metric_name}</div>
                <div class="metric-value">{metric_value}</div>
            </div>
            """

        html += """
            </div>
            <style>
                .key-metrics-panel { padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
                .metric-item { text-align: center; }
                .metric-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
                .metric-value { font-size: 1.2em; font-weight: bold; color: #333; }
            </style>
        </div>
        """

        return html

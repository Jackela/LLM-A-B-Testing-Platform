"""Interactive charts for the dashboard."""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_performance_comparison_chart(
    model_a_data: Dict, model_b_data: Dict, dimensions: List[str]
):
    """Render radar chart comparing model performance across dimensions."""

    # Create radar chart
    fig = go.Figure()

    # Model A
    fig.add_trace(
        go.Scatterpolar(
            r=[model_a_data.get(dim, 0) for dim in dimensions],
            theta=dimensions,
            fill="toself",
            name=model_a_data.get("name", "Model A"),
            line_color="rgb(31, 119, 180)",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    # Model B
    fig.add_trace(
        go.Scatterpolar(
            r=[model_b_data.get(dim, 0) for dim in dimensions],
            theta=dimensions,
            fill="toself",
            name=model_b_data.get("name", "Model B"),
            line_color="rgb(255, 127, 14)",
            fillcolor="rgba(255, 127, 14, 0.2)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Model Performance Comparison",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_test_progress_chart(progress_data: Dict):
    """Render test execution progress chart."""

    # Create progress data
    total = progress_data.get("total_samples", 100)
    completed = progress_data.get("completed_samples", 60)
    failed = progress_data.get("failed_samples", 5)
    remaining = total - completed - failed

    # Donut chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Completed", "Failed", "Remaining"],
                values=[completed, failed, remaining],
                hole=0.3,
                marker_colors=["#28a745", "#dc3545", "#f8f9fa"],
            )
        ]
    )

    fig.update_layout(
        title="Test Execution Progress",
        annotations=[
            dict(text=f"{completed}/{total}", x=0.5, y=0.5, font_size=20, showarrow=False)
        ],
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cost_trend_chart(cost_data: List[Dict]):
    """Render cost trend over time."""

    df = pd.DataFrame(cost_data)

    fig = px.line(
        df,
        x="date",
        y="cost",
        title="Daily Testing Costs",
        labels={"cost": "Cost ($)", "date": "Date"},
    )

    fig.update_layout(height=300, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def render_model_win_rate_chart(model_stats: List[Dict]):
    """Render model win rates comparison."""

    df = pd.DataFrame(model_stats)

    fig = px.bar(
        df,
        x="model",
        y="win_rate",
        title="Model Win Rates",
        labels={"win_rate": "Win Rate", "model": "Model"},
        color="win_rate",
        color_continuous_scale="viridis",
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution_chart(
    model_a_scores: List[float],
    model_b_scores: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
):
    """Render score distribution histogram."""

    fig = go.Figure()

    # Model A histogram
    fig.add_trace(
        go.Histogram(
            x=model_a_scores,
            name=model_a_name,
            opacity=0.7,
            nbinsx=20,
            marker_color="rgb(31, 119, 180)",
        )
    )

    # Model B histogram
    fig.add_trace(
        go.Histogram(
            x=model_b_scores,
            name=model_b_name,
            opacity=0.7,
            nbinsx=20,
            marker_color="rgb(255, 127, 14)",
        )
    )

    fig.update_layout(
        title="Score Distribution Comparison",
        xaxis_title="Score",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_response_time_chart(response_time_data: Dict):
    """Render response time comparison."""

    models = list(response_time_data.keys())
    times = list(response_time_data.values())

    fig = go.Figure(
        data=[go.Bar(x=models, y=times, marker_color=["rgb(31, 119, 180)", "rgb(255, 127, 14)"])]
    )

    fig.update_layout(
        title="Average Response Time Comparison",
        xaxis_title="Model",
        yaxis_title="Response Time (ms)",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_statistical_significance_chart(p_values: Dict[str, float], alpha: float = 0.05):
    """Render statistical significance chart."""

    dimensions = list(p_values.keys())
    p_vals = list(p_values.values())
    colors = ["green" if p < alpha else "red" for p in p_vals]

    fig = go.Figure(
        data=[
            go.Bar(
                x=dimensions,
                y=p_vals,
                marker_color=colors,
                text=[f"p={p:.3f}" for p in p_vals],
                textposition="auto",
            )
        ]
    )

    # Add significance threshold line
    fig.add_hline(y=alpha, line_dash="dash", line_color="black", annotation_text=f"α = {alpha}")

    fig.update_layout(
        title="Statistical Significance by Dimension",
        xaxis_title="Dimension",
        yaxis_title="p-value",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sample_size_power_chart(sample_sizes: List[int], power_values: List[float]):
    """Render sample size vs statistical power chart."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=power_values,
            mode="lines+markers",
            name="Statistical Power",
            line=dict(color="rgb(31, 119, 180)", width=3),
            marker=dict(size=8),
        )
    )

    # Add power threshold line
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Power = 0.8")

    fig.update_layout(
        title="Statistical Power Analysis",
        xaxis_title="Sample Size",
        yaxis_title="Statistical Power",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown_chart(cost_breakdown: Dict[str, float]):
    """Render cost breakdown pie chart."""

    labels = list(cost_breakdown.keys())
    values = list(cost_breakdown.values())

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, textinfo="label+percent", textposition="auto")]
    )

    fig.update_layout(title="Cost Breakdown by Provider", height=400)

    st.plotly_chart(fig, use_container_width=True)

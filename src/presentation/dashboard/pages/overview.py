"""Overview dashboard page."""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ..components.charts import (
    render_cost_breakdown_chart,
    render_cost_trend_chart,
    render_model_win_rate_chart,
    render_test_progress_chart,
)


def render_overview_page():
    """Render the overview dashboard page."""

    st.title("📊 Dashboard Overview")

    # Date range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        date_range = st.selectbox(
            "Time Period", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"], index=1
        )

    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    # Key metrics row
    render_key_metrics()

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        render_recent_tests_summary()
        render_active_tests_monitoring()

    with col2:
        render_model_performance_overview()
        render_cost_analysis_overview()

    # Recent activity
    render_recent_activity()


def render_key_metrics():
    """Render key performance metrics."""

    # Mock data - in production, fetch from API
    metrics = {
        "total_tests": 45,
        "completed_tests": 38,
        "running_tests": 3,
        "failed_tests": 4,
        "total_samples": 15420,
        "total_cost": 1847.32,
        "avg_test_duration": 4.2,
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Tests", value=metrics["total_tests"], delta=f"+{5} this week")

        st.metric(label="Running Tests", value=metrics["running_tests"], delta=None)

    with col2:
        completion_rate = (metrics["completed_tests"] / metrics["total_tests"]) * 100
        st.metric(label="Completion Rate", value=f"{completion_rate:.1f}%", delta=f"+{2.3:.1f}%")

        st.metric(
            label="Failed Tests",
            value=metrics["failed_tests"],
            delta=f"-{1} vs last week",
            delta_color="inverse",
        )

    with col3:
        st.metric(
            label="Total Samples", value=f"{metrics['total_samples']:,}", delta=f"+{1250} this week"
        )

        st.metric(
            label="Avg Duration", value=f"{metrics['avg_test_duration']:.1f}h", delta=f"-{0.5:.1f}h"
        )

    with col4:
        st.metric(
            label="Total Cost",
            value=f"${metrics['total_cost']:,.2f}",
            delta=f"+${234.56:.2f} this week",
        )

        avg_cost = metrics["total_cost"] / metrics["total_samples"]
        st.metric(label="Cost per Sample", value=f"${avg_cost:.3f}", delta=f"-${0.005:.3f}")


def render_recent_tests_summary():
    """Render summary of recent tests."""

    st.subheader("📈 Recent Tests Performance")

    # Mock recent tests data
    recent_tests = [
        {
            "name": "GPT-4 vs Claude Creative",
            "status": "Completed",
            "winner": "Claude-3-Opus",
            "confidence": 0.95,
            "samples": 100,
            "cost": 45.67,
        },
        {
            "name": "Code Generation Comparison",
            "status": "Running",
            "progress": 0.65,
            "samples": 150,
            "cost": 23.45,
        },
        {
            "name": "Factual QA Evaluation",
            "status": "Completed",
            "winner": "GPT-4",
            "confidence": 0.88,
            "samples": 200,
            "cost": 78.23,
        },
    ]

    for test in recent_tests:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**{test['name']}**")
                if test["status"] == "Completed":
                    if test.get("winner"):
                        st.markdown(
                            f"🏆 Winner: {test['winner']} ({test['confidence']:.0%} confidence)"
                        )
                elif test["status"] == "Running":
                    progress = test.get("progress", 0)
                    st.progress(progress)
                    st.caption(f"Progress: {progress:.0%}")

            with col2:
                status_color = {"Completed": "🟢", "Running": "🔄", "Failed": "🔴", "Paused": "⏸️"}
                st.markdown(f"{status_color.get(test['status'], '⚪')} {test['status']}")
                st.caption(f"{test['samples']} samples")

            with col3:
                st.markdown(f"**${test['cost']:.2f}**")
                st.caption(f"${test['cost']/test['samples']:.3f}/sample")

            st.markdown("---")


def render_active_tests_monitoring():
    """Render active tests monitoring."""

    st.subheader("🔄 Active Tests Monitor")

    # Mock active tests data
    active_tests = [
        {
            "id": "test-124",
            "name": "Creative Writing Eval",
            "progress": 0.75,
            "eta": "15 min",
            "samples_done": 75,
            "samples_total": 100,
        },
        {
            "id": "test-125",
            "name": "Translation Quality",
            "progress": 0.30,
            "eta": "2.5 hours",
            "samples_done": 45,
            "samples_total": 150,
        },
    ]

    if not active_tests:
        st.info("No tests currently running")
    else:
        for test in active_tests:
            with st.expander(f"🔄 {test['name']} ({test['progress']:.0%})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.progress(test["progress"])
                    st.caption(f"Progress: {test['samples_done']}/{test['samples_total']} samples")

                with col2:
                    st.metric("ETA", test["eta"])
                    if st.button(f"⏹️ Stop", key=f"stop_{test['id']}"):
                        st.warning(f"Stop requested for {test['name']}")


def render_model_performance_overview():
    """Render model performance overview."""

    st.subheader("🏆 Model Performance")

    # Mock model performance data
    model_stats = [
        {"model": "Claude-3-Opus", "win_rate": 0.78, "avg_score": 8.4, "tests": 12},
        {"model": "GPT-4", "win_rate": 0.72, "avg_score": 8.1, "tests": 15},
        {"model": "Gemini-Pro", "win_rate": 0.65, "avg_score": 7.8, "tests": 8},
        {"model": "GPT-3.5-Turbo", "win_rate": 0.42, "avg_score": 7.2, "tests": 10},
    ]

    # Top performer highlight
    top_performer = max(model_stats, key=lambda x: x["win_rate"])

    st.markdown(f"🥇 **Top Performer:** {top_performer['model']}")
    st.markdown(
        f"Win Rate: {top_performer['win_rate']:.1%} | Avg Score: {top_performer['avg_score']:.1f}"
    )

    # Performance table
    df = pd.DataFrame(model_stats)
    df["Win Rate"] = df["win_rate"].apply(lambda x: f"{x:.1%}")
    df["Avg Score"] = df["avg_score"].apply(lambda x: f"{x:.1f}")
    df["Tests"] = df["tests"]

    st.dataframe(
        df[["model", "Win Rate", "Avg Score", "Tests"]].rename(columns={"model": "Model"}),
        use_container_width=True,
        hide_index=True,
    )


def render_cost_analysis_overview():
    """Render cost analysis overview."""

    st.subheader("💰 Cost Analysis")

    # Mock cost data
    cost_data = {
        "daily_costs": [
            {"date": "2024-01-10", "cost": 42.34},
            {"date": "2024-01-11", "cost": 38.67},
            {"date": "2024-01-12", "cost": 55.23},
            {"date": "2024-01-13", "cost": 47.89},
            {"date": "2024-01-14", "cost": 45.67},
            {"date": "2024-01-15", "cost": 52.34},
        ],
        "cost_by_provider": {"OpenAI": 654.23, "Anthropic": 789.45, "Google": 403.64},
    }

    # Cost trend
    render_cost_trend_chart(cost_data["daily_costs"])

    # Provider breakdown
    total_cost = sum(cost_data["cost_by_provider"].values())

    st.markdown("**Cost by Provider:**")
    for provider, cost in cost_data["cost_by_provider"].items():
        percentage = (cost / total_cost) * 100
        st.markdown(f"- {provider}: ${cost:.2f} ({percentage:.1f}%)")


def render_recent_activity():
    """Render recent activity feed."""

    st.subheader("📝 Recent Activity")

    # Mock activity data
    activities = [
        {
            "type": "test_completed",
            "message": "Test 'GPT-4 vs Claude Creative' completed successfully",
            "timestamp": datetime.now() - timedelta(minutes=30),
            "user": "admin",
        },
        {
            "type": "test_started",
            "message": "Started test 'Translation Quality Comparison'",
            "timestamp": datetime.now() - timedelta(hours=1),
            "user": "user1",
        },
        {
            "type": "test_failed",
            "message": "Test 'Code Generation Eval' failed due to API timeout",
            "timestamp": datetime.now() - timedelta(hours=2),
            "user": "user2",
        },
        {
            "type": "user_login",
            "message": "User 'user3' logged in",
            "timestamp": datetime.now() - timedelta(hours=3),
            "user": "user3",
        },
    ]

    activity_icons = {
        "test_completed": "✅",
        "test_started": "🚀",
        "test_failed": "❌",
        "user_login": "👤",
    }

    for activity in activities:
        col1, col2 = st.columns([3, 1])

        with col1:
            icon = activity_icons.get(activity["type"], "📝")
            st.markdown(f"{icon} {activity['message']}")
            st.caption(f"by {activity['user']}")

        with col2:
            time_ago = datetime.now() - activity["timestamp"]
            if time_ago.seconds < 3600:
                time_str = f"{time_ago.seconds // 60}m ago"
            else:
                time_str = f"{time_ago.seconds // 3600}h ago"
            st.caption(time_str)

        st.markdown("---")

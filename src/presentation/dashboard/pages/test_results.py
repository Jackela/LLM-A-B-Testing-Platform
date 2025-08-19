"""Test results page."""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ..components.charts import (
    render_performance_comparison_chart,
    render_response_time_chart,
    render_score_distribution_chart,
    render_statistical_significance_chart,
)
from ..components.forms import render_test_filters_form


def render_test_results_page():
    """Render test results page."""

    st.title("📊 Test Results")

    # Filters sidebar
    filters = render_test_filters_form()

    # Test selection
    selected_test = render_test_selector()

    if selected_test:
        render_test_details(selected_test)
    else:
        render_test_list()


def render_test_selector():
    """Render test selection dropdown."""

    # Mock test data
    tests = [
        {"id": "test-123", "name": "GPT-4 vs Claude Creative Writing", "status": "completed"},
        {"id": "test-124", "name": "Code Generation Comparison", "status": "running"},
        {"id": "test-125", "name": "Factual QA Evaluation", "status": "completed"},
    ]

    if st.session_state.get("created_tests"):
        tests.extend(st.session_state.created_tests)

    test_options = {test["name"]: test for test in tests}

    selected_name = st.selectbox(
        "Select Test to View", ["None"] + list(test_options.keys()), index=0
    )

    return test_options.get(selected_name) if selected_name != "None" else None


def render_test_details(test: Dict[str, Any]):
    """Render detailed test results."""

    # Test header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(test["name"])
        st.caption(f"Test ID: {test['id']}")

    with col2:
        status_colors = {"completed": "🟢", "running": "🔄", "failed": "🔴", "paused": "⏸️"}
        status = test.get("status", "unknown")
        st.markdown(f"**Status:** {status_colors.get(status, '⚪')} {status.title()}")

    with col3:
        if st.button("🔄 Refresh Results"):
            st.rerun()

    # Show different content based on status
    if status == "running":
        render_running_test_details(test)
    elif status == "completed":
        render_completed_test_details(test)
    elif status == "failed":
        render_failed_test_details(test)
    else:
        st.info(f"Test status: {status}")


def render_running_test_details(test: Dict[str, Any]):
    """Render details for running test."""

    st.markdown("### 🔄 Test in Progress")

    # Progress metrics
    progress = test.get("progress", {"completed": 0, "total": 100})
    completion_rate = progress["completed"] / progress["total"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Progress", f"{completion_rate:.1%}")
        st.progress(completion_rate)

    with col2:
        st.metric("Completed", f"{progress['completed']}/{progress['total']}")

    with col3:
        eta_minutes = (progress["total"] - progress["completed"]) * 2  # 2 min per sample
        st.metric("ETA", f"{eta_minutes} min")

    with col4:
        if st.button("⏹️ Stop Test", type="secondary"):
            st.warning("Stop requested (demo)")

    # Live metrics (mock)
    st.markdown("### 📈 Live Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Success Rate", "94.5%", "+1.2%")
        st.metric("Avg Response Time", "1.8s", "-0.2s")

    with col2:
        st.metric("Current Cost", "$23.45", "+$2.10")
        st.metric("Cost per Sample", "$0.47", "-$0.03")


def render_completed_test_details(test: Dict[str, Any]):
    """Render details for completed test."""

    # Mock results data
    results = {
        "summary": {
            "winner": "Model B",
            "confidence": 0.95,
            "total_samples": 100,
            "model_a_score": 7.8,
            "model_b_score": 8.2,
            "statistical_significance": True,
            "p_value": 0.023,
        },
        "dimensions": {
            "accuracy": {"model_a": 7.5, "model_b": 8.1, "p_value": 0.023},
            "helpfulness": {"model_a": 8.0, "model_b": 8.3, "p_value": 0.045},
            "clarity": {"model_a": 7.9, "model_b": 8.2, "p_value": 0.087},
        },
        "costs": {"total": 45.67, "model_a": 22.34, "model_b": 23.33},
        "response_times": {"Model A": 1850, "Model B": 2100},
    }

    # Winner announcement
    st.markdown("### 🏆 Test Results")

    winner_color = "success" if results["summary"]["statistical_significance"] else "warning"

    st.markdown(
        f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {'#d4edda' if winner_color == 'success' else '#fff3cd'}; border: 1px solid {'#c3e6cb' if winner_color == 'success' else '#ffeaa7'};">
        <h4>🏆 Winner: {results['summary']['winner']}</h4>
        <p><strong>Confidence:</strong> {results['summary']['confidence']:.1%}</p>
        <p><strong>Statistical Significance:</strong> {'✅ Yes' if results['summary']['statistical_significance'] else '❌ No'} (p = {results['summary']['p_value']:.3f})</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key metrics
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model A Score", f"{results['summary']['model_a_score']:.1f}")
    with col2:
        st.metric("Model B Score", f"{results['summary']['model_b_score']:.1f}")
    with col3:
        st.metric("Total Cost", f"${results['costs']['total']:.2f}")
    with col4:
        st.metric("Samples", results["summary"]["total_samples"])

    # Performance comparison charts
    st.markdown("### 📈 Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart
        model_a_data = {
            "name": "Model A",
            "accuracy": results["dimensions"]["accuracy"]["model_a"],
            "helpfulness": results["dimensions"]["helpfulness"]["model_a"],
            "clarity": results["dimensions"]["clarity"]["model_a"],
        }

        model_b_data = {
            "name": "Model B",
            "accuracy": results["dimensions"]["accuracy"]["model_b"],
            "helpfulness": results["dimensions"]["helpfulness"]["model_b"],
            "clarity": results["dimensions"]["clarity"]["model_b"],
        }

        render_performance_comparison_chart(
            model_a_data, model_b_data, ["accuracy", "helpfulness", "clarity"]
        )

    with col2:
        render_response_time_chart(results["response_times"])

    # Statistical analysis
    st.markdown("### 🔬 Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        p_values = {dim: data["p_value"] for dim, data in results["dimensions"].items()}
        render_statistical_significance_chart(p_values)

    with col2:
        # Mock score distributions
        import numpy as np

        model_a_scores = np.random.normal(results["summary"]["model_a_score"], 1.0, 100)
        model_b_scores = np.random.normal(results["summary"]["model_b_score"], 1.0, 100)
        render_score_distribution_chart(model_a_scores, model_b_scores)

    # Export options
    st.markdown("### 📥 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export CSV", use_container_width=True):
            st.success("CSV export requested")

    with col2:
        if st.button("📊 Export Excel", use_container_width=True):
            st.success("Excel export requested")

    with col3:
        if st.button("📋 Generate Report", use_container_width=True):
            st.success("Report generation requested")


def render_failed_test_details(test: Dict[str, Any]):
    """Render details for failed test."""

    st.markdown("### ❌ Test Failed")

    st.error("This test failed to complete successfully.")

    # Mock error details
    error_details = {
        "error_type": "API Timeout",
        "error_message": "Model provider API timed out after 300 seconds",
        "failed_at": "Sample 45/100",
        "timestamp": "2024-01-15T14:30:00Z",
    }

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Error Details:**")
        st.write(f"Type: {error_details['error_type']}")
        st.write(f"Message: {error_details['error_message']}")
        st.write(f"Failed at: {error_details['failed_at']}")
        st.write(f"Time: {error_details['timestamp']}")

    with col2:
        st.markdown("**Recovery Options:**")
        if st.button("🔄 Retry Test", use_container_width=True):
            st.info("Test retry requested")
        if st.button("📝 Edit Configuration", use_container_width=True):
            st.info("Navigate to edit configuration")


def render_test_list():
    """Render list of all tests."""

    st.markdown("### 📋 All Tests")

    # Mock test data
    tests_data = [
        {
            "name": "GPT-4 vs Claude Creative",
            "status": "Completed",
            "winner": "Claude-3-Opus",
            "samples": 100,
            "cost": 45.67,
            "created": "2024-01-15",
        },
        {
            "name": "Code Generation Eval",
            "status": "Running",
            "winner": "TBD",
            "samples": 150,
            "cost": 23.45,
            "created": "2024-01-15",
        },
        {
            "name": "Translation Quality",
            "status": "Failed",
            "winner": "N/A",
            "samples": 75,
            "cost": 12.34,
            "created": "2024-01-14",
        },
    ]

    df = pd.DataFrame(tests_data)

    # Add color coding for status
    def color_status(val):
        colors = {
            "Completed": "background-color: #d4edda",
            "Running": "background-color: #d1ecf1",
            "Failed": "background-color: #f8d7da",
        }
        return colors.get(val, "")

    styled_df = df.style.applymap(color_status, subset=["status"])

    st.dataframe(styled_df, use_container_width=True)

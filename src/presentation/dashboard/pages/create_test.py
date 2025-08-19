"""Create test page."""

import json
from typing import Any, Dict, List, Optional

import streamlit as st

from ..components.forms import render_sample_upload_form, render_test_configuration_form


def render_create_test_page():
    """Render test creation page."""

    st.title("🧪 Create New Test")

    # Progress indicator
    if "test_creation_step" not in st.session_state:
        st.session_state.test_creation_step = 1

    # Step indicator
    steps = ["Configuration", "Samples", "Review", "Launch"]
    current_step = st.session_state.test_creation_step

    cols = st.columns(len(steps))
    for i, step in enumerate(steps, 1):
        with cols[i - 1]:
            if i < current_step:
                st.markdown(f"✅ **{i}. {step}**")
            elif i == current_step:
                st.markdown(f"🔄 **{i}. {step}**")
            else:
                st.markdown(f"⭕ {i}. {step}")

    st.markdown("---")

    # Render current step
    if current_step == 1:
        render_configuration_step()
    elif current_step == 2:
        render_samples_step()
    elif current_step == 3:
        render_review_step()
    elif current_step == 4:
        render_launch_step()


def render_configuration_step():
    """Render test configuration step."""

    st.markdown("### Step 1: Test Configuration")
    st.markdown("Configure your A/B test settings and model parameters.")

    # Render configuration form
    config = render_test_configuration_form()

    if config:
        # Store configuration in session state
        st.session_state.test_config = config

        # Show success message and preview
        st.success("✅ Test configuration saved!")

        with st.expander("Configuration Preview"):
            st.json(config)

        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Next: Add Samples →", use_container_width=True):
                st.session_state.test_creation_step = 2
                st.rerun()


def render_samples_step():
    """Render test samples step."""

    st.markdown("### Step 2: Test Samples")
    st.markdown("Add test samples that will be used to compare the models.")

    # Show current config summary
    if "test_config" in st.session_state:
        config = st.session_state.test_config
        with st.expander("Test Configuration Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Test Name:** {config['name']}")
                st.write(f"**Model A:** {config['model_a']['model_id']}")
                st.write(f"**Model B:** {config['model_b']['model_id']}")
            with col2:
                st.write(f"**Sample Size:** {config['sample_size']}")
                st.write(f"**Difficulty:** {config['difficulty_level'].title()}")
                st.write(f"**Max Cost:** ${config['max_cost']:.2f}")

    # Render sample upload form
    samples = render_sample_upload_form()

    if samples:
        # Store samples in session state
        st.session_state.test_samples = samples

        st.success(f"✅ {len(samples)} samples added!")

        # Show sample summary
        with st.expander("Sample Summary"):
            for i, sample in enumerate(samples[:3], 1):
                st.markdown(f"**Sample {i}:**")
                st.write(f"Prompt: {sample['prompt'][:100]}...")
                if sample.get("expected_response"):
                    st.write(f"Expected: {sample['expected_response'][:100]}...")
                st.markdown("---")

            if len(samples) > 3:
                st.write(f"... and {len(samples) - 3} more samples")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back: Configuration", use_container_width=True):
            st.session_state.test_creation_step = 1
            st.rerun()

    with col2:
        if st.session_state.get("test_samples"):
            if st.button("Next: Review →", use_container_width=True):
                st.session_state.test_creation_step = 3
                st.rerun()
        else:
            st.button("Next: Review →", disabled=True, use_container_width=True)
            st.caption("Add samples to continue")


def render_review_step():
    """Render test review step."""

    st.markdown("### Step 3: Review Test")
    st.markdown("Review your test configuration before launching.")

    if "test_config" not in st.session_state or "test_samples" not in st.session_state:
        st.error(
            "Missing test configuration or samples. Please go back and complete previous steps."
        )
        return

    config = st.session_state.test_config
    samples = st.session_state.test_samples

    # Test summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Test Configuration")
        st.write(f"**Name:** {config['name']}")
        st.write(f"**Description:** {config.get('description', 'N/A')}")
        st.write(f"**Difficulty:** {config['difficulty_level'].title()}")
        st.write(f"**Evaluation:** {config['evaluation_template']}")
        st.write(f"**Sample Size:** {config['sample_size']}")
        st.write(f"**Actual Samples:** {len(samples)}")

        # Tags
        if config.get("tags"):
            st.write(f"**Tags:** {', '.join(config['tags'])}")

    with col2:
        st.markdown("#### Model Configuration")

        # Model A
        st.markdown("**Model A:**")
        st.write(f"- Provider: {config['model_a']['provider'].title()}")
        st.write(f"- Model: {config['model_a']['model_id']}")
        st.write(f"- Temperature: {config['model_a']['parameters']['temperature']}")
        st.write(f"- Max Tokens: {config['model_a']['parameters']['max_tokens']}")

        # Model B
        st.markdown("**Model B:**")
        st.write(f"- Provider: {config['model_b']['provider'].title()}")
        st.write(f"- Model: {config['model_b']['model_id']}")
        st.write(f"- Temperature: {config['model_b']['parameters']['temperature']}")
        st.write(f"- Max Tokens: {config['model_b']['parameters']['max_tokens']}")

    # Execution settings
    st.markdown("#### Execution Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Max Cost", f"${config['max_cost']:.2f}")
    with col2:
        st.metric("Concurrent Workers", config["concurrent_workers"])
    with col3:
        st.metric("Timeout", f"{config['timeout_seconds']}s")

    # Sample preview
    st.markdown("#### Sample Preview")
    with st.expander(f"View {len(samples)} samples"):
        for i, sample in enumerate(samples[:5], 1):
            st.markdown(f"**Sample {i}:**")
            st.write(f"Prompt: {sample['prompt']}")
            if sample.get("expected_response"):
                st.write(f"Expected: {sample['expected_response']}")
            if sample.get("context"):
                st.write(f"Context: {sample['context']}")
            st.markdown("---")

        if len(samples) > 5:
            st.write(f"... and {len(samples) - 5} more samples")

    # Cost estimation
    st.markdown("#### Cost Estimation")
    estimated_cost = estimate_test_cost(config, samples)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Cost", f"${estimated_cost:.2f}")
    with col2:
        cost_per_sample = estimated_cost / len(samples)
        st.metric("Cost per Sample", f"${cost_per_sample:.3f}")
    with col3:
        if estimated_cost > config["max_cost"]:
            st.error(f"⚠️ Exceeds budget by ${estimated_cost - config['max_cost']:.2f}")
        else:
            remaining = config["max_cost"] - estimated_cost
            st.success(f"✅ Within budget (${remaining:.2f} remaining)")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back: Samples", use_container_width=True):
            st.session_state.test_creation_step = 2
            st.rerun()

    with col2:
        if estimated_cost <= config["max_cost"]:
            if st.button("Launch Test →", use_container_width=True, type="primary"):
                st.session_state.test_creation_step = 4
                st.rerun()
        else:
            st.button("Launch Test →", disabled=True, use_container_width=True)
            st.caption("Cost exceeds budget")


def render_launch_step():
    """Render test launch step."""

    st.markdown("### Step 4: Launch Test")

    if "test_config" not in st.session_state or "test_samples" not in st.session_state:
        st.error("Missing test configuration or samples.")
        return

    config = st.session_state.test_config
    samples = st.session_state.test_samples

    # Launch confirmation
    st.markdown("#### Final Confirmation")
    st.warning("⚠️ Once launched, the test will begin processing samples and incurring costs.")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Test:** {config['name']}")
        st.info(f"**Samples:** {len(samples)} samples")
    with col2:
        estimated_cost = estimate_test_cost(config, samples)
        st.info(f"**Estimated Cost:** ${estimated_cost:.2f}")
        st.info(f"**Models:** {config['model_a']['model_id']} vs {config['model_b']['model_id']}")

    # Launch button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Test Now", use_container_width=True, type="primary"):
            # Simulate test creation
            test_id = create_test(config, samples)

            if test_id:
                st.success("🎉 Test launched successfully!")
                st.markdown(f"**Test ID:** `{test_id}`")
                st.markdown("You can monitor the test progress in the Test Results page.")

                # Clear session state
                if "test_config" in st.session_state:
                    del st.session_state.test_config
                if "test_samples" in st.session_state:
                    del st.session_state.test_samples
                st.session_state.test_creation_step = 1

                # Add buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("View Test Results", use_container_width=True):
                        st.session_state.selected_test = test_id
                        # Would navigate to results page
                        st.info("Navigate to Test Results page to monitor progress")

                with col2:
                    if st.button("Create Another Test", use_container_width=True):
                        st.rerun()
            else:
                st.error("Failed to create test. Please try again.")

    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back: Review", use_container_width=True):
            st.session_state.test_creation_step = 3
            st.rerun()


def estimate_test_cost(config: Dict[str, Any], samples: List[Dict[str, Any]]) -> float:
    """Estimate test cost based on configuration and samples."""

    # Mock cost calculation
    base_cost_per_sample = 0.02

    # Model-specific multipliers
    model_costs = {
        "gpt-4": 1.5,
        "gpt-3.5-turbo": 0.5,
        "claude-3-opus": 1.2,
        "claude-3-sonnet": 0.8,
        "claude-3-haiku": 0.4,
        "gemini-pro": 0.6,
    }

    model_a_cost = model_costs.get(config["model_a"]["model_id"], 1.0)
    model_b_cost = model_costs.get(config["model_b"]["model_id"], 1.0)

    # Calculate cost
    total_cost = 0
    for sample in samples:
        prompt_length = len(sample["prompt"])
        length_multiplier = max(1.0, prompt_length / 100)  # Longer prompts cost more

        sample_cost = base_cost_per_sample * length_multiplier
        sample_cost *= model_a_cost + model_b_cost

        total_cost += sample_cost

    # Add evaluation cost
    evaluation_cost = len(samples) * 0.01  # $0.01 per evaluation
    total_cost += evaluation_cost

    return total_cost


def create_test(config: Dict[str, Any], samples: List[Dict[str, Any]]) -> str:
    """Create test with given configuration and samples."""

    # Mock test creation - in production, call API
    import uuid

    test_id = str(uuid.uuid4())

    # Simulate test creation delay
    import time

    time.sleep(2)

    # Store test in session state for demo
    if "created_tests" not in st.session_state:
        st.session_state.created_tests = []

    test_data = {
        "id": test_id,
        "config": config,
        "samples": samples,
        "status": "running",
        "created_at": "2024-01-15T10:00:00Z",
        "progress": {"completed": 0, "total": len(samples)},
    }

    st.session_state.created_tests.append(test_data)

    return test_id

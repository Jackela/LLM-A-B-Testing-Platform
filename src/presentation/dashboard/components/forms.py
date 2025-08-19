"""Interactive forms for the dashboard."""

import json
from typing import Any, Dict, List, Optional

import streamlit as st


def render_test_configuration_form() -> Optional[Dict[str, Any]]:
    """Render test configuration form."""

    st.subheader("Test Configuration")

    with st.form("test_config_form"):
        # Basic Information
        st.markdown("#### Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            test_name = st.text_input(
                "Test Name*",
                placeholder="e.g., GPT-4 vs Claude Creative Writing",
                help="Choose a descriptive name for your test",
            )

            sample_size = st.number_input(
                "Sample Size*",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                help="Number of test samples to process",
            )

        with col2:
            difficulty_level = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard", "Expert"],
                index=1,
                help="Complexity level of test samples",
            )

            evaluation_template = st.selectbox(
                "Evaluation Template*",
                [
                    "Standard Evaluation",
                    "Creative Writing Evaluation",
                    "Factual Accuracy Evaluation",
                    "Custom",
                ],
                help="Choose evaluation criteria template",
            )

        description = st.text_area(
            "Description",
            placeholder="Describe the purpose and goals of this test...",
            help="Optional description of the test objectives",
        )

        # Model Configuration
        st.markdown("#### Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model A**")
            model_a_provider = st.selectbox(
                "Provider A",
                ["OpenAI", "Anthropic", "Google", "HuggingFace", "Local"],
                key="provider_a",
            )

            model_a_options = {
                "OpenAI": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "Google": ["gemini-pro", "gemini-pro-vision"],
                "HuggingFace": ["llama-2-70b", "mistral-7b"],
                "Local": ["local-model-1", "local-model-2"],
            }

            model_a_id = st.selectbox(
                "Model A", model_a_options.get(model_a_provider, []), key="model_a"
            )

            # Model A Parameters
            with st.expander("Model A Parameters"):
                temp_a = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="temp_a")
                max_tokens_a = st.number_input("Max Tokens", 1, 8192, 2048, key="tokens_a")
                top_p_a = st.slider("Top P", 0.0, 1.0, 1.0, 0.1, key="top_p_a")

        with col2:
            st.markdown("**Model B**")
            model_b_provider = st.selectbox(
                "Provider B",
                ["OpenAI", "Anthropic", "Google", "HuggingFace", "Local"],
                key="provider_b",
            )

            model_b_id = st.selectbox(
                "Model B", model_a_options.get(model_b_provider, []), key="model_b"
            )

            # Model B Parameters
            with st.expander("Model B Parameters"):
                temp_b = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="temp_b")
                max_tokens_b = st.number_input("Max Tokens", 1, 8192, 2048, key="tokens_b")
                top_p_b = st.slider("Top P", 0.0, 1.0, 1.0, 0.1, key="top_p_b")

        # Advanced Configuration
        with st.expander("Advanced Configuration"):
            col1, col2 = st.columns(2)

            with col1:
                max_cost = st.number_input(
                    "Maximum Cost ($)",
                    min_value=1.0,
                    max_value=1000.0,
                    value=100.0,
                    step=10.0,
                    help="Maximum budget for this test",
                )

                concurrent_workers = st.number_input(
                    "Concurrent Workers",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of parallel workers for test execution",
                )

            with col2:
                timeout_seconds = st.number_input(
                    "Timeout (seconds)",
                    min_value=30,
                    max_value=3600,
                    value=300,
                    help="Maximum time to wait for each model response",
                )

                tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="e.g., creative, comparison, gpt-4",
                    help="Tags for organizing and filtering tests",
                )

        # Submit button
        submitted = st.form_submit_button("Create Test", use_container_width=True)

        if submitted:
            # Validation
            if not test_name:
                st.error("Test name is required")
                return None

            if not model_a_id or not model_b_id:
                st.error("Both models must be selected")
                return None

            # Create configuration dictionary
            config = {
                "name": test_name,
                "description": description,
                "sample_size": sample_size,
                "difficulty_level": difficulty_level.lower(),
                "evaluation_template": evaluation_template,
                "model_a": {
                    "provider": model_a_provider.lower(),
                    "model_id": model_a_id,
                    "parameters": {
                        "temperature": temp_a,
                        "max_tokens": max_tokens_a,
                        "top_p": top_p_a,
                    },
                },
                "model_b": {
                    "provider": model_b_provider.lower(),
                    "model_id": model_b_id,
                    "parameters": {
                        "temperature": temp_b,
                        "max_tokens": max_tokens_b,
                        "top_p": top_p_b,
                    },
                },
                "max_cost": max_cost,
                "concurrent_workers": concurrent_workers,
                "timeout_seconds": timeout_seconds,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            }

            return config

    return None


def render_sample_upload_form() -> Optional[List[Dict[str, Any]]]:
    """Render sample upload form."""

    st.subheader("Test Samples")

    upload_method = st.radio(
        "Upload Method", ["Manual Entry", "File Upload", "Generate Samples"], horizontal=True
    )

    samples = []

    if upload_method == "Manual Entry":
        with st.form("manual_samples_form"):
            st.markdown("#### Add Samples Manually")

            num_samples = st.number_input(
                "Number of samples to add", min_value=1, max_value=20, value=3
            )

            for i in range(num_samples):
                st.markdown(f"**Sample {i+1}**")
                col1, col2 = st.columns([2, 1])

                with col1:
                    prompt = st.text_area(
                        f"Prompt {i+1}",
                        placeholder="Enter the prompt for the models to respond to...",
                        key=f"prompt_{i}",
                        height=100,
                    )

                with col2:
                    expected_response = st.text_area(
                        f"Expected Response (optional)",
                        placeholder="Enter expected response if available...",
                        key=f"expected_{i}",
                        height=100,
                    )

                context = st.text_input(
                    f"Context (optional)",
                    placeholder="Additional context for this sample...",
                    key=f"context_{i}",
                )

                if prompt:
                    samples.append(
                        {
                            "prompt": prompt,
                            "expected_response": expected_response or None,
                            "context": context or None,
                            "metadata": {},
                        }
                    )

                st.markdown("---")

            if st.form_submit_button("Add Samples"):
                if samples:
                    return samples
                else:
                    st.error("Please add at least one sample with a prompt")

    elif upload_method == "File Upload":
        st.markdown("#### Upload Samples from File")

        uploaded_file = st.file_uploader(
            "Choose a CSV or JSON file",
            type=["csv", "json"],
            help="File should contain columns: prompt, expected_response (optional), context (optional)",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    import pandas as pd

                    df = pd.read_csv(uploaded_file)

                    # Validate required columns
                    if "prompt" not in df.columns:
                        st.error("CSV file must contain a 'prompt' column")
                        return None

                    for _, row in df.iterrows():
                        samples.append(
                            {
                                "prompt": row["prompt"],
                                "expected_response": row.get("expected_response"),
                                "context": row.get("context"),
                                "metadata": {},
                            }
                        )

                elif uploaded_file.name.endswith(".json"):
                    data = json.load(uploaded_file)

                    if isinstance(data, list):
                        samples = data
                    else:
                        st.error("JSON file should contain an array of sample objects")
                        return None

                st.success(f"Loaded {len(samples)} samples from file")

                # Preview samples
                if samples:
                    with st.expander("Preview Samples"):
                        for i, sample in enumerate(samples[:5]):
                            st.markdown(f"**Sample {i+1}:**")
                            st.write(f"Prompt: {sample['prompt'][:100]}...")
                            if sample.get("expected_response"):
                                st.write(f"Expected: {sample['expected_response'][:100]}...")
                            st.markdown("---")

                        if len(samples) > 5:
                            st.write(f"... and {len(samples) - 5} more samples")

                if st.button("Use These Samples"):
                    return samples

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    elif upload_method == "Generate Samples":
        st.markdown("#### Generate Samples Automatically")

        with st.form("generate_samples_form"):
            sample_type = st.selectbox(
                "Sample Type",
                [
                    "Creative Writing",
                    "Question Answering",
                    "Code Generation",
                    "Summarization",
                    "Translation",
                ],
            )

            num_generate = st.number_input(
                "Number of samples to generate", min_value=10, max_value=100, value=20
            )

            if st.form_submit_button("Generate Samples"):
                # Mock sample generation
                for i in range(num_generate):
                    if sample_type == "Creative Writing":
                        prompt = f"Write a creative short story about {['space exploration', 'time travel', 'magical creatures', 'post-apocalyptic world'][i % 4]}"
                    elif sample_type == "Question Answering":
                        prompt = f"What is {['the capital of France', 'quantum computing', 'photosynthesis', 'machine learning'][i % 4]}?"
                    elif sample_type == "Code Generation":
                        prompt = f"Write a Python function to {['sort a list', 'find prime numbers', 'calculate fibonacci', 'reverse a string'][i % 4]}"
                    else:
                        prompt = f"Sample prompt {i+1} for {sample_type}"

                    samples.append(
                        {
                            "prompt": prompt,
                            "expected_response": None,
                            "context": None,
                            "metadata": {"generated": True, "type": sample_type},
                        }
                    )

                st.success(f"Generated {len(samples)} samples")
                return samples

    return None


def render_test_filters_form() -> Dict[str, Any]:
    """Render test filtering form."""

    with st.sidebar:
        st.markdown("### Filter Tests")

        # Status filter
        status_filter = st.multiselect(
            "Status",
            ["Draft", "Ready", "Running", "Paused", "Completed", "Failed", "Cancelled"],
            default=["Completed", "Running"],
        )

        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")

        # Other filters
        difficulty_filter = st.multiselect("Difficulty", ["Easy", "Medium", "Hard", "Expert"])

        tags_filter = st.text_input("Tags (comma-separated)", placeholder="e.g., gpt-4, creative")

        created_by_filter = st.text_input("Created By", placeholder="Username")

        if st.button("Apply Filters", use_container_width=True):
            return {
                "status": [s.lower() for s in status_filter],
                "start_date": start_date,
                "end_date": end_date,
                "difficulty": [d.lower() for d in difficulty_filter],
                "tags": [tag.strip() for tag in tags_filter.split(",") if tag.strip()],
                "created_by": created_by_filter if created_by_filter else None,
            }

    return {}

"""Configuration validators for CLI operations."""

from typing import Any, Dict, List


def validate_test_config(config: Dict[str, Any]) -> List[str]:
    """Validate test configuration.

    Args:
        config: Test configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["name", "model_a", "model_b"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate models
    if "model_a" in config:
        model_errors = validate_model_config(config["model_a"], "model_a")
        errors.extend(model_errors)

    if "model_b" in config:
        model_errors = validate_model_config(config["model_b"], "model_b")
        errors.extend(model_errors)

    # Validate evaluation template
    if "evaluation_template_id" in config:
        template_id = config["evaluation_template_id"]
        if not isinstance(template_id, str) or not template_id.strip():
            errors.append("evaluation_template_id must be a non-empty string")

    # Validate max cost
    if "max_cost" in config:
        max_cost = config["max_cost"]
        if not isinstance(max_cost, (int, float)) or max_cost <= 0:
            errors.append("max_cost must be a positive number")

    # Validate samples
    if "samples" in config:
        samples_errors = validate_samples(config["samples"])
        errors.extend(samples_errors)

    return errors


def validate_model_config(model_config: Dict[str, Any], model_name: str) -> List[str]:
    """Validate model configuration.

    Args:
        model_config: Model configuration dictionary
        model_name: Name of the model for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["provider", "model_id"]
    for field in required_fields:
        if field not in model_config:
            errors.append(f"{model_name}: Missing required field '{field}'")

    # Validate provider
    if "provider" in model_config:
        provider = model_config["provider"]
        if not isinstance(provider, str) or not provider.strip():
            errors.append(f"{model_name}: provider must be a non-empty string")

    # Validate model_id
    if "model_id" in model_config:
        model_id = model_config["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"{model_name}: model_id must be a non-empty string")

    # Validate parameters
    if "parameters" in model_config:
        param_errors = validate_model_parameters(model_config["parameters"], model_name)
        errors.extend(param_errors)

    return errors


def validate_model_parameters(parameters: Dict[str, Any], model_name: str) -> List[str]:
    """Validate model parameters.

    Args:
        parameters: Model parameters dictionary
        model_name: Name of the model for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    # Validate temperature
    if "temperature" in parameters:
        temp = parameters["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            errors.append(f"{model_name}: temperature must be between 0 and 2")

    # Validate max_tokens
    if "max_tokens" in parameters:
        max_tokens = parameters["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 100000:
            errors.append(f"{model_name}: max_tokens must be between 1 and 100000")

    # Validate top_p
    if "top_p" in parameters:
        top_p = parameters["top_p"]
        if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
            errors.append(f"{model_name}: top_p must be between 0 and 1")

    # Validate top_k
    if "top_k" in parameters:
        top_k = parameters["top_k"]
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            errors.append(f"{model_name}: top_k must be between 1 and 1000")

    return errors


def validate_samples(samples: List[Dict[str, Any]]) -> List[str]:
    """Validate test samples.

    Args:
        samples: List of sample dictionaries

    Returns:
        List of validation error messages
    """
    errors = []

    if not isinstance(samples, list):
        return ["samples must be a list"]

    if len(samples) == 0:
        return ["samples list cannot be empty"]

    # Validate each sample
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"Sample {i}: must be a dictionary")
            continue

        # Check for required fields
        if "input" not in sample:
            errors.append(f"Sample {i}: missing required field 'input'")

        # Validate input
        if "input" in sample:
            input_value = sample["input"]
            if not isinstance(input_value, str) or not input_value.strip():
                errors.append(f"Sample {i}: input must be a non-empty string")

        # Validate expected_output if present
        if "expected_output" in sample:
            expected = sample["expected_output"]
            if not isinstance(expected, str):
                errors.append(f"Sample {i}: expected_output must be a string")

        # Validate metadata if present
        if "metadata" in sample:
            metadata = sample["metadata"]
            if not isinstance(metadata, dict):
                errors.append(f"Sample {i}: metadata must be a dictionary")

    return errors


def validate_provider_config(config: Dict[str, Any]) -> List[str]:
    """Validate provider configuration.

    Args:
        config: Provider configuration dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["name", "type", "api_key"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate name
    if "name" in config:
        name = config["name"]
        if not isinstance(name, str) or not name.strip():
            errors.append("name must be a non-empty string")

    # Validate type
    if "type" in config:
        provider_type = config["type"]
        valid_types = ["openai", "anthropic", "google", "azure", "aws"]
        if provider_type not in valid_types:
            errors.append(f"type must be one of: {', '.join(valid_types)}")

    # Validate api_key
    if "api_key" in config:
        api_key = config["api_key"]
        if not isinstance(api_key, str) or not api_key.strip():
            errors.append("api_key must be a non-empty string")

    # Validate base_url if present
    if "base_url" in config:
        base_url = config["base_url"]
        if not isinstance(base_url, str) or not base_url.strip():
            errors.append("base_url must be a non-empty string")

    return errors


def validate_evaluation_template(template: Dict[str, Any]) -> List[str]:
    """Validate evaluation template configuration.

    Args:
        template: Evaluation template dictionary

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["name", "dimensions"]
    for field in required_fields:
        if field not in template:
            errors.append(f"Missing required field: {field}")

    # Validate name
    if "name" in template:
        name = template["name"]
        if not isinstance(name, str) or not name.strip():
            errors.append("name must be a non-empty string")

    # Validate dimensions
    if "dimensions" in template:
        dimensions = template["dimensions"]
        if not isinstance(dimensions, list) or len(dimensions) == 0:
            errors.append("dimensions must be a non-empty list")
        else:
            for i, dimension in enumerate(dimensions):
                dim_errors = validate_evaluation_dimension(dimension, i)
                errors.extend(dim_errors)

    return errors


def validate_evaluation_dimension(dimension: Dict[str, Any], index: int) -> List[str]:
    """Validate evaluation dimension.

    Args:
        dimension: Evaluation dimension dictionary
        index: Index of the dimension for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["name", "description", "scale"]
    for field in required_fields:
        if field not in dimension:
            errors.append(f"Dimension {index}: missing required field '{field}'")

    # Validate name
    if "name" in dimension:
        name = dimension["name"]
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Dimension {index}: name must be a non-empty string")

    # Validate description
    if "description" in dimension:
        description = dimension["description"]
        if not isinstance(description, str) or not description.strip():
            errors.append(f"Dimension {index}: description must be a non-empty string")

    # Validate scale
    if "scale" in dimension:
        scale = dimension["scale"]
        if not isinstance(scale, dict):
            errors.append(f"Dimension {index}: scale must be a dictionary")
        else:
            scale_errors = validate_evaluation_scale(scale, index)
            errors.extend(scale_errors)

    return errors


def validate_evaluation_scale(scale: Dict[str, Any], dimension_index: int) -> List[str]:
    """Validate evaluation scale.

    Args:
        scale: Evaluation scale dictionary
        dimension_index: Index of the dimension for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    # Required fields
    required_fields = ["min", "max", "type"]
    for field in required_fields:
        if field not in scale:
            errors.append(f"Dimension {dimension_index} scale: missing required field '{field}'")

    # Validate min and max
    if "min" in scale and "max" in scale:
        min_val = scale["min"]
        max_val = scale["max"]

        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            errors.append(f"Dimension {dimension_index} scale: min and max must be numbers")
        elif min_val >= max_val:
            errors.append(f"Dimension {dimension_index} scale: min must be less than max")

    # Validate type
    if "type" in scale:
        scale_type = scale["type"]
        valid_types = ["numeric", "categorical", "binary"]
        if scale_type not in valid_types:
            errors.append(
                f"Dimension {dimension_index} scale: type must be one of: {', '.join(valid_types)}"
            )

    return errors

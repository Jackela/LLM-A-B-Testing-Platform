"""Configuration utilities for CLI."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    config = _expand_env_vars(config)

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid
    """
    required_sections = ["api", "logging", "defaults"]

    for section in required_sections:
        if section not in config:
            return False

    # Validate API section
    api_config = config["api"]
    if "base_url" not in api_config:
        return False

    # Validate logging section
    logging_config = config["logging"]
    if "level" not in logging_config:
        return False

    # Validate defaults section
    defaults_config = config["defaults"]
    required_defaults = ["max_cost", "timeout_seconds", "output_format"]
    for default in required_defaults:
        if default not in defaults_config:
            return False

    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'api.base_url')
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override with

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in configuration.

    Args:
        obj: Configuration object (dict, list, or string)

    Returns:
        Object with expanded environment variables
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def get_default_config_path() -> Path:
    """Get default configuration file path.

    Returns:
        Default configuration file path
    """
    # Try current directory first
    local_config = Path.cwd() / "llm-test-config.yaml"
    if local_config.exists():
        return local_config

    # Try user config directory
    user_config_dir = Path.home() / ".config" / "llm-test"
    user_config = user_config_dir / "config.yaml"
    if user_config.exists():
        return user_config

    # Return local path as default
    return local_config


def load_default_config() -> Optional[Dict[str, Any]]:
    """Load configuration from default locations.

    Returns:
        Configuration dictionary or None if not found
    """
    config_path = get_default_config_path()

    try:
        return load_config(str(config_path))
    except FileNotFoundError:
        return None


def create_user_config_dir() -> Path:
    """Create user configuration directory.

    Returns:
        Path to user configuration directory
    """
    user_config_dir = Path.home() / ".config" / "llm-test"
    user_config_dir.mkdir(parents=True, exist_ok=True)
    return user_config_dir


def get_cache_dir() -> Path:
    """Get cache directory path.

    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".cache" / "llm-test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_log_dir() -> Path:
    """Get log directory path.

    Returns:
        Path to log directory
    """
    log_dir = Path.home() / ".local" / "share" / "llm-test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

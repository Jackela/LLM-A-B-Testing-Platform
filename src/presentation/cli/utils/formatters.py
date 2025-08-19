"""Output formatting utilities for CLI."""

import json
from typing import Any, Dict, List

import yaml

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def format_output(data: Any, format_type: str) -> str:
    """Format data for output.

    Args:
        data: Data to format
        format_type: Output format ('json', 'yaml', 'table')

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    elif format_type == "yaml":
        return yaml.dump(data, default_flow_style=False, indent=2)
    elif format_type == "table":
        if isinstance(data, dict):
            return format_dict_as_table(data)
        elif isinstance(data, list):
            return format_list_as_table(data)
        else:
            return str(data)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def format_dict_as_table(data: Dict[str, Any], max_depth: int = 2) -> str:
    """Format dictionary as table.

    Args:
        data: Dictionary to format
        max_depth: Maximum nesting depth to display

    Returns:
        Formatted table string
    """
    rows = []

    def flatten_dict(d: Dict[str, Any], prefix: str = "", depth: int = 0):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict) and depth < max_depth:
                flatten_dict(value, full_key, depth + 1)
            elif isinstance(value, list) and len(value) <= 5:
                rows.append([full_key, ", ".join(map(str, value))])
            else:
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                rows.append([full_key, str_value])

    flatten_dict(data)

    if rows:
        if HAS_TABULATE:
            return tabulate(rows, headers=["Key", "Value"], tablefmt="grid")
        else:
            # Fallback formatting without tabulate
            result = "Key".ljust(30) + " | Value\n"
            result += "-" * 50 + "\n"
            for row in rows:
                result += f"{str(row[0])[:30].ljust(30)} | {row[1]}\n"
            return result
    else:
        return "No data to display"


def format_list_as_table(data: List[Dict[str, Any]]) -> str:
    """Format list of dictionaries as table.

    Args:
        data: List of dictionaries

    Returns:
        Formatted table string
    """
    if not data:
        return "No data to display"

    # Get all unique keys
    all_keys = set()
    for item in data:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    if not all_keys:
        return "No data to display"

    # Sort keys for consistent column order
    headers = sorted(all_keys)

    # Build rows
    rows = []
    for item in data:
        if isinstance(item, dict):
            row = []
            for header in headers:
                value = item.get(header, "")
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 30:
                    str_value = str_value[:27] + "..."
                row.append(str_value)
            rows.append(row)

    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="grid")
    else:
        # Fallback formatting without tabulate
        if not headers:
            return "No data to display"

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format header
        result = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + "\n"
        result += "-" * sum(col_widths) + "-" * (len(headers) - 1) * 3 + "\n"

        # Format rows
        for row in rows:
            result += (
                " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + "\n"
            )

        return result


def format_table(data: List[List[str]], headers: List[str], tablefmt: str = "grid") -> str:
    """Format data as table with custom headers.

    Args:
        data: Table data (list of rows)
        headers: Column headers
        tablefmt: Table format style

    Returns:
        Formatted table string
    """
    if HAS_TABULATE:
        return tabulate(data, headers=headers, tablefmt=tablefmt)
    else:
        # Fallback formatting
        if not headers:
            return "No data to display"

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format header
        result = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + "\n"
        result += "-" * sum(col_widths) + "-" * (len(headers) - 1) * 3 + "\n"

        # Format rows
        for row in data:
            result += (
                " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + "\n"
            )

        return result


def format_json_pretty(data: Any) -> str:
    """Format data as pretty-printed JSON.

    Args:
        data: Data to format

    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def format_yaml_pretty(data: Any) -> str:
    """Format data as pretty-printed YAML.

    Args:
        data: Data to format

    Returns:
        Pretty-printed YAML string
    """
    return yaml.dump(data, default_flow_style=False, indent=2, sort_keys=True)


def format_size_bytes(size_bytes: int) -> str:
    """Format byte size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration_seconds(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format value as percentage.

    Args:
        value: Value to format (0.0 to 1.0)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency.

    Args:
        amount: Amount to format
        currency: Currency symbol/code

    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string if too long.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Format progress bar.

    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar

    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + " " * width + "]"

    progress = current / total
    filled = int(progress * width)
    bar = "█" * filled + " " * (width - filled)
    percentage = progress * 100

    return f"[{bar}] {percentage:.1f}%"

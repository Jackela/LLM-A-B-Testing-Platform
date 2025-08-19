"""Logging setup utilities for CLI."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_colors: bool = True,
) -> None:
    """Setup logging configuration for CLI.

    Args:
        level: Logging level
        log_file: Optional log file path
        log_format: Optional log format string
        enable_colors: Whether to enable colored output
    """
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    if enable_colors and hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        # Use colored formatter if terminal supports it
        console_handler.setFormatter(ColoredFormatter(log_format))
    else:
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce verbosity of third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with colors
        """
        # Get the original formatted message
        message = super().format(record)

        # Add color based on log level
        color = self.COLORS.get(record.levelname, "")
        if color:
            # Color the level name only
            levelname = record.levelname
            colored_levelname = f"{color}{levelname}{self.RESET}"
            message = message.replace(levelname, colored_levelname, 1)

        return message


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_verbose_logging() -> None:
    """Enable verbose logging (DEBUG level)."""
    logging.getLogger().setLevel(logging.DEBUG)

    # Also enable debug for our modules
    logging.getLogger("llm_test").setLevel(logging.DEBUG)


def set_quiet_logging() -> None:
    """Enable quiet logging (WARNING level only)."""
    logging.getLogger().setLevel(logging.WARNING)

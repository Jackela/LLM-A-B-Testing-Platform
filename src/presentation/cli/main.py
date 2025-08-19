"""Main CLI application for LLM A/B Testing Platform."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import click

from .commands.data_commands import data_cli
from .commands.provider_commands import provider_cli
from .commands.system_commands import system_cli
from .commands.test_commands import test_cli
from .utils.config import load_config, validate_config
from .utils.logging_setup import setup_logging

# Version information
__version__ = "1.0.0"


# Global context for CLI
class CLIContext:
    """Global CLI context."""

    def __init__(self):
        self.config = {}
        self.verbose = False
        self.debug = False
        self.api_base_url = "http://localhost:8000"
        self.api_token = None


# Create global context
pass_context = click.make_pass_decorator(CLIContext, ensure=True)


@click.group(invoke_without_command=True)
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--api-url", default="http://localhost:8000", help="API base URL")
@click.option("--token", "-t", help="API authentication token")
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx, config, verbose, debug, api_url, token):
    """
    LLM A/B Testing Platform CLI

    A comprehensive command-line interface for managing LLM model comparisons,
    tests, data operations, and system administration.

    Examples:
        llm-test create --name "GPT vs Claude" --config test.yaml
        llm-test start --test-id abc123
        llm-test results --test-id abc123 --format json
        llm-test health --verbose
    """
    # Initialize context
    ctx.ensure_object(CLIContext)
    ctx.obj.verbose = verbose
    ctx.obj.debug = debug
    ctx.obj.api_base_url = api_url
    ctx.obj.api_token = token

    # Setup logging
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    setup_logging(log_level)

    # Load configuration
    if config:
        try:
            ctx.obj.config = load_config(config)
            if not validate_config(ctx.obj.config):
                click.echo("Error: Invalid configuration file", err=True)
                sys.exit(1)
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

        # Show quick status if API is available
        try:
            from .utils.api_client import APIClient

            client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
            health = client.get_health()
            if health.get("status") == "healthy":
                click.echo(f"\n✅ API is healthy at {api_url}")
            else:
                click.echo(f"\n⚠️  API status unknown at {api_url}")
        except Exception:
            click.echo(f"\n❌ API not available at {api_url}")


# Add command groups
cli.add_command(test_cli, name="test")
cli.add_command(data_cli, name="data")
cli.add_command(system_cli, name="system")
cli.add_command(provider_cli, name="provider")


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@pass_context
def status(ctx, output_format):
    """Show platform status and overview."""

    try:
        from .utils.api_client import APIClient
        from .utils.formatters import format_output

        client = APIClient(ctx.api_base_url, ctx.api_token)

        # Get system status
        health = client.get_health()
        overview = client.get_dashboard_overview(days=7)

        status_data = {
            "api_status": health.get("status", "unknown"),
            "api_version": health.get("version", "unknown"),
            "total_tests": overview.get("summary_stats", {}).get("total_tests", 0),
            "running_tests": overview.get("summary_stats", {}).get("running_tests", 0),
            "completed_tests": overview.get("summary_stats", {}).get("completed_tests", 0),
            "total_cost": overview.get("summary_stats", {}).get("total_cost", 0),
            "samples_processed": overview.get("summary_stats", {}).get(
                "total_samples_processed", 0
            ),
        }

        click.echo(format_output(status_data, output_format))

    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for configuration template")
def init(output):
    """Initialize configuration file."""

    config_template = {
        "api": {"base_url": "http://localhost:8000", "timeout": 30, "retry_attempts": 3},
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "llm-test.log",
        },
        "defaults": {
            "max_cost": 100.0,
            "timeout_seconds": 300,
            "concurrent_workers": 3,
            "output_format": "table",
        },
        "providers": {
            "openai": {"api_key": "${OPENAI_API_KEY}", "models": ["gpt-4", "gpt-3.5-turbo"]},
            "anthropic": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "models": ["claude-3-opus", "claude-3-sonnet"],
            },
        },
    }

    import yaml

    if output:
        output_path = Path(output)
    else:
        output_path = Path.cwd() / "llm-test-config.yaml"

    try:
        with open(output_path, "w") as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)

        click.echo(f"✅ Configuration template created: {output_path}")
        click.echo("Please edit the configuration file and set your API keys.")

    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file to validate")
def validate(config):
    """Validate configuration file."""

    if not config:
        config = Path.cwd() / "llm-test-config.yaml"
        if not config.exists():
            click.echo("No configuration file found. Use 'llm-test init' to create one.", err=True)
            sys.exit(1)

    try:
        config_data = load_config(config)

        if validate_config(config_data):
            click.echo("✅ Configuration is valid")
        else:
            click.echo("❌ Configuration is invalid", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

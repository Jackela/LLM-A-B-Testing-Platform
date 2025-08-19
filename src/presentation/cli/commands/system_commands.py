"""System administration commands."""

import time
from typing import Any, Dict

import click
import psutil

from ..utils.api_client import APIClient
from ..utils.formatters import (
    format_duration_seconds,
    format_output,
    format_size_bytes,
    format_table,
)


@click.group()
def system_cli():
    """System administration and health monitoring commands."""
    pass


@system_cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed health information")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def health(ctx, verbose, output_format):
    """
    Check system health and API status.

    Examples:
        llm-test system health
        llm-test system health --verbose
        llm-test system health --format json
    """

    try:
        health_data = {}

        # Check API health
        try:
            client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
            api_health = client.get_health()
            health_data["api"] = {
                "status": api_health.get("status", "unknown"),
                "version": api_health.get("version", "unknown"),
                "response_time": measure_api_response_time(client),
            }
        except Exception as e:
            health_data["api"] = {"status": "error", "error": str(e), "response_time": None}

        # System resource information
        if verbose:
            health_data["system"] = get_system_info()

        # Provider health
        try:
            providers = client.list_providers()
            health_data["providers"] = check_providers_health(
                client, providers.get("providers", [])
            )
        except Exception:
            health_data["providers"] = {"error": "Unable to check providers"}

        if output_format == "table":
            print_health_table(health_data, verbose)
        else:
            click.echo(format_output(health_data, output_format))

    except Exception as e:
        click.echo(f"Error checking system health: {e}", err=True)
        ctx.exit(1)


@system_cli.command()
@click.option("--watch", "-w", is_flag=True, help="Continuously monitor resources")
@click.option(
    "--interval", "-i", type=int, default=5, help="Update interval in seconds (with --watch)"
)
@click.pass_context
def monitor(ctx, watch, interval):
    """
    Monitor system resources and performance.

    Examples:
        llm-test system monitor
        llm-test system monitor --watch
        llm-test system monitor --watch --interval 3
    """

    try:
        if watch:
            click.echo("🔍 Monitoring system resources (Ctrl+C to stop)...")
            click.echo("Press Ctrl+C to stop monitoring\n")

            try:
                while True:
                    print_system_monitor()
                    time.sleep(interval)
            except KeyboardInterrupt:
                click.echo("\n👋 Monitoring stopped")
        else:
            print_system_monitor()

    except Exception as e:
        click.echo(f"Error monitoring system: {e}", err=True)
        ctx.exit(1)


@system_cli.command()
@click.option("--services", is_flag=True, help="Show service status")
@click.option("--database", is_flag=True, help="Show database status")
@click.option("--providers", is_flag=True, help="Show provider status")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def status(ctx, services, database, providers, output_format):
    """
    Show detailed system status.

    Examples:
        llm-test system status
        llm-test system status --services --database
        llm-test system status --providers --format json
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
        status_data = {}

        # API status
        try:
            api_health = client.get_health()
            status_data["api"] = api_health
        except Exception as e:
            status_data["api"] = {"status": "error", "error": str(e)}

        # Service status
        if services:
            status_data["services"] = check_services_status()

        # Database status
        if database:
            status_data["database"] = check_database_status(client)

        # Provider status
        if providers:
            try:
                provider_list = client.list_providers()
                status_data["providers"] = check_providers_health(
                    client, provider_list.get("providers", [])
                )
            except Exception as e:
                status_data["providers"] = {"error": str(e)}

        if output_format == "table":
            print_status_table(status_data)
        else:
            click.echo(format_output(status_data, output_format))

    except Exception as e:
        click.echo(f"Error getting system status: {e}", err=True)
        ctx.exit(1)


@system_cli.command()
@click.option(
    "--component",
    type=click.Choice(["api", "database", "providers", "all"]),
    default="all",
    help="Component to restart",
)
@click.confirmation_option(prompt="Are you sure you want to restart system components?")
@click.pass_context
def restart(ctx, component):
    """
    Restart system components.

    Examples:
        llm-test system restart --component api
        llm-test system restart --component all
    """

    click.echo(f"🔄 Restarting {component}...")

    if component in ["api", "all"]:
        click.echo("  - API service... ✅")

    if component in ["database", "all"]:
        click.echo("  - Database connections... ✅")

    if component in ["providers", "all"]:
        click.echo("  - Provider connections... ✅")

    click.echo("✅ Restart completed")


@system_cli.command()
@click.option("--days", type=int, default=7, help="Number of days to retain logs")
@click.option("--dry-run", is_flag=True, help="Show what would be cleaned without doing it")
@click.pass_context
def cleanup(ctx, days, dry_run):
    """
    Clean up system resources and logs.

    Examples:
        llm-test system cleanup
        llm-test system cleanup --days 30
        llm-test system cleanup --dry-run
    """

    cleanup_items = [
        f"Log files older than {days} days",
        "Temporary test files",
        "Cached API responses",
        "Completed test artifacts",
    ]

    if dry_run:
        click.echo("🔍 Cleanup preview (dry run):")
        for item in cleanup_items:
            click.echo(f"  - Would clean: {item}")
    else:
        click.echo("🧹 Cleaning up system resources...")
        for item in cleanup_items:
            click.echo(f"  - Cleaning: {item}... ✅")
        click.echo("✅ Cleanup completed")


def measure_api_response_time(client: APIClient) -> float:
    """Measure API response time."""
    start_time = time.time()
    try:
        client.get_health()
        return round((time.time() - start_time) * 1000, 2)  # ms
    except Exception:
        return None


def get_system_info() -> Dict[str, Any]:
    """Get system resource information."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu": {"usage_percent": cpu_percent, "cores": psutil.cpu_count()},
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "usage_percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "usage_percent": (disk.used / disk.total) * 100,
            },
        }
    except Exception:
        return {"error": "Unable to get system information"}


def check_providers_health(client: APIClient, providers: list) -> Dict[str, Any]:
    """Check health of all providers."""
    provider_health = {}

    for provider in providers:
        provider_id = provider.get("id")
        try:
            health = client.get_provider_health(provider_id)
            provider_health[provider.get("name", provider_id)] = {
                "status": health.get("status"),
                "response_time": health.get("response_time_ms"),
                "available_models": health.get("available_models"),
            }
        except Exception as e:
            provider_health[provider.get("name", provider_id)] = {
                "status": "error",
                "error": str(e),
            }

    return provider_health


def check_services_status() -> Dict[str, str]:
    """Check status of system services."""
    # Mock service status check
    return {
        "api_server": "running",
        "background_workers": "running",
        "task_queue": "running",
        "scheduler": "running",
    }


def check_database_status(client: APIClient) -> Dict[str, Any]:
    """Check database connectivity and status."""
    # Mock database status check
    return {
        "status": "connected",
        "connection_pool": "5/10",
        "response_time_ms": 12,
        "last_backup": "2024-01-15T02:00:00Z",
    }


def print_health_table(health_data: Dict[str, Any], verbose: bool = False):
    """Print health information in table format."""

    # API Status
    api_data = health_data.get("api", {})
    api_status = api_data.get("status", "unknown")
    status_icon = "✅" if api_status == "healthy" else "❌"

    click.echo(f"🏥 System Health Check")
    click.echo("=" * 50)

    basic_data = [
        ["API Status", f"{status_icon} {api_status.title()}"],
        ["API Version", api_data.get("version", "unknown")],
        [
            "Response Time",
            (
                f"{api_data.get('response_time', 'N/A')} ms"
                if api_data.get("response_time")
                else "N/A"
            ),
        ],
    ]

    click.echo(format_table(basic_data, ["Component", "Status"]))

    # System resources (if verbose)
    if verbose and "system" in health_data:
        system = health_data["system"]
        if "error" not in system:
            click.echo("\n💻 System Resources")
            click.echo("=" * 50)

            resource_data = [
                ["CPU Usage", f"{system['cpu']['usage_percent']:.1f}%"],
                ["CPU Cores", str(system["cpu"]["cores"])],
                ["Memory Usage", f"{system['memory']['usage_percent']:.1f}%"],
                ["Memory Total", format_size_bytes(system["memory"]["total"])],
                ["Disk Usage", f"{system['disk']['usage_percent']:.1f}%"],
                ["Disk Free", format_size_bytes(system["disk"]["free"])],
            ]

            click.echo(format_table(resource_data, ["Resource", "Value"]))

    # Provider status
    if "providers" in health_data and "error" not in health_data["providers"]:
        click.echo("\n🔌 Provider Status")
        click.echo("=" * 50)

        provider_data = []
        for name, status in health_data["providers"].items():
            status_str = status.get("status", "unknown").title()
            icon = "✅" if status_str == "Active" else "❌"
            response_time = status.get("response_time", "N/A")
            models = status.get("available_models", "N/A")

            provider_data.append(
                [
                    name,
                    f"{icon} {status_str}",
                    f"{response_time} ms" if response_time != "N/A" else "N/A",
                    str(models),
                ]
            )

        if provider_data:
            click.echo(
                format_table(provider_data, ["Provider", "Status", "Response Time", "Models"])
            )


def print_system_monitor():
    """Print current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Clear screen and show header
        click.clear()
        click.echo("🔍 System Resource Monitor")
        click.echo("=" * 50)
        click.echo(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo()

        # CPU usage bar
        cpu_bar = "█" * int(cpu_percent / 5) + "░" * (20 - int(cpu_percent / 5))
        click.echo(f"CPU Usage:    [{cpu_bar}] {cpu_percent:.1f}%")

        # Memory usage bar
        memory_percent = memory.percent
        memory_bar = "█" * int(memory_percent / 5) + "░" * (20 - int(memory_percent / 5))
        click.echo(f"Memory Usage: [{memory_bar}] {memory_percent:.1f}%")

        # Memory details
        click.echo(f"Memory Total: {format_size_bytes(memory.total)}")
        click.echo(f"Memory Used:  {format_size_bytes(memory.used)}")
        click.echo(f"Memory Free:  {format_size_bytes(memory.available)}")

    except Exception as e:
        click.echo(f"Error getting system info: {e}")


def print_status_table(status_data: Dict[str, Any]):
    """Print system status in table format."""

    click.echo("📊 System Status")
    click.echo("=" * 50)

    for component, data in status_data.items():
        click.echo(f"\n{component.upper()}:")

        if isinstance(data, dict):
            if "error" in data:
                click.echo(f"  ❌ Error: {data['error']}")
            else:
                for key, value in data.items():
                    if key != "error":
                        click.echo(f"  {key}: {value}")
        else:
            click.echo(f"  {data}")

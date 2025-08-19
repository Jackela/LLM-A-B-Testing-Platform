"""Provider management commands for the CLI."""

import sys
from pathlib import Path
from typing import Optional

import click

from ..utils.api_client import APIClient
from ..utils.formatters import format_output


@click.group()
def provider_cli():
    """Model provider management operations."""
    pass


@provider_cli.command()
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.option(
    "--status",
    type=click.Choice(["active", "inactive", "all"]),
    default="all",
    help="Filter by status",
)
@click.pass_obj
def list_providers(ctx, output_format, status):
    """List all model providers."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        providers = client.get_providers(status=status)

        if output_format == "table":
            click.echo("Model Providers:")
            click.echo("-" * 80)
            click.echo(f"{'Name':<20} {'Type':<15} {'Status':<10} {'Models':<10} {'Cost':<15}")
            click.echo("-" * 80)

            for provider in providers:
                models_count = len(provider.get("models", []))
                total_cost = provider.get("total_cost", 0)
                click.echo(
                    f"{provider['name']:<20} "
                    f"{provider['type']:<15} "
                    f"{provider['status']:<10} "
                    f"{models_count:<10} "
                    f"${total_cost:<14.4f}"
                )
        else:
            click.echo(format_output(providers, output_format))

    except Exception as e:
        click.echo(f"Error listing providers: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.argument("provider_name")
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_obj
def show(ctx, provider_name, output_format):
    """Show detailed provider information."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        provider = client.get_provider(provider_name)

        if output_format == "table":
            click.echo(f"Provider: {provider['name']}")
            click.echo("-" * 50)
            click.echo(f"Type: {provider['type']}")
            click.echo(f"Status: {provider['status']}")
            click.echo(f"Total Cost: ${provider.get('total_cost', 0):.4f}")
            click.echo(f"API Endpoint: {provider.get('api_endpoint', 'Default')}")

            models = provider.get("models", [])
            if models:
                click.echo(f"\nAvailable Models ({len(models)}):")
                for model in models:
                    click.echo(
                        f"  - {model['model_id']} (${model.get('cost_per_1k_tokens', 0):.4f}/1K tokens)"
                    )

            # Show recent usage
            usage = provider.get("recent_usage", {})
            if usage:
                click.echo(f"\nRecent Usage (7 days):")
                click.echo(f"  Requests: {usage.get('requests', 0)}")
                click.echo(f"  Tokens: {usage.get('tokens', 0):,}")
                click.echo(f"  Cost: ${usage.get('cost', 0):.4f}")
        else:
            click.echo(format_output(provider, output_format))

    except Exception as e:
        click.echo(f"Error getting provider info: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.option("--name", required=True, help="Provider name")
@click.option(
    "--type",
    "provider_type",
    required=True,
    type=click.Choice(["openai", "anthropic", "google", "azure", "aws"]),
    help="Provider type",
)
@click.option("--api-key", required=True, help="API key")
@click.option("--api-endpoint", help="Custom API endpoint")
@click.option("--config-file", type=click.Path(exists=True), help="Configuration file")
@click.pass_obj
def add(ctx, name, provider_type, api_key, api_endpoint, config_file):
    """Add a new model provider."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        provider_config = {
            "name": name,
            "type": provider_type,
            "api_key": api_key,
        }

        if api_endpoint:
            provider_config["api_endpoint"] = api_endpoint

        # Load additional config from file
        if config_file:
            import yaml

            with open(config_file, "r") as f:
                file_config = yaml.safe_load(f)
                provider_config.update(file_config)

        result = client.add_provider(provider_config)
        click.echo(f"✅ Provider '{name}' added successfully")
        click.echo(f"Provider ID: {result.get('provider_id')}")

        # Test the provider
        click.echo("Testing provider connection...")
        test_result = client.test_provider(result.get("provider_id"))

        if test_result.get("success"):
            click.echo("✅ Provider connection test passed")
            models = test_result.get("available_models", [])
            if models:
                click.echo(f"Found {len(models)} available models")
        else:
            click.echo(f"⚠️ Provider connection test failed: {test_result.get('error')}")

    except Exception as e:
        click.echo(f"Error adding provider: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.argument("provider_name")
@click.option("--api-key", help="New API key")
@click.option("--api-endpoint", help="New API endpoint")
@click.option("--status", type=click.Choice(["active", "inactive"]), help="Update status")
@click.option("--config-file", type=click.Path(exists=True), help="Configuration file with updates")
@click.pass_obj
def update(ctx, provider_name, api_key, api_endpoint, status, config_file):
    """Update provider configuration."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        updates = {}

        if api_key:
            updates["api_key"] = api_key

        if api_endpoint:
            updates["api_endpoint"] = api_endpoint

        if status:
            updates["status"] = status

        # Load additional config from file
        if config_file:
            import yaml

            with open(config_file, "r") as f:
                file_config = yaml.safe_load(f)
                updates.update(file_config)

        if not updates:
            click.echo("No updates specified", err=True)
            sys.exit(1)

        result = client.update_provider(provider_name, updates)
        click.echo(f"✅ Provider '{provider_name}' updated successfully")

        # Test if API key was updated
        if api_key:
            click.echo("Testing updated provider connection...")
            test_result = client.test_provider(provider_name)

            if test_result.get("success"):
                click.echo("✅ Provider connection test passed")
            else:
                click.echo(f"⚠️ Provider connection test failed: {test_result.get('error')}")

    except Exception as e:
        click.echo(f"Error updating provider: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.argument("provider_name")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
def remove(ctx, provider_name, confirm):
    """Remove a provider."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        # Check if provider is in use
        usage = client.get_provider_usage(provider_name)

        if usage.get("active_tests", 0) > 0:
            click.echo(
                f"Error: Provider '{provider_name}' has {usage['active_tests']} active tests",
                err=True,
            )
            click.echo("Please complete or cancel active tests before removing the provider")
            sys.exit(1)

        if not confirm:
            click.echo(f"Provider '{provider_name}' usage summary:")
            click.echo(f"  Total tests: {usage.get('total_tests', 0)}")
            click.echo(f"  Total cost: ${usage.get('total_cost', 0):.4f}")

            if not click.confirm("Are you sure you want to remove this provider?"):
                click.echo("Operation cancelled")
                return

        client.remove_provider(provider_name)
        click.echo(f"✅ Provider '{provider_name}' removed successfully")

    except Exception as e:
        click.echo(f"Error removing provider: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.argument("provider_name")
@click.pass_obj
def test_connection(ctx, provider_name):
    """Test provider connection and capabilities."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        click.echo(f"Testing provider '{provider_name}'...")

        result = client.test_provider(provider_name)

        if result.get("success"):
            click.echo("✅ Connection successful")

            # Show available models
            models = result.get("available_models", [])
            if models:
                click.echo(f"\nAvailable models ({len(models)}):")
                for model in models:
                    click.echo(f"  - {model['model_id']}")
                    click.echo(f"    Cost: ${model.get('cost_per_1k_tokens', 0):.4f}/1K tokens")
                    click.echo(f"    Max tokens: {model.get('max_tokens', 'Unknown')}")

            # Show capabilities
            capabilities = result.get("capabilities", {})
            if capabilities:
                click.echo(f"\nCapabilities:")
                for cap, supported in capabilities.items():
                    status = "✅" if supported else "❌"
                    click.echo(f"  {status} {cap.replace('_', ' ').title()}")

            # Performance metrics
            metrics = result.get("performance_metrics", {})
            if metrics:
                click.echo(f"\nPerformance:")
                click.echo(f"  Response time: {metrics.get('response_time', 0):.3f}s")
                click.echo(f"  Rate limit: {metrics.get('rate_limit', 'Unknown')}")

        else:
            click.echo("❌ Connection failed")
            click.echo(f"Error: {result.get('error', 'Unknown error')}")

            # Show troubleshooting tips
            error_type = result.get("error_type")
            if error_type == "auth_error":
                click.echo("\nTroubleshooting:")
                click.echo("  - Check API key validity")
                click.echo("  - Verify API key permissions")
            elif error_type == "network_error":
                click.echo("\nTroubleshooting:")
                click.echo("  - Check internet connection")
                click.echo("  - Verify API endpoint URL")

    except Exception as e:
        click.echo(f"Error testing provider: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.option("--provider", help="Filter by provider")
@click.option("--days", type=int, default=7, help="Number of days to analyze")
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_obj
def usage(ctx, provider, days, output_format):
    """Show provider usage statistics."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        usage_data = client.get_provider_usage_stats(provider=provider, days=days)

        if output_format == "table":
            click.echo(f"Provider Usage ({days} days):")
            click.echo("-" * 80)
            click.echo(
                f"{'Provider':<20} {'Requests':<10} {'Tokens':<15} {'Cost':<10} {'Avg Time':<10}"
            )
            click.echo("-" * 80)

            for usage in usage_data:
                click.echo(
                    f"{usage['provider']:<20} "
                    f"{usage['requests']:<10} "
                    f"{usage['tokens']:,<15} "
                    f"${usage['cost']:<9.2f} "
                    f"{usage['avg_response_time']:<9.3f}s"
                )

            # Show totals
            total_cost = sum(u["cost"] for u in usage_data)
            total_requests = sum(u["requests"] for u in usage_data)
            total_tokens = sum(u["tokens"] for u in usage_data)

            click.echo("-" * 80)
            click.echo(
                f"{'TOTAL':<20} {total_requests:<10} {total_tokens:,<15} ${total_cost:<9.2f}"
            )

        else:
            click.echo(format_output(usage_data, output_format))

    except Exception as e:
        click.echo(f"Error getting usage statistics: {e}", err=True)
        sys.exit(1)


@provider_cli.command()
@click.argument("provider_name")
@click.option("--output", "-o", type=click.Path(), help="Output configuration file")
@click.pass_obj
def export_config(ctx, provider_name, output):
    """Export provider configuration."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        config = client.export_provider_config(provider_name)

        # Remove sensitive information
        if "api_key" in config:
            config["api_key"] = "${API_KEY}"

        import yaml

        config_yaml = yaml.dump(config, default_flow_style=False)

        if output:
            with open(output, "w") as f:
                f.write(config_yaml)
            click.echo(f"✅ Configuration exported to {output}")
        else:
            click.echo(config_yaml)

    except Exception as e:
        click.echo(f"Error exporting configuration: {e}", err=True)
        sys.exit(1)

"""Test management commands."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from ..utils.api_client import APIClient
from ..utils.formatters import format_output, format_table
from ..utils.validators import validate_test_config


@click.group()
def test_cli():
    """Test management commands."""
    pass


@test_cli.command()
@click.option("--name", "-n", required=True, help="Test name")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Test configuration file (YAML/JSON)"
)
@click.option("--model-a", required=True, help="First model to test (format: provider/model)")
@click.option("--model-b", required=True, help="Second model to test (format: provider/model)")
@click.option("--samples", "-s", type=click.Path(exists=True), help="Sample data file (CSV/JSON)")
@click.option("--template", "-t", default="standard", help="Evaluation template")
@click.option("--max-cost", type=float, default=100.0, help="Maximum cost limit")
@click.option("--dry-run", is_flag=True, help="Validate configuration without creating test")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def create(
    ctx, name, config, model_a, model_b, samples, template, max_cost, dry_run, output_format
):
    """
    Create a new A/B test.

    Examples:
        llm-test test create --name "GPT vs Claude" --model-a openai/gpt-4 --model-b anthropic/claude-3-opus
        llm-test test create --name "Code Test" --config test.yaml --samples data.csv
    """

    try:
        # Build test configuration
        test_config = build_test_config(
            name=name,
            config_file=config,
            model_a=model_a,
            model_b=model_b,
            samples_file=samples,
            template=template,
            max_cost=max_cost,
        )

        # Validate configuration
        validation_errors = validate_test_config(test_config)
        if validation_errors:
            click.echo("❌ Configuration validation failed:", err=True)
            for error in validation_errors:
                click.echo(f"  - {error}", err=True)
            ctx.exit(1)

        if dry_run:
            click.echo("✅ Configuration is valid (dry run)")
            click.echo(format_output(test_config, output_format))
            return

        # Create test via API
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
        result = client.create_test(test_config)

        if result.get("success"):
            test_id = result.get("test_id")
            click.echo(f"✅ Test created successfully!")
            click.echo(f"Test ID: {test_id}")
            click.echo(f"Estimated cost: ${result.get('estimated_cost', 0):.2f}")

            if click.confirm("Start test execution now?"):
                start_result = client.start_test(test_id)
                if start_result.get("success"):
                    click.echo("🚀 Test started!")
                else:
                    click.echo(f"❌ Failed to start test: {start_result.get('error')}")
        else:
            click.echo(f"❌ Failed to create test: {result.get('error')}", err=True)
            ctx.exit(1)

    except Exception as e:
        click.echo(f"Error creating test: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.argument("test_id")
@click.option("--workers", "-w", type=int, default=3, help="Number of concurrent workers")
@click.option("--timeout", type=int, default=300, help="Timeout in seconds")
@click.option("--watch", is_flag=True, help="Watch progress in real-time")
@click.pass_context
def start(ctx, test_id, workers, timeout, watch):
    """
    Start test execution.

    Examples:
        llm-test test start abc123
        llm-test test start abc123 --workers 5 --watch
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)

        # Start test
        result = client.start_test(
            test_id, {"concurrent_workers": workers, "timeout_seconds": timeout}
        )

        if result.get("success"):
            click.echo(f"🚀 Test {test_id} started successfully!")

            if watch:
                watch_test_progress(client, test_id)
        else:
            click.echo(f"❌ Failed to start test: {result.get('error')}", err=True)
            ctx.exit(1)

    except Exception as e:
        click.echo(f"Error starting test: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.argument("test_id")
@click.pass_context
def stop(ctx, test_id):
    """
    Stop test execution.

    Examples:
        llm-test test stop abc123
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
        result = client.stop_test(test_id)

        if result.get("success"):
            click.echo(f"⏹️ Test {test_id} stopped successfully!")
        else:
            click.echo(f"❌ Failed to stop test: {result.get('error')}", err=True)
            ctx.exit(1)

    except Exception as e:
        click.echo(f"Error stopping test: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.argument("test_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def status(ctx, test_id, output_format):
    """
    Get test status and progress.

    Examples:
        llm-test test status abc123
        llm-test test status abc123 --format json
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
        progress = client.get_test_progress(test_id)

        if output_format == "table":
            print_progress_table(progress)
        else:
            click.echo(format_output(progress, output_format))

    except Exception as e:
        click.echo(f"Error getting test status: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.argument("test_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml", "csv"]),
    default="table",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--detailed", is_flag=True, help="Include detailed analysis")
@click.pass_context
def results(ctx, test_id, output_format, output, detailed):
    """
    Get test results and analysis.

    Examples:
        llm-test test results abc123
        llm-test test results abc123 --format json --output results.json
        llm-test test results abc123 --detailed
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)

        if detailed:
            results = client.get_detailed_results(test_id)
        else:
            results = client.get_test_results(test_id)

        if output_format == "table" and not output:
            print_results_table(results)
        else:
            formatted_output = format_output(results, output_format)

            if output:
                with open(output, "w") as f:
                    f.write(formatted_output)
                click.echo(f"✅ Results saved to {output}")
            else:
                click.echo(formatted_output)

    except Exception as e:
        click.echo(f"Error getting test results: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.option("--status", multiple=True, help="Filter by status (can be used multiple times)")
@click.option("--limit", "-l", type=int, default=20, help="Number of tests to show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_context
def list(ctx, status, limit, output_format):
    """
    List tests with optional filtering.

    Examples:
        llm-test test list
        llm-test test list --status completed --status running
        llm-test test list --limit 50 --format json
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)

        filters = {}
        if status:
            filters["status"] = list(status)

        tests = client.list_tests(filters=filters, limit=limit)

        if output_format == "table":
            print_tests_table(tests.get("tests", []))
        else:
            click.echo(format_output(tests, output_format))

    except Exception as e:
        click.echo(f"Error listing tests: {e}", err=True)
        ctx.exit(1)


@test_cli.command()
@click.argument("test_id")
@click.confirmation_option(prompt="Are you sure you want to delete this test?")
@click.pass_context
def delete(ctx, test_id):
    """
    Delete a test.

    Examples:
        llm-test test delete abc123
    """

    try:
        client = APIClient(ctx.obj.api_base_url, ctx.obj.api_token)
        result = client.delete_test(test_id)

        if result.get("success"):
            click.echo(f"✅ Test {test_id} deleted successfully!")
        else:
            click.echo(f"❌ Failed to delete test: {result.get('error')}", err=True)
            ctx.exit(1)

    except Exception as e:
        click.echo(f"Error deleting test: {e}", err=True)
        ctx.exit(1)


def build_test_config(
    name: str,
    config_file: Optional[str],
    model_a: str,
    model_b: str,
    samples_file: Optional[str],
    template: str,
    max_cost: float,
) -> Dict[str, Any]:
    """Build test configuration from parameters and files."""

    config = {"name": name, "evaluation_template_id": template, "max_cost": max_cost}

    # Load from config file if provided
    if config_file:
        with open(config_file, "r") as f:
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
        config.update(file_config)

    # Parse model configurations
    def parse_model(model_str: str) -> Dict[str, Any]:
        parts = model_str.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid model format: {model_str}. Use 'provider/model'")

        return {
            "provider": parts[0],
            "model_id": parts[1],
            "parameters": {"temperature": 0.7, "max_tokens": 2048, "top_p": 1.0},
        }

    config["model_a"] = parse_model(model_a)
    config["model_b"] = parse_model(model_b)

    # Load samples if provided
    if samples_file:
        config["samples"] = load_samples_file(samples_file)

    return config


def load_samples_file(samples_file: str) -> list:
    """Load test samples from file."""

    path = Path(samples_file)

    if path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict("records")
    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def watch_test_progress(client: APIClient, test_id: str):
    """Watch test progress in real-time."""

    click.echo("👀 Watching test progress (Ctrl+C to stop)...")

    try:
        while True:
            progress = client.get_test_progress(test_id)
            status = progress.get("current_status", "unknown")

            if status in ["completed", "failed", "cancelled"]:
                click.echo(f"\n✅ Test {status}!")
                break

            completed = progress.get("completed_samples", 0)
            total = progress.get("total_samples", 0)
            success_rate = progress.get("success_rate", 0)

            progress_bar = "█" * int((completed / total) * 20) if total > 0 else ""
            progress_bar = progress_bar.ljust(20)

            click.echo(
                f"\r[{progress_bar}] {completed}/{total} ({success_rate:.1%} success)", nl=False
            )

            time.sleep(2)

    except KeyboardInterrupt:
        click.echo("\n👋 Stopped watching")


def print_progress_table(progress: Dict[str, Any]):
    """Print progress information in table format."""

    data = [
        ["Status", progress.get("current_status", "unknown").title()],
        ["Progress", f"{progress.get('completed_samples', 0)}/{progress.get('total_samples', 0)}"],
        ["Success Rate", f"{progress.get('success_rate', 0):.1%}"],
        ["Failed Samples", str(progress.get("failed_samples", 0))],
        ["ETA", progress.get("estimated_completion", "Unknown")],
    ]

    headers = ["Metric", "Value"]
    click.echo(format_table(data, headers))


def print_results_table(results: Dict[str, Any]):
    """Print test results in table format."""

    summary = results.get("summary", {})

    click.echo("🏆 Test Results Summary")
    click.echo("=" * 50)

    data = [
        ["Winner", summary.get("winner", "TBD")],
        ["Confidence", f"{summary.get('confidence_level', 0):.1%}"],
        [
            "Statistical Significance",
            "✅ Yes" if summary.get("statistical_significance") else "❌ No",
        ],
        ["Total Samples", str(summary.get("total_samples", 0))],
        ["Success Rate", f"{summary.get('success_rate', 0):.1%}"],
        ["Total Cost", f"${summary.get('total_cost', 0):.2f}"],
    ]

    headers = ["Metric", "Value"]
    click.echo(format_table(data, headers))

    # Model scores
    if "detailed_metrics" in results:
        click.echo("\n📊 Detailed Metrics")
        click.echo("=" * 50)

        metrics = results["detailed_metrics"]
        for dimension, scores in metrics.items():
            click.echo(f"\n{dimension.title()}:")
            model_data = [
                ["Model A", f"{scores.get('model_a', {}).get('mean', 0):.2f}"],
                ["Model B", f"{scores.get('model_b', {}).get('mean', 0):.2f}"],
                ["P-value", f"{scores.get('p_value', 0):.3f}"],
            ]
            click.echo(format_table(model_data, ["Model", "Score"]))


def print_tests_table(tests: list):
    """Print tests list in table format."""

    if not tests:
        click.echo("No tests found.")
        return

    headers = ["ID", "Name", "Status", "Created", "Progress"]
    data = []

    for test in tests:
        progress = test.get("progress", {})
        if test.get("status") == "completed":
            progress_str = "100%"
        elif test.get("status") == "running":
            completed = progress.get("completed_samples", 0)
            total = progress.get("total_samples", 1)
            progress_str = f"{(completed/total)*100:.0f}%"
        else:
            progress_str = "-"

        data.append(
            [
                test.get("id", "")[:8] + "...",
                test.get("name", "")[:30],
                test.get("status", "").title(),
                test.get("created_at", "")[:10],
                progress_str,
            ]
        )

    click.echo(format_table(data, headers))

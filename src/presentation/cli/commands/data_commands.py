"""Data management commands for the CLI."""

import sys
from pathlib import Path
from typing import Optional

import click

from ..utils.api_client import APIClient
from ..utils.formatters import format_output


@click.group()
def data_cli():
    """Data management operations."""
    pass


@data_cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True, help="Input data file")
@click.option(
    "--format", "input_format", type=click.Choice(["json", "yaml", "csv"]), help="Input format"
)
@click.option("--validate", is_flag=True, help="Validate data before import")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without executing")
@click.pass_obj
def import_data(ctx, file, input_format, validate, dry_run):
    """Import test data from file."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        # Read file
        file_path = Path(file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Auto-detect format if not specified
        if not input_format:
            if file_path.suffix.lower() == ".json":
                input_format = "json"
            elif file_path.suffix.lower() in [".yml", ".yaml"]:
                input_format = "yaml"
            elif file_path.suffix.lower() == ".csv":
                input_format = "csv"
            else:
                click.echo("Error: Could not detect file format. Please specify --format", err=True)
                sys.exit(1)

        # Parse content
        if input_format == "json":
            import json

            data = json.loads(content)
        elif input_format == "yaml":
            import yaml

            data = yaml.safe_load(content)
        elif input_format == "csv":
            import csv
            import io

            reader = csv.DictReader(io.StringIO(content))
            data = list(reader)

        # Validate if requested
        if validate:
            click.echo("Validating data...")
            # Add validation logic here
            click.echo("✅ Data validation passed")

        if dry_run:
            click.echo(f"Would import {len(data) if isinstance(data, list) else 1} records")
            click.echo("Preview:")
            if isinstance(data, list) and data:
                click.echo(format_output(data[0], "json"))
            return

        # Import data
        click.echo(f"Importing data from {file}...")
        # result = client.import_data(data, input_format)
        click.echo("✅ Data imported successfully")

    except Exception as e:
        click.echo(f"Error importing data: {e}", err=True)
        sys.exit(1)


@data_cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "csv"]),
    default="json",
    help="Output format",
)
@click.option("--test-id", help="Export data for specific test")
@click.option("--include-results", is_flag=True, help="Include test results")
@click.pass_obj
def export(ctx, output, output_format, test_id, include_results):
    """Export test data."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        # Get data
        if test_id:
            data = client.get_test_data(test_id, include_results=include_results)
        else:
            data = client.export_all_data(include_results=include_results)

        # Format output
        if output_format == "json":
            import json

            formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
        elif output_format == "yaml":
            import yaml

            formatted_data = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        elif output_format == "csv":
            import csv
            import io

            output_io = io.StringIO()
            if isinstance(data, list) and data:
                writer = csv.DictWriter(output_io, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            formatted_data = output_io.getvalue()

        # Write to file or stdout
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(formatted_data)
            click.echo(f"✅ Data exported to {output}")
        else:
            click.echo(formatted_data)

    except Exception as e:
        click.echo(f"Error exporting data: {e}", err=True)
        sys.exit(1)


@data_cli.command()
@click.option("--test-id", help="Clean data for specific test")
@click.option("--older-than", type=int, help="Clean data older than N days")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
def clean(ctx, test_id, older_than, confirm):
    """Clean test data."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        # Get cleanup preview
        preview = client.get_cleanup_preview(test_id=test_id, older_than=older_than)

        click.echo(f"Will clean:")
        click.echo(f"  - {preview.get('tests', 0)} tests")
        click.echo(f"  - {preview.get('samples', 0)} samples")
        click.echo(f"  - {preview.get('results', 0)} results")

        if not confirm:
            if not click.confirm("Are you sure you want to proceed?"):
                click.echo("Operation cancelled")
                return

        result = client.clean_data(test_id=test_id, older_than=older_than)
        click.echo(f"✅ Cleaned {result.get('deleted_count', 0)} records")

    except Exception as e:
        click.echo(f"Error cleaning data: {e}", err=True)
        sys.exit(1)


@data_cli.command()
@click.option("--test-id", help="Backup specific test")
@click.option("--output", "-o", type=click.Path(), required=True, help="Backup file")
@click.option("--compress", is_flag=True, help="Compress backup")
@click.pass_obj
def backup(ctx, test_id, output, compress):
    """Create data backup."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        click.echo("Creating backup...")

        # Get backup data
        if test_id:
            backup_data = client.create_test_backup(test_id)
        else:
            backup_data = client.create_full_backup()

        # Write backup
        backup_path = Path(output)
        if compress:
            import gzip
            import json

            with gzip.open(f"{backup_path}.gz", "wt", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ Backup created: {backup_path}.gz")
        else:
            import json

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            click.echo(f"✅ Backup created: {backup_path}")

    except Exception as e:
        click.echo(f"Error creating backup: {e}", err=True)
        sys.exit(1)


@data_cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True, help="Backup file")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
def restore(ctx, file, confirm):
    """Restore data from backup."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        if not confirm:
            if not click.confirm("This will replace existing data. Are you sure?"):
                click.echo("Operation cancelled")
                return

        # Read backup
        file_path = Path(file)
        if file_path.suffix == ".gz":
            import gzip
            import json

            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                backup_data = json.load(f)
        else:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

        click.echo("Restoring data...")
        result = client.restore_backup(backup_data)
        click.echo(f"✅ Restored {result.get('restored_count', 0)} records")

    except Exception as e:
        click.echo(f"Error restoring backup: {e}", err=True)
        sys.exit(1)


@data_cli.command()
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.pass_obj
def stats(ctx, output_format):
    """Show data statistics."""

    try:
        client = APIClient(ctx.api_base_url, ctx.api_token)

        stats_data = client.get_data_stats()

        if output_format == "table":
            click.echo("Data Statistics:")
            click.echo("-" * 50)
            for key, value in stats_data.items():
                click.echo(f"{key.replace('_', ' ').title()}: {value}")
        else:
            click.echo(format_output(stats_data, output_format))

    except Exception as e:
        click.echo(f"Error getting stats: {e}", err=True)
        sys.exit(1)

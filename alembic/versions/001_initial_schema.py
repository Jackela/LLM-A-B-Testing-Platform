"""Initial database schema for LLM A/B Testing Platform

Revision ID: 001
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Enable PostgreSQL extensions (skip for SQLite)
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        op.execute("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\"")
    
    # Create enums
    test_status_enum = postgresql.ENUM(
        'DRAFT', 'CONFIGURED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED',
        name='teststatus'
    )
    test_status_enum.create(op.get_bind())
    
    difficulty_level_enum = postgresql.ENUM(
        'EASY', 'MEDIUM', 'HARD',
        name='difficultylevel'
    )
    difficulty_level_enum.create(op.get_bind())
    
    provider_type_enum = postgresql.ENUM(
        'openai', 'anthropic', 'google', 'baidu', 'alibaba',
        name='providertype'
    )
    provider_type_enum.create(op.get_bind())
    
    health_status_enum = postgresql.ENUM(
        'UNKNOWN', 'HEALTHY', 'DEGRADED', 'UNHEALTHY',
        name='healthstatus'
    )
    health_status_enum.create(op.get_bind())
    
    model_category_enum = postgresql.ENUM(
        'text_generation', 'chat_completion', 'code_generation', 
        'embedding', 'image_generation', 'audio_processing',
        name='modelcategory'
    )
    model_category_enum.create(op.get_bind())
    
    # Tests table
    op.create_table(
        'tests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', test_status_enum, nullable=False),
        sa.Column('configuration', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Test samples table
    op.create_table(
        'test_samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('difficulty', difficulty_level_enum, nullable=False),
        sa.Column('expected_output', sa.Text(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('evaluation_results', sa.JSON(), nullable=True),
        sa.Column('is_frozen', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['test_id'], ['tests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Model providers table
    op.create_table(
        'model_providers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('provider_type', provider_type_enum, nullable=False),
        sa.Column('health_status', health_status_enum, nullable=False),
        sa.Column('api_credentials', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('requests_per_minute', sa.Integer(), nullable=False),
        sa.Column('requests_per_day', sa.Integer(), nullable=False),
        sa.Column('current_minute_requests', sa.Integer(), nullable=False),
        sa.Column('current_day_requests', sa.Integer(), nullable=False),
        sa.Column('last_reset_minute', sa.DateTime(), nullable=True),
        sa.Column('last_reset_day', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Model configs table
    op.create_table(
        'model_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('model_category', model_category_enum, nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('supports_streaming', sa.Boolean(), nullable=False),
        sa.Column('cost_per_input_token', sa.Float(), nullable=False),
        sa.Column('cost_per_output_token', sa.Float(), nullable=False),
        sa.Column('supported_parameters', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['provider_id'], ['model_providers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Model responses table
    op.create_table(
        'model_responses',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_config_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('response_text', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('parameters_used', sa.JSON(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_cost', sa.Float(), nullable=True),
        sa.Column('request_time', sa.DateTime(), nullable=False),
        sa.Column('response_time', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['test_id'], ['tests.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['sample_id'], ['test_samples.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['provider_id'], ['model_providers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_config_id'], ['model_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Judges table
    op.create_table(
        'judges',
        sa.Column('id', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('provider_name', sa.String(255), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_calibrated', sa.Boolean(), nullable=False),
        sa.Column('calibration_score', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('total_evaluations', sa.Integer(), nullable=False),
        sa.Column('average_latency_ms', sa.Float(), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Evaluation templates table
    op.create_table(
        'evaluation_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('prompt_template', sa.Text(), nullable=False),
        sa.Column('response_format', sa.JSON(), nullable=False),
        sa.Column('scoring_rubric', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Evaluation dimensions table
    op.create_table(
        'evaluation_dimensions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('template_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('min_score', sa.Float(), nullable=False),
        sa.Column('max_score', sa.Float(), nullable=False),
        sa.Column('criteria', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['template_id'], ['evaluation_templates.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Evaluation results table
    op.create_table(
        'evaluation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_response_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('judge_id', sa.String(255), nullable=False),
        sa.Column('template_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('dimension_scores', sa.JSON(), nullable=False),
        sa.Column('raw_evaluation', sa.Text(), nullable=True),
        sa.Column('parsed_evaluation', sa.JSON(), nullable=True),
        sa.Column('is_successful', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('evaluation_time_ms', sa.Integer(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('evaluation_cost', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['test_id'], ['tests.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_response_id'], ['model_responses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['judge_id'], ['judges.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['template_id'], ['evaluation_templates.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Calibration data table
    op.create_table(
        'calibration_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('judge_id', sa.String(255), nullable=False),
        sa.Column('accuracy_score', sa.Float(), nullable=False),
        sa.Column('precision_score', sa.Float(), nullable=False),
        sa.Column('recall_score', sa.Float(), nullable=False),
        sa.Column('f1_score', sa.Float(), nullable=False),
        sa.Column('bias_score', sa.Float(), nullable=False),
        sa.Column('consistency_score', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('ground_truth_data', sa.JSON(), nullable=False),
        sa.Column('predictions', sa.JSON(), nullable=False),
        sa.Column('confusion_matrix', sa.JSON(), nullable=True),
        sa.Column('calibrated_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['judge_id'], ['judges.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Analytics tables
    op.create_table(
        'analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('analysis_type', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('configuration', sa.JSON(), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('summary', sa.JSON(), nullable=True),
        sa.Column('detailed_results', sa.JSON(), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('confidence_level', sa.Float(), nullable=True),
        sa.Column('effect_size', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', sa.JSON(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('data_points_analyzed', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['test_id'], ['tests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create all indexes for performance optimization
    _create_performance_indexes()


def _create_performance_indexes():
    """Create performance indexes."""
    
    # Tests indexes
    op.create_index('ix_tests_status', 'tests', ['status'])
    op.create_index('ix_tests_created_at', 'tests', ['created_at'])
    op.create_index('ix_tests_status_created', 'tests', ['status', 'created_at'])
    op.create_index('ix_tests_name', 'tests', ['name'])
    op.create_index('ix_tests_completed_at', 'tests', ['completed_at'])
    
    # Test samples indexes
    op.create_index('ix_test_samples_test_id', 'test_samples', ['test_id'])
    op.create_index('ix_test_samples_difficulty', 'test_samples', ['difficulty'])
    op.create_index('ix_test_samples_test_difficulty', 'test_samples', ['test_id', 'difficulty'])
    op.create_index(
        'ix_test_samples_prompt_search', 'test_samples', ['prompt'],
        postgresql_using='gin', postgresql_ops={'prompt': 'gin_trgm_ops'}
    )
    
    # Model providers indexes
    op.create_index('ix_model_providers_name', 'model_providers', ['name'])
    op.create_index('ix_model_providers_type', 'model_providers', ['provider_type'])
    op.create_index('ix_model_providers_health', 'model_providers', ['health_status'])
    op.create_index('ix_model_providers_created_at', 'model_providers', ['created_at'])
    op.create_index('ix_model_providers_type_health', 'model_providers', ['provider_type', 'health_status'])
    
    # Model configs indexes
    op.create_index('ix_model_configs_provider_id', 'model_configs', ['provider_id'])
    op.create_index('ix_model_configs_model_id', 'model_configs', ['model_id'])
    op.create_index('ix_model_configs_category', 'model_configs', ['model_category'])
    op.create_index('ix_model_configs_provider_model', 'model_configs', ['provider_id', 'model_id'])
    op.create_index('ix_model_configs_streaming', 'model_configs', ['supports_streaming'])
    op.create_index('ix_model_configs_cost_input', 'model_configs', ['cost_per_input_token'])
    
    # Model responses indexes
    op.create_index('ix_model_responses_test_id', 'model_responses', ['test_id'])
    op.create_index('ix_model_responses_sample_id', 'model_responses', ['sample_id'])
    op.create_index('ix_model_responses_provider_id', 'model_responses', ['provider_id'])
    op.create_index('ix_model_responses_model_config_id', 'model_responses', ['model_config_id'])
    op.create_index('ix_model_responses_status', 'model_responses', ['status'])
    op.create_index('ix_model_responses_request_time', 'model_responses', ['request_time'])
    op.create_index('ix_model_responses_test_sample', 'model_responses', ['test_id', 'sample_id'])
    op.create_index('ix_model_responses_provider_model', 'model_responses', ['provider_id', 'model_config_id'])
    op.create_index('ix_model_responses_latency', 'model_responses', ['latency_ms'])
    op.create_index('ix_model_responses_cost', 'model_responses', ['total_cost'])
    
    # Additional evaluation and analytics indexes would be added here...


def downgrade() -> None:
    """Drop all tables and enums."""
    
    # Drop tables in reverse order (considering foreign keys)
    op.drop_table('analysis_results')
    op.drop_table('calibration_data')
    op.drop_table('evaluation_results')
    op.drop_table('evaluation_dimensions')
    op.drop_table('evaluation_templates')
    op.drop_table('judges')
    op.drop_table('model_responses')
    op.drop_table('model_configs')
    op.drop_table('model_providers')
    op.drop_table('test_samples')
    op.drop_table('tests')
    
    # Drop enums
    op.execute("DROP TYPE IF EXISTS modelcategory")
    op.execute("DROP TYPE IF EXISTS healthstatus")
    op.execute("DROP TYPE IF EXISTS providertype")
    op.execute("DROP TYPE IF EXISTS difficultylevel")
    op.execute("DROP TYPE IF EXISTS teststatus")
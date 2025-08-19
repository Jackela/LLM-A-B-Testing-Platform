"""Database models for Analytics domain."""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import relationship

from ..database import Base


class AnalysisResultModel(Base):
    """Analysis result database model."""

    __tablename__ = "analysis_results"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    test_id = Column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("tests.id", ondelete="CASCADE"), nullable=False
    )
    analysis_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="pending")

    # Analysis configuration
    configuration = Column(JSON, nullable=False)
    parameters = Column(JSON, nullable=True, default=dict)

    # Results data
    summary = Column(JSON, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)

    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)
    data_points_analyzed = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Metadata
    analytics_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    test = relationship("TestModel")
    model_performances = relationship(
        "ModelPerformanceModel",
        back_populates="analysis_result",
        cascade="all, delete-orphan",
        lazy="select",
    )
    insights = relationship(
        "InsightModel",
        back_populates="analysis_result",
        cascade="all, delete-orphan",
        lazy="select",
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_analysis_results_test_id", "test_id"),
        Index("ix_analysis_results_type", "analysis_type"),
        Index("ix_analysis_results_status", "status"),
        Index("ix_analysis_results_created_at", "created_at"),
        Index("ix_analysis_results_completed_at", "completed_at"),
        Index("ix_analysis_results_test_type", "test_id", "analysis_type"),
        Index("ix_analysis_results_significance", "statistical_significance"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisResultModel(id={self.id}, test_id={self.test_id}, type='{self.analysis_type}')>"


class ModelPerformanceModel(Base):
    """Model performance database model."""

    __tablename__ = "model_performances"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    analysis_result_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id = Column(String(255), nullable=False)
    provider_name = Column(String(255), nullable=False)

    # Performance metrics
    overall_score = Column(Float, nullable=False)
    dimension_scores = Column(JSON, nullable=False)
    sample_count = Column(Integer, nullable=False)
    success_rate = Column(Float, nullable=False)

    # Statistical metrics
    mean_score = Column(Float, nullable=False)
    median_score = Column(Float, nullable=False)
    std_deviation = Column(Float, nullable=False)
    confidence_interval = Column(JSON, nullable=True)
    percentiles = Column(JSON, nullable=True)

    # Cost and efficiency metrics
    total_cost = Column(Float, nullable=True)
    average_latency_ms = Column(Float, nullable=True)
    input_tokens_total = Column(Integer, nullable=True)
    output_tokens_total = Column(Integer, nullable=True)
    cost_per_sample = Column(Float, nullable=True)

    # Quality metrics
    error_rate = Column(Float, nullable=False, default=0.0)
    timeout_rate = Column(Float, nullable=False, default=0.0)
    quality_score = Column(Float, nullable=True)

    # Rankings and comparisons
    rank = Column(Integer, nullable=True)
    rank_percentile = Column(Float, nullable=True)
    relative_performance = Column(JSON, nullable=True)

    # Timestamps
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Metadata
    analytics_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    analysis_result = relationship("AnalysisResultModel", back_populates="model_performances")

    # Indexes for performance
    __table_args__ = (
        Index("ix_model_performances_analysis_id", "analysis_result_id"),
        Index("ix_model_performances_model_id", "model_id"),
        Index("ix_model_performances_provider", "provider_name"),
        Index("ix_model_performances_overall_score", "overall_score"),
        Index("ix_model_performances_rank", "rank"),
        Index("ix_model_performances_success_rate", "success_rate"),
        Index("ix_model_performances_cost", "total_cost"),
        Index("ix_model_performances_latency", "average_latency_ms"),
        Index("ix_model_performances_calculated_at", "calculated_at"),
        Index("ix_model_performances_model_provider", "model_id", "provider_name"),
    )

    def __repr__(self) -> str:
        return f"<ModelPerformanceModel(id={self.id}, model_id='{self.model_id}', score={self.overall_score})>"


class StatisticalTestModel(Base):
    """Statistical test database model."""

    __tablename__ = "statistical_tests"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    analysis_result_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    test_name = Column(String(255), nullable=False)
    test_type = Column(String(100), nullable=False)

    # Test configuration
    null_hypothesis = Column(Text, nullable=False)
    alternative_hypothesis = Column(Text, nullable=False)
    significance_level = Column(Float, nullable=False, default=0.05)
    power = Column(Float, nullable=True)
    effect_size_threshold = Column(Float, nullable=True)

    # Test inputs
    sample_sizes = Column(JSON, nullable=False)
    group_data = Column(JSON, nullable=False)
    assumptions_checked = Column(JSON, nullable=True)

    # Test results
    test_statistic = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)
    confidence_interval = Column(JSON, nullable=True)
    is_significant = Column(Boolean, nullable=True)

    # Additional metrics
    degrees_of_freedom = Column(Integer, nullable=True)
    critical_value = Column(Float, nullable=True)
    observed_power = Column(Float, nullable=True)

    # Test quality
    assumptions_met = Column(Boolean, nullable=True)
    warnings = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)

    # Timestamps
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Metadata
    analytics_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    analysis_result = relationship("AnalysisResultModel")

    # Indexes for performance
    __table_args__ = (
        Index("ix_statistical_tests_analysis_id", "analysis_result_id"),
        Index("ix_statistical_tests_name", "test_name"),
        Index("ix_statistical_tests_type", "test_type"),
        Index("ix_statistical_tests_p_value", "p_value"),
        Index("ix_statistical_tests_significant", "is_significant"),
        Index("ix_statistical_tests_executed_at", "executed_at"),
        Index("ix_statistical_tests_effect_size", "effect_size"),
    )

    def __repr__(self) -> str:
        return f"<StatisticalTestModel(id={self.id}, test_name='{self.test_name}', p_value={self.p_value})>"


class InsightModel(Base):
    """Insight database model."""

    __tablename__ = "insights"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    analysis_result_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    insight_type = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)

    # Insight content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=True)

    # Supporting data
    supporting_data = Column(JSON, nullable=True)
    evidence_strength = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)

    # Impact metrics
    potential_impact = Column(String(50), nullable=True)
    implementation_difficulty = Column(String(50), nullable=True)
    priority_score = Column(Float, nullable=True)

    # Relevance
    affected_models = Column(JSON, nullable=True)
    affected_dimensions = Column(JSON, nullable=True)
    context_tags = Column(JSON, nullable=True)

    # Timestamps
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # Metadata
    analytics_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    analysis_result = relationship("AnalysisResultModel", back_populates="insights")

    # Indexes for performance
    __table_args__ = (
        Index("ix_insights_analysis_id", "analysis_result_id"),
        Index("ix_insights_type", "insight_type"),
        Index("ix_insights_category", "category"),
        Index("ix_insights_severity", "severity"),
        Index("ix_insights_confidence", "confidence_score"),
        Index("ix_insights_priority", "priority_score"),
        Index("ix_insights_generated_at", "generated_at"),
        Index("ix_insights_expires_at", "expires_at"),
        Index("ix_insights_category_severity", "category", "severity"),
    )

    def __repr__(self) -> str:
        return (
            f"<InsightModel(id={self.id}, type='{self.insight_type}', severity='{self.severity}')>"
        )


class AggregationRuleModel(Base):
    """Aggregation rule database model."""

    __tablename__ = "aggregation_rules"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    rule_type = Column(String(100), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Rule configuration
    aggregation_function = Column(String(100), nullable=False)
    weight_function = Column(String(100), nullable=True)
    filter_criteria = Column(JSON, nullable=True)
    grouping_dimensions = Column(JSON, nullable=False)

    # Rule logic
    mathematical_expression = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True, default=dict)
    validation_rules = Column(JSON, nullable=True)

    # Usage tracking
    usage_count = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Metadata
    analytics_metadata = Column(JSON, nullable=True, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("ix_aggregation_rules_name", "name"),
        Index("ix_aggregation_rules_type", "rule_type"),
        Index("ix_aggregation_rules_active", "is_active"),
        Index("ix_aggregation_rules_usage_count", "usage_count"),
        Index("ix_aggregation_rules_last_used", "last_used_at"),
        Index("ix_aggregation_rules_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AggregationRuleModel(id={self.id}, name='{self.name}', type='{self.rule_type}')>"

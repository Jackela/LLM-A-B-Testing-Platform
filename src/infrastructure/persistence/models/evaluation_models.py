"""Database models for Evaluation domain."""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
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


class JudgeModel(Base):
    """Judge database model."""

    __tablename__ = "judges"

    # Primary fields
    id = Column(String(255), primary_key=True)  # Judge ID is string (e.g., "gpt-4-judge")
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    model_id = Column(String(255), nullable=False)
    provider_name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=False)
    temperature = Column(Float, nullable=False, default=0.0)
    max_tokens = Column(Integer, nullable=False, default=1000)
    is_active = Column(Boolean, nullable=False, default=True)
    is_calibrated = Column(Boolean, nullable=False, default=False)
    calibration_score = Column(Float, nullable=True)
    eval_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Performance metrics
    total_evaluations = Column(Integer, nullable=False, default=0)
    average_latency_ms = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)

    # Relationships
    evaluation_results = relationship(
        "EvaluationResultModel", back_populates="judge", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_judges_name", "name"),
        Index("ix_judges_model_id", "model_id"),
        Index("ix_judges_provider", "provider_name"),
        Index("ix_judges_active", "is_active"),
        Index("ix_judges_calibrated", "is_calibrated"),
        Index("ix_judges_active_calibrated", "is_active", "is_calibrated"),
        Index("ix_judges_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<JudgeModel(id='{self.id}', name='{self.name}', active={self.is_active})>"


class EvaluationTemplateModel(Base):
    """Evaluation template database model."""

    __tablename__ = "evaluation_templates"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Template configuration
    prompt_template = Column(Text, nullable=False)
    response_format = Column(JSON, nullable=False)
    scoring_rubric = Column(JSON, nullable=False)
    eval_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    dimensions = relationship(
        "DimensionModel", back_populates="template", cascade="all, delete-orphan", lazy="selectin"
    )
    evaluation_results = relationship(
        "EvaluationResultModel", back_populates="template", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_evaluation_templates_name", "name"),
        Index("ix_evaluation_templates_version", "version"),
        Index("ix_evaluation_templates_active", "is_active"),
        Index("ix_evaluation_templates_created_by", "created_by"),
        Index("ix_evaluation_templates_name_version", "name", "version"),
        Index("ix_evaluation_templates_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationTemplateModel(id={self.id}, name='{self.name}', version='{self.version}')>"
        )


class DimensionModel(Base):
    """Evaluation dimension database model."""

    __tablename__ = "evaluation_dimensions"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    template_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("evaluation_templates.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    weight = Column(Float, nullable=False, default=1.0)
    min_score = Column(Float, nullable=False, default=0.0)
    max_score = Column(Float, nullable=False, default=1.0)
    criteria = Column(JSON, nullable=False)
    eval_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    template = relationship("EvaluationTemplateModel", back_populates="dimensions")

    # Indexes for performance
    __table_args__ = (
        Index("ix_evaluation_dimensions_template_id", "template_id"),
        Index("ix_evaluation_dimensions_name", "name"),
        Index("ix_evaluation_dimensions_weight", "weight"),
    )

    def __repr__(self) -> str:
        return f"<DimensionModel(id={self.id}, name='{self.name}', weight={self.weight})>"


class EvaluationResultModel(Base):
    """Evaluation result database model."""

    __tablename__ = "evaluation_results"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    test_id = Column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("tests.id", ondelete="CASCADE"), nullable=False
    )
    model_response_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("model_responses.id", ondelete="CASCADE"),
        nullable=False,
    )
    judge_id = Column(String(255), ForeignKey("judges.id", ondelete="CASCADE"), nullable=False)
    template_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("evaluation_templates.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Evaluation data
    overall_score = Column(Float, nullable=False)
    dimension_scores = Column(JSON, nullable=False)
    raw_evaluation = Column(Text, nullable=True)
    parsed_evaluation = Column(JSON, nullable=True)
    is_successful = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)

    # Performance metrics
    evaluation_time_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    evaluation_cost = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Metadata
    eval_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    test = relationship("TestModel", back_populates="evaluation_results")
    model_response = relationship("ModelResponseModel", back_populates="evaluations")
    judge = relationship("JudgeModel", back_populates="evaluation_results")
    template = relationship("EvaluationTemplateModel", back_populates="evaluation_results")

    # Indexes for performance
    __table_args__ = (
        Index("ix_evaluation_results_test_id", "test_id"),
        Index("ix_evaluation_results_model_response_id", "model_response_id"),
        Index("ix_evaluation_results_judge_id", "judge_id"),
        Index("ix_evaluation_results_template_id", "template_id"),
        Index("ix_evaluation_results_overall_score", "overall_score"),
        Index("ix_evaluation_results_successful", "is_successful"),
        Index("ix_evaluation_results_created_at", "created_at"),
        Index("ix_evaluation_results_test_judge", "test_id", "judge_id"),
        Index("ix_evaluation_results_test_template", "test_id", "template_id"),
        Index("ix_evaluation_results_response_judge", "model_response_id", "judge_id"),
    )

    def __repr__(self) -> str:
        return f"<EvaluationResultModel(id={self.id}, test_id={self.test_id}, score={self.overall_score})>"


class CalibrationDataModel(Base):
    """Calibration data database model."""

    __tablename__ = "calibration_data"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    judge_id = Column(String(255), ForeignKey("judges.id", ondelete="CASCADE"), nullable=False)

    # Calibration metrics
    accuracy_score = Column(Float, nullable=False)
    precision_score = Column(Float, nullable=False)
    recall_score = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    bias_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)

    # Calibration details
    sample_size = Column(Integer, nullable=False)
    ground_truth_data = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    confusion_matrix = Column(JSON, nullable=True)

    # Timestamps
    calibrated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # Metadata
    eval_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    judge = relationship("JudgeModel")

    # Indexes for performance
    __table_args__ = (
        Index("ix_calibration_data_judge_id", "judge_id"),
        Index("ix_calibration_data_accuracy", "accuracy_score"),
        Index("ix_calibration_data_calibrated_at", "calibrated_at"),
        Index("ix_calibration_data_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<CalibrationDataModel(id={self.id}, judge_id='{self.judge_id}', accuracy={self.accuracy_score})>"

"""Database models for Test Management domain."""

from datetime import datetime
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import JSON, Boolean, Column, DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import relationship

from ....domain.test_management.value_objects.difficulty_level import DifficultyLevel
from ....domain.test_management.value_objects.test_status import TestStatus
from ..database import Base


class TestModel(Base):
    """Test aggregate root database model."""

    __tablename__ = "tests"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False)
    status = Column(Enum(TestStatus), nullable=False, default=TestStatus.DRAFT)
    configuration = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    failure_reason = Column(Text, nullable=True)

    # Relationships
    samples = relationship(
        "TestSampleModel", back_populates="test", cascade="all, delete-orphan", lazy="selectin"
    )
    model_responses = relationship(
        "ModelResponseModel", back_populates="test", cascade="all, delete-orphan", lazy="select"
    )
    evaluation_results = relationship(
        "EvaluationResultModel", back_populates="test", cascade="all, delete-orphan", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_tests_status", "status"),
        Index("ix_tests_created_at", "created_at"),
        Index("ix_tests_status_created", "status", "created_at"),
        Index("ix_tests_name", "name"),
        Index("ix_tests_completed_at", "completed_at"),
    )

    def __repr__(self) -> str:
        return f"<TestModel(id={self.id}, name='{self.name}', status={self.status})>"


class TestSampleModel(Base):
    """Test sample database model."""

    __tablename__ = "test_samples"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    test_id = Column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("tests.id", ondelete="CASCADE"), nullable=False
    )
    prompt = Column(Text, nullable=False)
    difficulty = Column(Enum(DifficultyLevel), nullable=False)
    expected_output = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True, default=list)
    sample_metadata = Column(JSON, nullable=True, default=dict)
    evaluation_results = Column(JSON, nullable=True, default=dict)
    is_frozen = Column(Boolean, nullable=False, default=False)

    # Relationships
    test = relationship("TestModel", back_populates="samples")

    # Indexes for performance
    __table_args__ = (
        Index("ix_test_samples_test_id", "test_id"),
        Index("ix_test_samples_difficulty", "difficulty"),
        Index("ix_test_samples_test_difficulty", "test_id", "difficulty"),
        # Full text search index for prompt (PostgreSQL specific)
        Index(
            "ix_test_samples_prompt_search",
            "prompt",
            postgresql_using="gin",
            postgresql_ops={"prompt": "gin_trgm_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<TestSampleModel(id={self.id}, test_id={self.test_id}, difficulty={self.difficulty})>"
        )


class TestConfigurationModel(Base):
    """Test configuration database model (if needed separately)."""

    __tablename__ = "test_configurations"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    test_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("tests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    models = Column(JSON, nullable=False)
    evaluation_templates = Column(JSON, nullable=False)
    randomization_seed = Column(String(255), nullable=True)
    parallel_executions = Column(JSON, nullable=False, default=dict)
    timeout_seconds = Column(JSON, nullable=False, default=dict)
    retry_config = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    test = relationship("TestModel", uselist=False)

    # Indexes
    __table_args__ = (
        Index("ix_test_configurations_test_id", "test_id"),
        Index("ix_test_configurations_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<TestConfigurationModel(id={self.id}, test_id={self.test_id})>"

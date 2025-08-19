"""Database models for Model Provider domain."""

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

from ....domain.model_provider.value_objects.health_status import HealthStatus
from ....domain.model_provider.value_objects.model_category import ModelCategory
from ....domain.model_provider.value_objects.provider_type import ProviderType
from ..database import Base


class ModelProviderModel(Base):
    """Model provider database model."""

    __tablename__ = "model_providers"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    provider_type = Column(Enum(ProviderType), nullable=False)
    health_status = Column(Enum(HealthStatus), nullable=False, default=HealthStatus.UNKNOWN)
    api_credentials = Column(JSON, nullable=False)
    provider_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Rate limiting fields
    requests_per_minute = Column(Integer, nullable=False, default=100)
    requests_per_day = Column(Integer, nullable=False, default=10000)
    current_minute_requests = Column(Integer, nullable=False, default=0)
    current_day_requests = Column(Integer, nullable=False, default=0)
    last_reset_minute = Column(DateTime, nullable=True)
    last_reset_day = Column(DateTime, nullable=True)

    # Relationships
    supported_models = relationship(
        "ModelConfigModel", back_populates="provider", cascade="all, delete-orphan", lazy="selectin"
    )
    model_responses = relationship("ModelResponseModel", back_populates="provider", lazy="select")

    # Indexes for performance
    __table_args__ = (
        Index("ix_model_providers_name", "name"),
        Index("ix_model_providers_type", "provider_type"),
        Index("ix_model_providers_health", "health_status"),
        Index("ix_model_providers_created_at", "created_at"),
        Index("ix_model_providers_type_health", "provider_type", "health_status"),
    )

    def __repr__(self) -> str:
        return f"<ModelProviderModel(id={self.id}, name='{self.name}', type={self.provider_type})>"


class ModelConfigModel(Base):
    """Model configuration database model."""

    __tablename__ = "model_configs"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    provider_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("model_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_id = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    model_category = Column(Enum(ModelCategory), nullable=False)
    max_tokens = Column(Integer, nullable=False)
    supports_streaming = Column(Boolean, nullable=False, default=False)
    cost_per_input_token = Column(Float, nullable=False)
    cost_per_output_token = Column(Float, nullable=False)
    supported_parameters = Column(JSON, nullable=False, default=list)
    config_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    provider = relationship("ModelProviderModel", back_populates="supported_models")
    responses = relationship("ModelResponseModel", back_populates="model_config", lazy="select")

    # Indexes for performance
    __table_args__ = (
        Index("ix_model_configs_provider_id", "provider_id"),
        Index("ix_model_configs_model_id", "model_id"),
        Index("ix_model_configs_category", "model_category"),
        Index("ix_model_configs_provider_model", "provider_id", "model_id"),
        Index("ix_model_configs_streaming", "supports_streaming"),
        Index("ix_model_configs_cost_input", "cost_per_input_token"),
    )

    def __repr__(self) -> str:
        return f"<ModelConfigModel(id={self.id}, model_id='{self.model_id}', provider_id={self.provider_id})>"


class ModelResponseModel(Base):
    """Model response database model."""

    __tablename__ = "model_responses"

    # Primary fields
    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    test_id = Column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("tests.id", ondelete="CASCADE"), nullable=False
    )
    sample_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("test_samples.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("model_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_config_id = Column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("model_configs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Request/Response data
    prompt = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    parameters_used = Column(JSON, nullable=True, default=dict)

    # Performance metrics
    latency_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    total_cost = Column(Float, nullable=True)

    # Timestamps
    request_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    response_time = Column(DateTime, nullable=True)

    # Metadata
    response_metadata = Column(JSON, nullable=True, default=dict)

    # Relationships
    test = relationship("TestModel", back_populates="model_responses")
    sample = relationship("TestSampleModel")
    provider = relationship("ModelProviderModel", back_populates="model_responses")
    model_config = relationship("ModelConfigModel", back_populates="responses")
    evaluations = relationship(
        "EvaluationResultModel", back_populates="model_response", lazy="select"
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_model_responses_test_id", "test_id"),
        Index("ix_model_responses_sample_id", "sample_id"),
        Index("ix_model_responses_provider_id", "provider_id"),
        Index("ix_model_responses_model_config_id", "model_config_id"),
        Index("ix_model_responses_status", "status"),
        Index("ix_model_responses_request_time", "request_time"),
        Index("ix_model_responses_test_sample", "test_id", "sample_id"),
        Index("ix_model_responses_provider_model", "provider_id", "model_config_id"),
        Index("ix_model_responses_latency", "latency_ms"),
        Index("ix_model_responses_cost", "total_cost"),
    )

    def __repr__(self) -> str:
        return f"<ModelResponseModel(id={self.id}, test_id={self.test_id}, status='{self.status}')>"

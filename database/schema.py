from sqlalchemy import Column, Integer, String, Text, DateTime, ARRAY, Float, Index, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func
from config.database import Base
import json


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    rate_limit_per_minute = Column(Integer, default=100, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_tenants_tenant_id", "tenant_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "is_active": self.is_active,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String, nullable=False)
    name = Column(String, nullable=False, default="default")
    permissions = Column(String, nullable=False, default="read_write")
    tenant_id = Column(String, ForeignKey("tenants.tenant_id", ondelete="CASCADE"), nullable=True)
    collection_id = Column(String, ForeignKey("collections.collection_id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_api_keys_key_hash", "key_hash"),
        Index("idx_api_keys_tenant_id", "tenant_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "key_hash": self.key_hash,
            "name": self.name,
            "permissions": self.permissions,
            "tenant_id": self.tenant_id,
            "collection_id": self.collection_id,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    modality = Column(String, nullable=False, default="text")
    embedding_model = Column(String, nullable=False)
    dimension = Column(Integer, nullable=False)
    distance_metric = Column(String, nullable=False, default="cosine")
    tenant_id = Column(String, ForeignKey("tenants.tenant_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_collections_collection_id", "collection_id"),
        Index("idx_collections_modality", "modality"),
        Index("idx_collections_tenant_id", "tenant_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "name": self.name,
            "description": self.description,
            "modality": self.modality,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Vector(Base):
    __tablename__ = "vectors"
    
    id = Column(Integer, primary_key=True, index=True)
    vector_data = Column(ARRAY(Float), nullable=False)
    meta_data = Column(JSON, nullable=True)  # Using JSON for flexible metadata
    vector_id = Column(String, unique=True, index=True, nullable=False)
    collection_id = Column(
        String,
        ForeignKey("collections.collection_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes for better performance
    __table_args__ = (
        Index('idx_vectors_vector_id', 'vector_id'),
        Index('idx_vectors_created_at', 'created_at'),
        Index('idx_vectors_collection_id', 'collection_id'),
    )
    
    def __repr__(self):
        return f"<Vector(id={self.id}, vector_id={self.vector_id})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "vector_data": self.vector_data,
            "meta_data": self.meta_data,
            "vector_id": self.vector_id,
            "collection_id": self.collection_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class VectorPgVector(Base):
    __tablename__ = "vectors_pgvector"

    id = Column(Integer, primary_key=True, index=True)
    vector_id = Column(String, unique=True, index=True, nullable=False)
    vector = Column(ARRAY(Float), nullable=False)
    meta_data = Column(JSON, nullable=True)
    collection_id = Column(String, nullable=True, index=True)
    content_type = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "vector_id": self.vector_id,
            "vector": self.vector,
            "meta_data": self.meta_data,
            "collection_id": self.collection_id,
            "content_type": self.content_type,
            "created_at": self.created_at,
        }


class VectorBatch(Base):
    __tablename__ = "vector_batches"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_name = Column(String, nullable=False, index=True)
    batch_size = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(Text, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "batch_name": self.batch_name,
            "batch_size": self.batch_size,
            "created_at": self.created_at,
            "description": self.description
        }

class VectorBatchMapping(Base):
    __tablename__ = "vector_batch_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_batch_mappings_batch_id', 'batch_id'),
        Index('idx_batch_mappings_vector_id', 'vector_id'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "vector_id": self.vector_id,
            "created_at": self.created_at
        }


class ApiTemplate(Base):
    __tablename__ = "api_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    method = Column(String, nullable=False)
    path = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_api_templates_name', 'name'),
        Index('idx_api_templates_path', 'path'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "path": self.path,
            "payload": self.payload,
            "created_at": self.created_at
        }


class FeedbackEntry(Base):
    __tablename__ = "feedback_entries"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    rating = Column(Integer, nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_feedback_entries_created_at', 'created_at'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "rating": self.rating,
            "message": self.message,
            "created_at": self.created_at
        }
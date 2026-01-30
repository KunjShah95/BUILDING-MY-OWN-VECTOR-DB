from sqlalchemy import Column, Integer, String, Text, DateTime, ARRAY, Float, Index, JSON
from sqlalchemy.sql import func
from config.database import Base
import json

class Vector(Base):
    __tablename__ = "vectors"
    
    id = Column(Integer, primary_key=True, index=True)
    vector_data = Column(ARRAY(Float), nullable=False)
    meta_data = Column(JSON, nullable=True)  # Using JSON for flexible metadata
    vector_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes for better performance
    __table_args__ = (
        Index('idx_vectors_vector_id', 'vector_id'),
        Index('idx_vectors_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Vector(id={self.id}, vector_id={self.vector_id})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "vector_data": self.vector_data,
            "meta_data": self.meta_data,
            "vector_id": self.vector_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at
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
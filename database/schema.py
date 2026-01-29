from sqlalchemy import Column, Integer, String, Text, DateTime, ARRAY, Float
from sqlalchemy.sql import func
from config.database import Base
import numpy as np 

class Vector(Base):
    __tablename__ = "vectors"
    id = Column(Integer, primary_key=True, index=True)
    vector_data = Column(ARRAY(Float), nullable=False)
    meta_data = Column(Text, nullable=True)
    vector_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def _repr_(self):
        return f"<Vector(id={self.id}, vector_id={self.vector_id})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "vector_data": self.vector_data,
            "meta_data": self.meta_data,
            "vector_id": self.vector_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
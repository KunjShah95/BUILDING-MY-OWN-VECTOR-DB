from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import get_settings

settings = get_settings()

# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,          # Persistent connections in pool
    max_overflow=20,       # Extra connections under load
    pool_timeout=30,       # Wait time for connection from pool
    pool_recycle=3600,     # Recycle connections after 1 hour (prevents stale)
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Async database configuration — ported from ANN Search Engine
from typing import Any, AsyncGenerator
import logging

logger = logging.getLogger(__name__)

# Global engine and session factory
_async_engine = None
_async_session_maker = None

async def init_async_db(
    database_url: str = None,
    pool_size: int = 10,
    max_overflow: int = 20,
    pool_recycle: int = 3600,
    echo: bool = False
):
    """
    Initialize async database with connection pooling
    """
    global _async_engine, _async_session_maker
    
    if _async_engine is not None:
        logger.warning("Database already initialized")
        return
    
    if not database_url:
        database_url = settings.DATABASE_URL
        
    # Convert to asyncpg URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    elif not database_url.startswith("postgresql+asyncpg://"):
        database_url = f"postgresql+asyncpg://{database_url}"
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from sqlalchemy import text
        
        _async_engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Verify connections before use
            echo=echo,
            future=True
        )
        
        # Test connection
        async with _async_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
        
        _async_session_maker = async_sessionmaker(
            _async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False
        )
        
        logger.info("Async database initialized with connection pooling")
        
    except ImportError:
        logger.warning("asyncpg or sqlalchemy.ext.asyncio not installed. Async database disabled.")
        _async_engine = None
        _async_session_maker = None
    except Exception as e:
        logger.error(f"Failed to initialize async database: {e}")
        raise


async def close_async_db():
    """Close database connections"""
    global _async_engine, _async_session_maker
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_maker = None
        logger.info("Database connections closed")


async def get_async_session() -> AsyncGenerator[Any, None]: # Using Any for generic AsyncSession typing
    """
    Get async database session
    """
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_async_db() first.")
    
    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> dict:
    """
    Check database health status
    """
    if _async_engine is None:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": "Database not initialized"
        }
    
    try:
        from sqlalchemy import text
        import time
        async with _async_engine.connect() as conn:
            start_time = time.time()
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
            response_time = (time.time() - start_time) * 1000
            
            # Get connection pool stats
            pool = _async_engine.pool
            return {
                "status": "healthy",
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

def get_engine():
    """Get the async engine instance"""
    return _async_engine

def is_initialized() -> bool:
    """Check if database is initialized"""
    return _async_engine is not None


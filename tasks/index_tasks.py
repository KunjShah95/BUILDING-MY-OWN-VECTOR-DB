"""
Background tasks for index operations
"""
import logging
import time
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
import numpy as np

from celery_app import celery_app
from config.settings import get_settings
from config.database import get_async_session

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def build_index(self, index_type: str, vector_ids: list = None, options: dict = None):
    """
    Build index in background
    
    Args:
        index_type: Type of index ('hnsw', 'ivf', 'brute')
        vector_ids: Optional list of vector IDs to include
        options: Index-specific options
    """
    try:
        logger.info(f"Starting background index build: {index_type}")
        start_time = time.time()
        
        # Import here to avoid circular imports
        from services.index_service import IndexService
        
        # Run async code in sync context
        import asyncio
        
        async def _build():
            async with get_async_session() as session:
                service = IndexService(session)
                
                # Get vectors
                if vector_ids:
                    vectors = await service.get_vectors_by_ids(vector_ids)
                else:
                    vectors = await service.get_all_vectors()
                
                # Build index
                result = await service.create_index(
                    index_type=index_type,
                    vectors=vectors,
                    options=options
                )
                
                return result
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_build())
        loop.close()
        
        duration = time.time() - start_time
        logger.info(f"Index build completed in {duration:.2f}s: {index_type}")
        
        return {
            "success": True,
            "index_type": index_type,
            "duration_seconds": duration,
            "vector_count": result.get("vector_count", 0),
            "task_id": self.request.id
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Index build timed out: {index_type}")
        raise self.retry(exc=Exception("Timeout"), countdown=300)
        
    except Exception as exc:
        logger.error(f"Index build failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=2)
def optimize_index(self, index_type: str):
    """
    Optimize existing index for better performance
    """
    try:
        logger.info(f"Optimizing index: {index_type}")
        
        import asyncio
        from services.index_service import IndexService
        
        async def _optimize():
            async with get_async_session() as session:
                service = IndexService(session)
                return await service.optimize_index(index_type)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_optimize())
        loop.close()
        
        logger.info(f"Index optimization completed: {index_type}")
        return result
        
    except Exception as exc:
        logger.error(f"Index optimization failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task
def save_all_indexes():
    """
    Save all indexes to disk
    """
    try:
        logger.info("Saving all indexes")
        
        import asyncio
        from services.index_service import IndexService
        
        async def _save():
            async with get_async_session() as session:
                service = IndexService(session)
                results = {}
                for index_type in ['hnsw', 'ivf', 'brute']:
                    try:
                        result = await service.save_index(index_type)
                        results[index_type] = result
                    except Exception as e:
                        results[index_type] = {"success": False, "error": str(e)}
                return results
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(_save())
        loop.close()
        
        logger.info("All indexes saved")
        return results
        
    except Exception as exc:
        logger.error(f"Save indexes failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task
def load_index_from_disk(index_type: str, filepath: str):
    """
    Load index from disk in background
    """
    try:
        logger.info(f"Loading index from disk: {index_type} from {filepath}")
        
        import asyncio
        from services.index_service import IndexService
        
        async def _load():
            async with get_async_session() as session:
                service = IndexService(session)
                return await service.load_index(index_type, filepath)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_load())
        loop.close()
        
        logger.info(f"Index loaded: {index_type}")
        return result
        
    except Exception as exc:
        logger.error(f"Load index failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task(bind=True, max_retries=3)
def incremental_index_update(self, index_type: str, new_vectors: list):
    """
    Incrementally update index with new vectors
    """
    try:
        logger.info(f"Incremental update for {index_type}: {len(new_vectors)} vectors")
        
        import asyncio
        from services.index_service import IndexService
        
        async def _update():
            async with get_async_session() as session:
                service = IndexService(session)
                return await service.add_vectors_to_index(index_type, new_vectors)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_update())
        loop.close()
        
        logger.info(f"Incremental update completed: {index_type}")
        return result
        
    except Exception as exc:
        logger.error(f"Incremental update failed: {exc}")
        raise self.retry(exc=exc, countdown=30)

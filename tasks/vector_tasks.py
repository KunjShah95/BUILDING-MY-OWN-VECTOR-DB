"""
Background tasks for vector operations
"""
import logging
import time
from typing import List, Dict, Any

from celery_app import celery_app
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def bulk_import_vectors(self, vectors_data: List[Dict[str, Any]], batch_name: str = None):
    """
    Bulk import vectors in background
    
    Args:
        vectors_data: List of vector data dictionaries
        batch_name: Optional name for the batch
    """
    try:
        logger.info(f"Starting bulk import: {len(vectors_data)} vectors, batch: {batch_name}")
        start_time = time.time()
        
        import asyncio
        from services.vector_service import VectorService
        from config.database import get_async_session
        
        results = {
            "success_count": 0,
            "failed_count": 0,
            "errors": [],
            "batch_name": batch_name
        }
        
        # Process in batches
        batch_size = settings.bulk_batch_size
        total_batches = (len(vectors_data) + batch_size - 1) // batch_size
        
        async def _import():
            async with get_async_session() as session:
                service = VectorService(session)
                
                for i in range(0, len(vectors_data), batch_size):
                    batch = vectors_data[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    try:
                        # Update progress
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'current': batch_num,
                                'total': total_batches,
                                'percent': int((batch_num / total_batches) * 100),
                                'success_count': results['success_count'],
                                'failed_count': results['failed_count']
                            }
                        )
                        
                        # Import batch
                        batch_result = await service.create_vector_batch(batch)
                        
                        if batch_result.get('success'):
                            results['success_count'] += batch_result.get('vector_count', len(batch))
                        else:
                            results['failed_count'] += len(batch)
                            results['errors'].append(batch_result.get('message', 'Unknown error'))
                            
                    except Exception as e:
                        results['failed_count'] += len(batch)
                        results['errors'].append(str(e))
                        logger.error(f"Batch {batch_num} failed: {e}")
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_import())
        loop.close()
        
        duration = time.time() - start_time
        
        logger.info(
            f"Bulk import completed: {results['success_count']} success, "
            f"{results['failed_count']} failed in {duration:.2f}s"
        )
        
        return {
            "success": results['failed_count'] == 0,
            "total": len(vectors_data),
            "success_count": results['success_count'],
            "failed_count": results['failed_count'],
            "duration_seconds": duration,
            "errors": results['errors'][:10],  # Limit errors in response
            "batch_name": batch_name
        }
        
    except Exception as exc:
        logger.error(f"Bulk import failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=2)
def bulk_delete_vectors(self, vector_ids: List[str]):
    """
    Bulk delete vectors in background
    """
    try:
        logger.info(f"Starting bulk delete: {len(vector_ids)} vectors")
        
        import asyncio
        from services.vector_service import VectorService
        from config.database import get_async_session
        
        results = {"deleted": 0, "not_found": 0, "errors": []}
        
        async def _delete():
            async with get_async_session() as session:
                service = VectorService(session)
                
                for i, vector_id in enumerate(vector_ids):
                    # Update progress every 100 items
                    if i % 100 == 0:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'current': i,
                                'total': len(vector_ids),
                                'percent': int((i / len(vector_ids)) * 100)
                            }
                        )
                    
                    try:
                        result = await service.delete_vector(vector_id)
                        if result.get('success'):
                            results['deleted'] += 1
                        else:
                            results['not_found'] += 1
                    except Exception as e:
                        results['errors'].append(f"{vector_id}: {str(e)}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_delete())
        loop.close()
        
        logger.info(f"Bulk delete completed: {results['deleted']} deleted")
        return results
        
    except Exception as exc:
        logger.error(f"Bulk delete failed: {exc}")
        raise self.retry(exc=exc, countdown=30)


@celery_app.task
def update_vector_metadata(vector_id: str, metadata: Dict[str, Any]):
    """
    Update vector metadata in background
    """
    try:
        import asyncio
        from services.vector_service import VectorService
        from config.database import get_async_session
        
        async def _update():
            async with get_async_session() as session:
                service = VectorService(session)
                return await service.update_vector_metadata(vector_id, metadata)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_update())
        loop.close()
        
        return result
        
    except Exception as exc:
        logger.error(f"Metadata update failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task(bind=True, max_retries=3)
def reindex_vectors(self, index_type: str = None, filter_criteria: Dict = None):
    """
    Reindex vectors based on criteria
    """
    try:
        logger.info(f"Starting reindex: type={index_type}, filter={filter_criteria}")
        
        import asyncio
        from services.vector_service import VectorService
        from services.index_service import IndexService
        from config.database import get_async_session
        
        async def _reindex():
            async with get_async_session() as session:
                vector_service = VectorService(session)
                index_service = IndexService(session)
                
                # Get vectors matching criteria
                vectors = await vector_service.get_vectors_by_filter(filter_criteria)
                
                # Rebuild index
                result = await index_service.create_index(
                    index_type=index_type or settings.default_index_type,
                    vectors=vectors
                )
                
                return result
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_reindex())
        loop.close()
        
        logger.info("Reindex completed")
        return result
        
    except Exception as exc:
        logger.error(f"Reindex failed: {exc}")
        raise self.retry(exc=exc, countdown=60)


@celery_app.task
def validate_vectors(vector_ids: List[str] = None):
    """
    Validate vector integrity and consistency
    """
    try:
        logger.info("Starting vector validation")
        
        import asyncio
        from services.vector_service import VectorService
        from config.database import get_async_session
        
        async def _validate():
            async with get_async_session() as session:
                service = VectorService(session)
                
                if vector_ids:
                    vectors = await service.get_vectors_by_ids(vector_ids)
                else:
                    vectors = await service.get_all_vectors()
                
                issues = []
                valid_count = 0
                
                for vector in vectors:
                    # Check dimension consistency
                    if 'vector_data' in vector:
                        dim = len(vector['vector_data'])
                        if dim != settings.default_dimension:
                            issues.append({
                                'vector_id': vector.get('vector_id'),
                                'issue': 'dimension_mismatch',
                                'expected': settings.default_dimension,
                                'actual': dim
                            })
                        else:
                            valid_count += 1
                
                return {
                    "total_checked": len(vectors),
                    "valid_count": valid_count,
                    "issues": issues,
                    "success": len(issues) == 0
                }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_validate())
        loop.close()
        
        logger.info(f"Validation completed: {result['valid_count']}/{result['total_checked']} valid")
        return result
        
    except Exception as exc:
        logger.error(f"Validation failed: {exc}")
        return {"success": False, "error": str(exc)}

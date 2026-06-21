"""
Maintenance and monitoring background tasks
"""
import logging
import os
import time
import shutil
from datetime import datetime, timedelta
from celery_app import celery_app
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task
def health_check():
    """
    Periodic health check task
    """
    try:
        import asyncio
        from config.database import check_database_health
        from services.cache_service import cache_manager
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "database": None,
            "cache": None,
            "disk": None
        }
        
        # Check database
        async def check_db():
            return await check_database_health()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health_status["database"] = loop.run_until_complete(check_db())
        loop.close()
        
        # Check cache
        if cache_manager._enabled and cache_manager._redis:
            try:
                import asyncio
                async def check_cache():
                    return await cache_manager._redis.ping()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ping_result = loop.run_until_complete(check_cache())
                loop.close()
                health_status["cache"] = {"status": "healthy" if ping_result else "unhealthy"}
            except Exception as e:
                health_status["cache"] = {"status": "unhealthy", "error": str(e)}
        else:
            health_status["cache"] = {"status": "disabled"}
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(settings.data_storage_path)
            free_percent = (disk_usage.free / disk_usage.total) * 100
            health_status["disk"] = {
                "status": "healthy" if free_percent > 10 else "warning",
                "free_percent": round(free_percent, 2),
                "free_gb": round(disk_usage.free / (1024**3), 2)
            }
        except Exception as e:
            health_status["disk"] = {"status": "error", "error": str(e)}
        
        # Log health status
        is_healthy = all(
            s.get("status") in ["healthy", "disabled"] 
            for s in [health_status["database"], health_status["cache"], health_status["disk"]]
            if s
        )
        
        if is_healthy:
            logger.debug(f"Health check passed: {health_status}")
        else:
            logger.warning(f"Health check issues detected: {health_status}")
        
        return health_status
        
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {"status": "error", "error": str(exc)}


@celery_app.task
def cleanup_old_logs():
    """
    Clean up old log files
    """
    try:
        log_dir = settings.log_storage_path
        if not os.path.exists(log_dir):
            return {"success": True, "message": "Log directory does not exist"}
        
        deleted_count = 0
        total_freed = 0
        cutoff_date = datetime.now() - timedelta(days=7)  # Keep 7 days
        
        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(log_dir, filename)
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if mtime < cutoff_date:
                        size = os.path.getsize(filepath)
                        os.remove(filepath)
                        deleted_count += 1
                        total_freed += size
                        logger.info(f"Deleted old log: {filename}")
                        
                except Exception as e:
                    logger.error(f"Error deleting log {filename}: {e}")
        
        return {
            "success": True,
            "deleted_files": deleted_count,
            "freed_bytes": total_freed,
            "freed_mb": round(total_freed / (1024 * 1024), 2)
        }
        
    except Exception as exc:
        logger.error(f"Log cleanup failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task
def backup_indexes():
    """
    Backup all indexes to backup directory
    """
    try:
        import asyncio
        from services.index_service import IndexService
        from config.database import get_async_session
        
        backup_dir = settings.backup_storage_path
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_results = {}
        
        async def _backup():
            async with get_async_session() as session:
                service = IndexService(session)
                
                for index_type in ['hnsw', 'ivf', 'brute']:
                    try:
                        backup_path = os.path.join(
                            backup_dir, 
                            f"{index_type}_index_{timestamp}.bin"
                        )
                        result = await service.backup_index(index_type, backup_path)
                        backup_results[index_type] = result
                    except Exception as e:
                        backup_results[index_type] = {"success": False, "error": str(e)}
                        logger.error(f"Backup failed for {index_type}: {e}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_backup())
        loop.close()
        
        # Clean up old backups (keep last 10)
        cleanup_old_backups(backup_dir, keep_count=10)
        
        logger.info(f"Index backup completed: {backup_results}")
        return backup_results
        
    except Exception as exc:
        logger.error(f"Index backup failed: {exc}")
        return {"success": False, "error": str(exc)}


def cleanup_old_backups(backup_dir: str, keep_count: int = 10):
    """
    Keep only the most recent backups
    """
    try:
        backups = []
        for filename in os.listdir(backup_dir):
            if filename.endswith(('.bin', '.json', '.zip')):
                filepath = os.path.join(backup_dir, filename)
                backups.append({
                    'path': filepath,
                    'mtime': os.path.getmtime(filepath)
                })
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Delete old backups
        for backup in backups[keep_count:]:
            try:
                os.remove(backup['path'])
                logger.info(f"Deleted old backup: {backup['path']}")
            except Exception as e:
                logger.error(f"Error deleting backup {backup['path']}: {e}")
                
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")


@celery_app.task
def update_statistics():
    """
    Update and cache system statistics
    """
    try:
        import asyncio
        from services.index_service import IndexService
        from services.vector_service import VectorService
        from config.database import get_async_session
        from services.cache_service import cache_manager
        
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "vectors": {},
            "indexes": {},
            "system": {}
        }
        
        async def _update_stats():
            async with get_async_session() as session:
                # Vector statistics
                vector_service = VectorService(session)
                stats["vectors"] = await vector_service.get_statistics()
                
                # Index statistics
                index_service = IndexService(session)
                stats["indexes"] = await index_service.get_all_index_stats()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_update_stats())
        loop.close()
        
        # System stats
        import psutil
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage(settings.data_storage_path).percent
        }
        
        # Cache the statistics
        if cache_manager._enabled:
            import asyncio
            async def cache_stats():
                await cache_manager.set("system:statistics", stats, ttl=300)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cache_stats())
            loop.close()
        
        return stats
        
    except Exception as exc:
        logger.error(f"Statistics update failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task
def cleanup_expired_cache():
    """
    Clean up expired cache entries (Redis handles this automatically, 
    but this task can be used for additional cleanup)
    """
    try:
        from services.cache_service import cache_manager
        
        if not cache_manager._enabled:
            return {"success": True, "message": "Cache is disabled"}
        
        # Redis automatically expires keys, but we can log cache stats
        import asyncio
        async def get_stats():
            if cache_manager._redis:
                info = await cache_manager._redis.info('memory')
                return {
                    "used_memory": info.get('used_memory', 0),
                    "used_memory_human": info.get('used_memory_human', '0B'),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0)
                }
            return {}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats = loop.run_until_complete(get_stats())
        loop.close()
        
        logger.info(f"Cache stats: {stats}")
        return {"success": True, "stats": stats}
        
    except Exception as exc:
        logger.error(f"Cache cleanup failed: {exc}")
        return {"success": False, "error": str(exc)}


@celery_app.task(bind=True, max_retries=3)
def optimize_database(self):
    """
    Run database optimization (VACUUM, ANALYZE)
    """
    try:
        logger.info("Starting database optimization")
        
        import asyncio
        from config.database import get_async_engine
        
        async def _optimize():
            engine = get_async_engine()
            async with engine.begin() as conn:
                # Run VACUUM ANALYZE
                await conn.execute("VACUUM ANALYZE")
                logger.info("Database VACUUM ANALYZE completed")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_optimize())
        loop.close()
        
        return {"success": True, "message": "Database optimization completed"}
        
    except Exception as exc:
        logger.error(f"Database optimization failed: {exc}")
        raise self.retry(exc=exc, countdown=300)

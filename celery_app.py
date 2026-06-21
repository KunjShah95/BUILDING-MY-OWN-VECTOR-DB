"""
Celery configuration for background tasks
Handles index building, bulk operations, and scheduled maintenance
"""
import os
import sys
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    'ann_search_engine',
    broker=settings.celery_broker_url or settings.redis_url,
    backend=settings.celery_result_backend or settings.redis_url,
    include=[
        'tasks.index_tasks',
        'tasks.vector_tasks',
        'tasks.maintenance_tasks',
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_always_eager=settings.celery_task_always_eager,  # For testing
    task_store_eager_result=True,
    task_ignore_result=False,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3000,  # 50 minutes soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=500000,  # 500MB
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,
    
    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_heartbeat=30,
    
    # Task routing
    task_routes={
        'tasks.index_tasks.*': {'queue': 'index'},
        'tasks.vector_tasks.*': {'queue': 'vector'},
        'tasks.maintenance_tasks.*': {'queue': 'maintenance'},
    },
    
    # Scheduled tasks
    beat_schedule={
        'health-check': {
            'task': 'tasks.maintenance_tasks.health_check',
            'schedule': 60.0,  # Every minute
        },
        'cleanup-old-logs': {
            'task': 'tasks.maintenance_tasks.cleanup_old_logs',
            'schedule': 3600.0,  # Every hour
        },
        'backup-indexes': {
            'task': 'tasks.maintenance_tasks.backup_indexes',
            'schedule': 86400.0,  # Every day
        },
        'update-statistics': {
            'task': 'tasks.maintenance_tasks.update_statistics',
            'schedule': 300.0,  # Every 5 minutes
        },
    },
)


# Task signals for monitoring
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **extras):
    """Log task start"""
    logger.info(f"Task started: {task.name}[{task_id}]")


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **extras):
    """Log task completion"""
    logger.info(f"Task completed: {task.name}[{task_id}] with state {state}")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **extras):
    """Log task failure"""
    logger.error(f"Task failed: {task_id} with exception {exception}")


if __name__ == '__main__':
    celery_app.start()

"""
Plugin marketplace API router.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.plugin_registry import get_registry, PluginManifest
from services.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/plugins", tags=["Plugins"])

_loader = PluginLoader()
# name -> live plugin instance (for search/stats)
_instances: Dict[str, Any] = {}


# ── request/response models ────────────────────────────────────────────────

class LoadPluginRequest(BaseModel):
    path: str


class SearchPluginRequest(BaseModel):
    query_vector: List[float]
    k: int = 5


# ── helpers ────────────────────────────────────────────────────────────────

def _manifest_to_dict(m: PluginManifest) -> Dict[str, Any]:
    return {
        "name": m.name,
        "version": m.version,
        "plugin_type": m.plugin_type,
        "description": m.description,
        "author": m.author,
        "entry_point": m.entry_point,
        "requires": m.requires,
    }


# ── endpoints ──────────────────────────────────────────────────────────────

@router.get("")
async def list_plugins(plugin_type: Optional[str] = None):
    """List all registered plugins, optionally filtered by type."""
    registry = get_registry()
    manifests = registry.list_plugins(plugin_type=plugin_type)
    return {
        "success": True,
        "plugins": [_manifest_to_dict(m) for m in manifests],
        "total": len(manifests),
    }


@router.post("/load")
async def load_plugin(body: LoadPluginRequest):
    """Hot-load a plugin from a file path and register it."""
    try:
        plugin_class = _loader.load_from_file(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Instantiate so it's ready for search/stats
    try:
        instance = plugin_class()
        name = plugin_class.__name__
        _instances[name] = instance
    except Exception as exc:
        logger.warning("Could not instantiate plugin: %s", exc)
        name = plugin_class.__name__

    return {
        "success": True,
        "message": f"Plugin '{name}' loaded from {body.path}",
        "class": name,
    }


@router.delete("/{name}")
async def unregister_plugin(name: str, version: Optional[str] = None):
    """Unregister a plugin by name (and optional version)."""
    registry = get_registry()
    # If no version given, remove the latest
    if version is None:
        try:
            manifest, _ = registry.get(name)
            version = manifest.version
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found")

    removed = registry.unregister(name, version)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' v{version} not found")

    _instances.pop(name, None)
    return {"success": True, "message": f"Plugin '{name}' v{version} unregistered"}


@router.post("/{name}/search")
async def search_with_plugin(name: str, body: SearchPluginRequest):
    """Use a named plugin to run a vector search."""
    instance = _instances.get(name)
    if instance is None:
        # Try to get it from registry and instantiate
        registry = get_registry()
        try:
            _, plugin_class = registry.get(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Plugin '{name}' not found or not loaded")
        if plugin_class is None:
            raise HTTPException(
                status_code=400,
                detail=f"Plugin '{name}' is registered but class is not available (re-load it)",
            )
        instance = plugin_class()
        _instances[name] = instance

    try:
        results = instance.search(body.query_vector, body.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Plugin search error: {exc}")

    return {"success": True, "results": results}


@router.get("/{name}/stats")
async def plugin_stats(name: str):
    """Return stats from a loaded plugin instance."""
    instance = _instances.get(name)
    if instance is None:
        raise HTTPException(
            status_code=404,
            detail=f"Plugin '{name}' not loaded; POST /api/plugins/load first",
        )
    try:
        stats = instance.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Plugin stats error: {exc}")

    return {"success": True, "name": name, "stats": stats}

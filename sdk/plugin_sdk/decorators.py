"""
@plugin decorator — auto-registers a class with the global PluginRegistry.
"""
from __future__ import annotations

import sys
import os
from typing import Type

# Allow running from the repo root without installing the package.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from services.plugin_registry import PluginManifest, get_registry


def plugin(
    name: str,
    version: str,
    plugin_type: str,
    description: str = "",
    author: str = "",
    requires=None,
    entry_point: str = "",
):
    """
    Class decorator that creates a PluginManifest and registers the class
    with the global PluginRegistry.

    Usage::

        @plugin(name="my_index", version="1.0.0", plugin_type="index")
        class MyIndexPlugin(BaseIndexPlugin):
            ...
    """
    def decorator(cls: Type) -> Type:
        manifest = PluginManifest(
            name=name,
            version=version,
            plugin_type=plugin_type,
            entry_point=entry_point or f"{cls.__module__}.{cls.__qualname__}",
            description=description,
            author=author,
            requires=requires or [],
        )
        get_registry().register(manifest, cls)
        return cls

    return decorator

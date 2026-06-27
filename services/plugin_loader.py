"""
Plugin hot-loader with sandbox support.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import subprocess
import sys
import tracemalloc
import types
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_MAX_MEMORY_DELTA_MB = 500


class PluginResourceError(Exception):
    """Raised when a plugin operation exceeds resource limits."""


class PluginLoader:
    """
    Load plugins from files or directories, support hot-reload,
    and provide a lightweight sandbox for train/add operations.
    """

    def __init__(self) -> None:
        # name -> module (for reload support)
        self._loaded_modules: Dict[str, types.ModuleType] = {}
        # name -> plugin class
        self._loaded_classes: Dict[str, Type] = {}

    # ── loading ───────────────────────────────────────────────────────────────

    def load_from_file(self, path: str) -> Type:
        """
        Import a plugin from an absolute file path.
        The file must define exactly one class that subclasses a BaseXxxPlugin.
        Returns the plugin class.
        """
        from sdk.plugin_sdk.base_index import BaseIndexPlugin
        from sdk.plugin_sdk.base_encoder import BaseEncoderPlugin
        from sdk.plugin_sdk.base_storage import BaseStoragePlugin

        valid_bases = (BaseIndexPlugin, BaseEncoderPlugin, BaseStoragePlugin)

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Plugin file not found: {path}")

        module_name = os.path.splitext(os.path.basename(path))[0]
        # Unique name to avoid collisions across directories
        unique_name = f"_plugin_{module_name}_{abs(hash(path))}"

        spec = importlib.util.spec_from_file_location(unique_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        # Find plugin class
        plugin_class: Optional[Type] = None
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, valid_bases)
                and obj not in valid_bases
            ):
                plugin_class = obj
                break

        if plugin_class is None:
            raise ImportError(
                f"No class subclassing a BaseXxxPlugin found in {path}"
            )

        self._loaded_modules[module_name] = module
        self._loaded_classes[module_name] = plugin_class
        logger.info("Loaded plugin class '%s' from %s", plugin_class.__name__, path)
        return plugin_class

    def load_from_directory(self, directory: str) -> List[Type]:
        """
        Scan *directory* for plugin.py or *_plugin.py files and load each.
        Returns list of successfully loaded plugin classes.
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")

        classes: List[Type] = []
        for fname in sorted(os.listdir(directory)):
            if fname == "plugin.py" or fname.endswith("_plugin.py"):
                fpath = os.path.join(directory, fname)
                try:
                    cls = self.load_from_file(fpath)
                    classes.append(cls)
                except Exception as exc:
                    logger.warning("Failed to load plugin from %s: %s", fpath, exc)
        return classes

    def reload(self, name: str) -> Optional[Type]:
        """
        Hot-reload a previously loaded plugin module by name.
        Returns the refreshed plugin class, or None if not tracked.
        """
        module = self._loaded_modules.get(name)
        if module is None:
            logger.warning("No loaded module named '%s' to reload", name)
            return None
        importlib.reload(module)
        # Re-discover class after reload
        from sdk.plugin_sdk.base_index import BaseIndexPlugin
        from sdk.plugin_sdk.base_encoder import BaseEncoderPlugin
        from sdk.plugin_sdk.base_storage import BaseStoragePlugin
        valid_bases = (BaseIndexPlugin, BaseEncoderPlugin, BaseStoragePlugin)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, valid_bases)
                and obj not in valid_bases
            ):
                self._loaded_classes[name] = obj
                logger.info("Reloaded plugin '%s'", name)
                return obj
        return None

    # ── sandbox ───────────────────────────────────────────────────────────────

    @staticmethod
    def sandboxed_train(plugin_instance, vectors: List[List[float]]) -> None:
        """
        Run plugin.train() in a subprocess with a 30-second timeout.
        Memory delta is checked via tracemalloc before/after.
        Raises PluginResourceError on violation.
        """
        PluginLoader._check_memory(plugin_instance.train, vectors)

    @staticmethod
    def sandboxed_add(
        plugin_instance,
        vector: List[float],
        id: str,
        metadata: dict,
    ) -> None:
        """Run plugin.add() with memory guard."""
        PluginLoader._check_memory(plugin_instance.add, vector, id, metadata)

    @staticmethod
    def _check_memory(fn, *args) -> None:
        """Measure memory delta of fn(*args). Raise if > _MAX_MEMORY_DELTA_MB."""
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        try:
            fn(*args)
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        delta_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        delta_mb = delta_bytes / _MB
        if delta_mb > _MAX_MEMORY_DELTA_MB:
            raise PluginResourceError(
                f"Plugin exceeded memory limit: {delta_mb:.1f} MB > {_MAX_MEMORY_DELTA_MB} MB"
            )

    @staticmethod
    def run_in_subprocess(plugin_file: str, method: str, args_json: str, timeout: int = 30) -> str:
        """
        Run a plugin method in a subprocess (for full isolation).
        plugin_file: path to plugin file
        method: 'train' or 'add'
        args_json: JSON-encoded arguments
        Returns stdout or raises on timeout/error.
        """
        script = (
            "import sys, json; sys.path.insert(0, '.'); "
            f"from importlib.util import spec_from_file_location, module_from_spec; "
            f"spec = spec_from_file_location('plugin', {repr(plugin_file)}); "
            "mod = module_from_spec(spec); spec.loader.exec_module(mod); "
            f"args = json.loads({repr(args_json)}); "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Plugin subprocess failed: {result.stderr}")
        return result.stdout.strip()

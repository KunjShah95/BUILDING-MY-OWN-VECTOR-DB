"""
Plugin Registry — versioned manifest store for index/encoder/storage plugins.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

VALID_PLUGIN_TYPES = {"index", "encoder", "storage"}
_REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins")
_REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "registry.json")


def _parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse 'X.Y.Z' into (X, Y, Z) integers. Raises ValueError on bad format."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver '{version}': expected X.Y.Z")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        raise ValueError(f"Invalid semver '{version}': components must be integers")


@dataclass
class PluginManifest:
    name: str
    version: str                      # semver X.Y.Z
    plugin_type: str                  # index | encoder | storage
    entry_point: str                  # dotted path or file path
    description: str = ""
    author: str = ""
    requires: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.plugin_type not in VALID_PLUGIN_TYPES:
            raise ValueError(
                f"plugin_type must be one of {VALID_PLUGIN_TYPES}, got '{self.plugin_type}'"
            )
        _parse_semver(self.version)  # validate on creation

    @property
    def semver(self) -> Tuple[int, int, int]:
        return _parse_semver(self.version)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        return cls(**data)


class PluginRegistry:
    """
    In-process registry with disk persistence.
    Key: (name, version) -> (manifest, plugin_class)
    """

    def __init__(self, registry_file: str = _REGISTRY_FILE) -> None:
        self._registry_file = registry_file
        # {name: {version: (PluginManifest, type|None)}}
        self._store: Dict[str, Dict[str, Tuple[PluginManifest, Optional[Type]]]] = {}
        self._load_from_disk()

    # ── public API ────────────────────────────────────────────────────────────

    def register(self, manifest: PluginManifest, plugin_class: Type) -> None:
        """Validate and store a plugin."""
        if not isinstance(manifest, PluginManifest):
            raise TypeError("manifest must be a PluginManifest instance")
        if manifest.name not in self._store:
            self._store[manifest.name] = {}
        self._store[manifest.name][manifest.version] = (manifest, plugin_class)
        logger.info(
            "Registered plugin '%s' v%s (%s)",
            manifest.name, manifest.version, manifest.plugin_type,
        )
        self._persist()

    def unregister(self, name: str, version: str) -> bool:
        """Remove a specific version. Returns True if it existed."""
        versions = self._store.get(name, {})
        if version not in versions:
            return False
        del versions[version]
        if not versions:
            del self._store[name]
        logger.info("Unregistered plugin '%s' v%s", name, version)
        self._persist()
        return True

    def get(
        self, name: str, version: Optional[str] = None
    ) -> Tuple[PluginManifest, Optional[Type]]:
        """
        Return (manifest, class). If version is None, returns the highest semver.
        Raises KeyError if not found.
        """
        versions = self._store.get(name)
        if not versions:
            raise KeyError(f"Plugin '{name}' not found")
        if version is not None:
            if version not in versions:
                raise KeyError(f"Plugin '{name}' v{version} not found")
            return versions[version]
        # latest by semver
        latest = max(versions.keys(), key=lambda v: _parse_semver(v))
        return versions[latest]

    def list_plugins(
        self, plugin_type: Optional[str] = None
    ) -> List[PluginManifest]:
        """Return all manifests, optionally filtered by type."""
        result: List[PluginManifest] = []
        for versions in self._store.values():
            for manifest, _ in versions.values():
                if plugin_type is None or manifest.plugin_type == plugin_type:
                    result.append(manifest)
        return result

    # ── persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self._registry_file), exist_ok=True)
        data: Dict[str, Any] = {}
        for name, versions in self._store.items():
            data[name] = {}
            for ver, (manifest, _) in versions.items():
                data[name][ver] = manifest.to_dict()
        try:
            with open(self._registry_file, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Could not persist registry: %s", exc)

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._registry_file):
            return
        try:
            with open(self._registry_file) as f:
                data = json.load(f)
            for name, versions in data.items():
                self._store[name] = {}
                for ver, manifest_dict in versions.items():
                    manifest = PluginManifest.from_dict(manifest_dict)
                    self._store[name][ver] = (manifest, None)  # class not persisted
        except Exception as exc:
            logger.warning("Could not load registry from disk: %s", exc)


# Module-level singleton
_global_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    return _global_registry

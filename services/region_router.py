"""
Region-aware query router for multi-region active-active deployments.

Picks the nearest healthy region by haversine distance.
Marks regions unhealthy after 3 consecutive failures.
Config via REGIONS env var (JSON list of {name, lat, lon, endpoint, healthy}).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_FAILURES = 3
_HEALTH_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Region:
    name: str
    lat: float
    lon: float
    endpoint: str
    healthy: bool = True
    failure_count: int = 0
    last_check: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# RegionRouter
# ---------------------------------------------------------------------------

class RegionRouter:
    """Routes queries to the nearest healthy region."""

    def __init__(self, regions: Optional[List[Region]] = None) -> None:
        self._regions: Dict[str, Region] = {}

        if regions:
            for r in regions:
                self._regions[r.name] = r

        env_regions = os.environ.get("REGIONS")
        if env_regions:
            try:
                for item in json.loads(env_regions):
                    r = Region(
                        name=item["name"],
                        lat=float(item["lat"]),
                        lon=float(item["lon"]),
                        endpoint=item["endpoint"],
                        healthy=bool(item.get("healthy", True)),
                    )
                    self._regions[r.name] = r
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Failed to parse REGIONS env var: %s", exc)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_region(
        self, name: str, lat: float, lon: float, endpoint: str, healthy: bool = True
    ) -> None:
        self._regions[name] = Region(name=name, lat=lat, lon=lon, endpoint=endpoint, healthy=healthy)

    def deregister_region(self, name: str) -> bool:
        return self._regions.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        client_lat: float,
        client_lon: float,
        query_type: str = "read",  # reserved for future policy
    ) -> Optional[str]:
        """Return endpoint of nearest healthy region, or None."""
        region = self.route_with_region(client_lat, client_lon, query_type)
        return region.endpoint if region else None

    def route_with_region(
        self,
        client_lat: float,
        client_lon: float,
        query_type: str = "read",
    ) -> Optional[Region]:
        if not self._regions:
            return None
        ranked = sorted(
            self._regions.values(),
            key=lambda r: _haversine_km(client_lat, client_lon, r.lat, r.lon),
        )
        for region in ranked:
            if region.healthy:
                return region
        return None

    # ------------------------------------------------------------------
    # Health checks (sync version; async optional via httpx)
    # ------------------------------------------------------------------

    def _ping_sync(self, url: str) -> bool:
        """Synchronous health ping using urllib (no extra deps)."""
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_S)
            return req.status < 500
        except Exception:
            return False

    def check_health_sync(self, region_name: str) -> bool:
        region = self._regions.get(region_name)
        if region is None:
            return False
        url = region.endpoint.rstrip("/") + "/health"
        ok = self._ping_sync(url)
        region.last_check = time.time()
        if ok:
            region.failure_count = 0
            region.healthy = True
        else:
            region.failure_count += 1
            if region.failure_count >= _MAX_FAILURES:
                region.healthy = False
                logger.error("Region %s marked unhealthy after %d failures", region_name, _MAX_FAILURES)
        return ok

    async def check_health(self, region_name: str) -> bool:
        """Async health check; falls back to sync if httpx unavailable."""
        region = self._regions.get(region_name)
        if region is None:
            return False
        url = region.endpoint.rstrip("/") + "/health"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT_S) as client:
                resp = await client.get(url)
            ok = resp.status_code < 500
        except ImportError:
            ok = self._ping_sync(url)
        except Exception:
            ok = False

        region.last_check = time.time()
        if ok:
            region.failure_count = 0
            region.healthy = True
        else:
            region.failure_count += 1
            if region.failure_count >= _MAX_FAILURES:
                region.healthy = False
                logger.error("Region %s marked unhealthy after %d failures", region_name, _MAX_FAILURES)
        return ok

    async def check_all_health(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for name in list(self._regions):
            results[name] = await self.check_health(name)
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_regions(self) -> List[Dict]:
        return [
            {
                "name": r.name,
                "lat": r.lat,
                "lon": r.lon,
                "endpoint": r.endpoint,
                "healthy": r.healthy,
                "failure_count": r.failure_count,
                "last_check": r.last_check,
            }
            for r in self._regions.values()
        ]

    def mark_healthy(self, name: str, healthy: bool = True) -> None:
        region = self._regions.get(name)
        if region:
            region.healthy = healthy
            if healthy:
                region.failure_count = 0

    def get_region(self, name: str) -> Optional[Region]:
        return self._regions.get(name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router_instance: Optional[RegionRouter] = None


def get_region_router() -> RegionRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = RegionRouter()
    return _router_instance

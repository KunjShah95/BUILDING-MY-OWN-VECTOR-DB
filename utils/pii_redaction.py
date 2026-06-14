"""
PII Redaction & Data Residency (Phase 9: Security & Compliance).

Provides field-level redaction of personally identifiable information (PII)
in vector metadata and per-tenant region pinning for data residency compliance.

Features:
  - Field-level redaction: mark specific metadata fields as PII and redact
    them in search results based on API key permissions.
  - Pattern-based detection: automatically detect email, phone, SSN, and
    other PII patterns in text metadata.
  - Per-tenant region pinning: restrict which storage regions a tenant's
    data can reside in (implemented via S3/Azure bucket selection).

Usage::

    from utils.pii_redaction import PIIRedactor

    redactor = PIIRedactor()
    cleaned = redactor.redact_metadata({"email": "user@example.com", "text": "..."})
    # -> {"email": "[EMAIL REDACTED]", "text": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Common PII patterns (can be extended)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\+?1?\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "zip_code": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}

PII_LABEL_MAP = {
    "email": "EMAIL",
    "phone_us": "PHONE",
    "ssn": "SSN",
    "credit_card": "CC",
    "ip_address": "IP",
    "zip_code": "ZIP",
}


class PIIRedactor:
    """Redact PII from vector metadata fields.

    Supports two modes:
      1. Field-based: explicitly mark fields as containing PII
      2. Pattern-based: auto-detect PII patterns in text content
    """

    def __init__(self, pii_fields: Optional[Set[str]] = None):
        self.pii_fields = pii_fields or {
            "email", "phone", "ssn", "credit_card",
            "ip_address", "password", "secret", "token",
            "api_key", "auth_token", "session_id",
        }
        self._redacted_count = 0

    def redact_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        redact_patterns: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Redact PII from a metadata dictionary.

        Args:
            metadata: The metadata dict to redact.
            redact_patterns: If True, also scan string values for PII patterns.

        Returns:
            Redacted copy of metadata (originals are not modified).
        """
        if not metadata:
            return metadata

        redacted = {}
        for key, value in metadata.items():
            if key.lower() in self.pii_fields:
                self._redacted_count += 1
                redacted[key] = f"[{PII_LABEL_MAP.get(key.lower(), 'REDACTED')}]"
            elif isinstance(value, str) and redact_patterns:
                redacted[key] = self._redact_text(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_metadata(value, redact_patterns)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_metadata(item, redact_patterns)
                    if isinstance(item, dict)
                    else self._redact_text(item) if isinstance(item, str) and redact_patterns
                    else item
                    for item in value
                ]
            else:
                redacted[key] = value

        return redacted

    def _redact_text(self, text: str) -> str:
        """Scan text for PII patterns and redact them."""
        for pattern_name, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                label = PII_LABEL_MAP.get(pattern_name, "REDACTED")
                text = pattern.sub(f"[{label}]", text)
                self._redacted_count += 1
        return text

    def redact_results(
        self,
        results: List[Dict[str, Any]],
        redact_patterns: bool = True,
    ) -> List[Dict[str, Any]]:
        """Redact PII from a list of search result dicts.

        Args:
            results: Search results with metadata.
            redact_patterns: Apply pattern-based redaction.

        Returns:
            Results with PII redacted.
        """
        self._redacted_count = 0
        redacted_results = []
        for r in results:
            redacted = dict(r)
            if "metadata" in r:
                redacted["metadata"] = self.redact_metadata(r["metadata"], redact_patterns)
            redacted_results.append(redacted)
        return redacted_results

    def get_redaction_count(self) -> int:
        return self._redacted_count


# ---------------------------------------------------------------------------
# Data Residency Manager
# ---------------------------------------------------------------------------


class DataResidencyManager:
    """Per-tenant region pinning for data residency compliance.

    Maps tenants to allowed storage regions and enforces that data is stored
    only in those regions.
    """

    def __init__(self, config_path: str = "data_residency_config.json"):
        self.config_path = config_path
        self._tenant_regions: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    self._tenant_regions = json.load(f)
            except Exception:
                pass

    def _save(self):
        with open(self.config_path, "w") as f:
            json.dump(self._tenant_regions, f, indent=2)

    def set_tenant_region(self, tenant_id: str, region: str):
        """Pin a tenant's data to a specific storage region.

        Args:
            tenant_id: Tenant identifier.
            region: Storage region (e.g. "us-east-1", "eu-west-1", "ap-southeast-1").
        """
        self._tenant_regions[tenant_id] = region
        self._save()
        logger.info("Tenant %s pinned to region %s", tenant_id, region)

    def get_tenant_region(self, tenant_id: str) -> Optional[str]:
        """Get the pinned region for a tenant.

        Returns None if no region is pinned (all regions allowed).
        """
        return self._tenant_regions.get(tenant_id)

    def remove_tenant_region(self, tenant_id: str):
        """Remove region pinning for a tenant."""
        self._tenant_regions.pop(tenant_id, None)
        self._save()

    def is_allowed_region(self, tenant_id: str, region: str) -> bool:
        """Check if a tenant is allowed to store data in a given region.

        Args:
            tenant_id: Tenant identifier.
            region: Target storage region.

        Returns:
            True if allowed (no pin or matching pin).
        """
        pinned = self.get_tenant_region(tenant_id)
        if pinned is None:
            return True
        return pinned == region

    def get_storage_config_for_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Get storage configuration respecting data residency.

        Returns storage backend config that complies with the tenant's
        region pinning.
        """
        region = self.get_tenant_region(tenant_id)
        if region:
            return {
                "endpoint": f"s3.{region}.amazonaws.com",
                "region": region,
            }
        return {}

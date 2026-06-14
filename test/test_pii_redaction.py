"""Tests for PII Redaction & Data Residency (Phase 9)."""

import os
import json
import pytest

from utils.pii_redaction import PIIRedactor, DataResidencyManager


class TestPIIRedactorInit:
    def test_default_pii_fields(self):
        r = PIIRedactor()
        assert "email" in r.pii_fields
        assert "ssn" in r.pii_fields
        assert "password" in r.pii_fields

    def test_custom_pii_fields(self):
        r = PIIRedactor(pii_fields={"secret", "token"})
        assert "secret" in r.pii_fields
        assert "email" not in r.pii_fields


class TestPIIRedactFieldBased:
    def test_redact_email_field(self):
        r = PIIRedactor()
        result = r.redact_metadata({"email": "user@example.com", "name": "John"})
        assert result["email"] == "[EMAIL]"
        assert result["name"] == "John"

    def test_redact_multiple_fields(self):
        r = PIIRedactor()
        result = r.redact_metadata({
            "email": "a@b.com",
            "phone": "555-1234",
            "ssn": "123-45-6789",
            "name": "John",
        })
        assert result["email"] == "[EMAIL]"
        assert result["phone"] == "[REDACTED]"  # 'phone' not in PII_LABEL_MAP, falls back to REDACTED
        assert result["ssn"] == "[SSN]"
        assert result["name"] == "John"

    def test_redact_case_insensitive(self):
        r = PIIRedactor()
        result = r.redact_metadata({"Email": "a@b.com", "PASSWORD": "secret123"})
        assert "[EMAIL]" in result["Email"]
        assert "[REDACTED]" in result["PASSWORD"]

    def test_no_pii_unchanged(self):
        r = PIIRedactor()
        result = r.redact_metadata({"name": "Alice", "age": 30, "city": "NYC"})
        assert result == {"name": "Alice", "age": 30, "city": "NYC"}

    def test_none_metadata(self):
        r = PIIRedactor()
        assert r.redact_metadata(None) is None

    def test_empty_metadata(self):
        r = PIIRedactor()
        assert r.redact_metadata({}) == {}


class TestPIIRedactPatternBased:
    def test_redact_email_in_text(self):
        r = PIIRedactor()
        result = r.redact_metadata({"bio": "Contact me at john@example.com for info"})
        assert "[EMAIL]" in result["bio"]
        assert "john@example.com" not in result["bio"]

    def test_redact_phone_in_text(self):
        r = PIIRedactor()
        result = r.redact_metadata({"text": "Call 555-123-4567 now!"})
        assert "[PHONE]" in result["text"]

    def test_redact_ssn_in_text(self):
        r = PIIRedactor()
        result = r.redact_metadata({"note": "SSN is 123-45-6789"})
        assert "[SSN]" in result["note"]

    def test_redact_ip_in_text(self):
        r = PIIRedactor()
        result = r.redact_metadata({"log": "from 192.168.1.1"})
        assert "[IP]" in result["log"]

    def test_redact_credit_card_in_text(self):
        r = PIIRedactor()
        result = r.redact_metadata({"payment": "4111-1111-1111-1111"})
        assert "[CC]" in result["payment"]

    def test_redact_patterns_disabled(self):
        r = PIIRedactor()
        result = r.redact_metadata(
            {"bio": "Email me@example.com"},
            redact_patterns=False,
        )
        assert "me@example.com" in result["bio"]


class TestPIIRedactNested:
    def test_redact_nested_dict(self):
        r = PIIRedactor()
        result = r.redact_metadata({
            "user": {"email": "a@b.com", "profile": {"ssn": "123-45-6789"}},
        })
        assert result["user"]["email"] == "[EMAIL]"
        assert result["user"]["profile"]["ssn"] == "[SSN]"

    def test_redact_list_of_dicts(self):
        r = PIIRedactor()
        result = r.redact_metadata({
            "users": [
                {"email": "a@b.com"},
                {"email": "c@d.com"},
            ],
        })
        assert result["users"][0]["email"] == "[EMAIL]"
        assert result["users"][1]["email"] == "[EMAIL]"    


class TestPIIRedactResults:
    def test_redact_results_list(self):
        r = PIIRedactor()
        results = [
            {"vector_id": "v1", "metadata": {"email": "a@b.com", "score": 0.9}},
            {"vector_id": "v2", "metadata": {"text": "Call me"}, "distance": 0.5},
        ]
        redacted = r.redact_results(results)
        assert redacted[0]["metadata"]["email"] == "[EMAIL]"
        assert redacted[0]["metadata"]["score"] == 0.9
        assert redacted[1]["vector_id"] == "v2"

    def test_redact_results_preserves_structure(self):
        r = PIIRedactor()
        results = [{"vector_id": "v1", "metadata": None}]
        redacted = r.redact_results(results)
        assert redacted[0]["metadata"] is None

    def test_redaction_count(self):
        r = PIIRedactor()
        r.redact_metadata({"email": "a@b.com", "phone": "555-1234"})
        assert r.get_redaction_count() == 2


class TestDataResidencyManager:
    def test_set_tenant_region(self, tmp_path):
        config = str(tmp_path / "residency.json")
        mgr = DataResidencyManager(config_path=config)
        mgr.set_tenant_region("acme", "us-east-1")
        assert mgr.get_tenant_region("acme") == "us-east-1"

    def test_get_tenant_region_default_none(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        assert mgr.get_tenant_region("unknown") is None

    def test_is_allowed_region_no_pin(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        assert mgr.is_allowed_region("any", "eu-west-1") is True

    def test_is_allowed_region_pinned(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        mgr.set_tenant_region("acme", "us-east-1")
        assert mgr.is_allowed_region("acme", "us-east-1") is True
        assert mgr.is_allowed_region("acme", "eu-west-1") is False

    def test_remove_tenant_region(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        mgr.set_tenant_region("acme", "us-east-1")
        mgr.remove_tenant_region("acme")
        assert mgr.get_tenant_region("acme") is None

    def test_persistence_across_instances(self, tmp_path):
        config = str(tmp_path / "residency.json")
        mgr1 = DataResidencyManager(config_path=config)
        mgr1.set_tenant_region("acme", "us-west-2")

        mgr2 = DataResidencyManager(config_path=config)
        assert mgr2.get_tenant_region("acme") == "us-west-2"

    def test_storage_config_for_pinned_tenant(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        mgr.set_tenant_region("acme", "eu-central-1")
        cfg = mgr.get_storage_config_for_tenant("acme")
        assert cfg["region"] == "eu-central-1"
        assert "s3.eu-central-1" in cfg["endpoint"]

    def test_storage_config_no_pin(self, tmp_path):
        mgr = DataResidencyManager(config_path=str(tmp_path / "r.json"))
        assert mgr.get_storage_config_for_tenant("unknown") == {}

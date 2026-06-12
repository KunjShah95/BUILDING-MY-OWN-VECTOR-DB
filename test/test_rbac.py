"""Tests for row-level RBAC policy."""

import pytest

from utils.rbac import RBACPolicy, Permission, AccessDenied, policy_from_validation


def test_permission_grants():
    assert RBACPolicy(permissions="read").allows(Permission.READ)
    assert not RBACPolicy(permissions="read").allows(Permission.WRITE)
    assert RBACPolicy(permissions="write").allows(Permission.WRITE)
    assert RBACPolicy(permissions="read_write").allows(Permission.DELETE)
    assert RBACPolicy(permissions="admin").allows(Permission.ADMIN)


def test_require_raises_when_denied():
    p = RBACPolicy(permissions="read")
    p.require(Permission.READ)  # no raise
    with pytest.raises(AccessDenied):
        p.require(Permission.WRITE)


def test_collection_scope_enforced():
    p = RBACPolicy(collection_id="c1")
    p.require_collection("c1")
    with pytest.raises(AccessDenied):
        p.require_collection("c2")


def test_collection_scope_none_allows_any():
    p = RBACPolicy(collection_id=None)
    p.require_collection("anything")  # no raise


def test_row_filter_allows_and_blocks():
    p = RBACPolicy(row_filter_dsl="department = 'eng' AND clearance < 3")
    assert p.row_allowed({"department": "eng", "clearance": 1})
    assert not p.row_allowed({"department": "sales", "clearance": 1})
    assert not p.row_allowed({"department": "eng", "clearance": 5})


def test_no_row_filter_allows_all():
    p = RBACPolicy()
    assert p.row_allowed({"anything": 1})
    assert p.row_allowed(None)


def test_filter_results():
    p = RBACPolicy(row_filter_dsl="dept = 'eng'")
    rows = [
        {"vector_id": "a", "metadata": {"dept": "eng"}},
        {"vector_id": "b", "metadata": {"dept": "sales"}},
        {"vector_id": "c", "metadata": {"dept": "eng"}},
    ]
    kept = [r["vector_id"] for r in p.filter_results(rows)]
    assert kept == ["a", "c"]


def test_metadata_filter_callable_for_search():
    p = RBACPolicy(row_filter_dsl="dept = 'eng'")
    f = p.metadata_filter()
    assert f({"dept": "eng"}) is True
    assert f({"dept": "sales"}) is False
    assert RBACPolicy().metadata_filter() is None


def test_policy_from_validation():
    v = {"tenant_id": "t1", "collection_id": "c1", "permissions": "write", "row_filter": "x = 1"}
    p = policy_from_validation(v)
    assert p.tenant_id == "t1"
    assert p.allows(Permission.WRITE)
    assert p.row_allowed({"x": 1})
    assert not p.row_allowed({"x": 2})

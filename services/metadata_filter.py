"""Advanced JSONB metadata filtering for vector queries."""
from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class MetadataFilter:
    """
    Builds SQLAlchemy filter conditions for JSONB metadata queries.

    Supports operators:
    - eq: equals (default)
    - neq: not equals
    - gt: greater than
    - gte: greater than or equal
    - lt: less than
    - lte: less than or equal
    - in: value in list
    - nin: value not in list
    - exists: key exists
    - contains: text contains (for string values)
    - regex: regex match on string values

    Example filter dict:
    {
        "price": {"gt": 100, "lt": 500},
        "category": {"in": ["electronics", "books"]},
        "rating": {"gte": 4.0},
        "tags": {"contains": "ai"},
    }
    """

    OPERATORS = {
        "eq": lambda col, key, val: col[key].as_string() == str(val),
        "neq": lambda col, key, val: col[key].as_string() != str(val),
        "gt": lambda col, key, val: col[key].as_float() > float(val),
        "gte": lambda col, key, val: col[key].as_float() >= float(val),
        "lt": lambda col, key, val: col[key].as_float() < float(val),
        "lte": lambda col, key, val: col[key].as_float() <= float(val),
        "in": lambda col, key, val: col[key].as_string().in_([str(v) for v in val]),
        "nin": lambda col, key, val: ~col[key].as_string().in_([str(v) for v in val]),
        "exists": lambda col, key, val: col[key].isnot(None) if val else col[key].is_(None),
        "contains": lambda col, key, val: col[key].as_string().contains(str(val)),
        "regex": lambda col, key, val: col[key].as_string().regexp_match(str(val)),
    # JSONB path operators
    "contains_key": lambda col, key, val: col[key].isnot(None),
    "type": lambda col, key, val: col[key].type == sa.cast(val, String),
}

    @staticmethod
    def build_filters(model_class, filters: Optional[Dict[str, Any]]):
        """
        Build SQLAlchemy filter conditions from a filter dict.

        Simple format: {"field": "value"} (implies eq)
        Advanced format: {"field": {"op": value}}
        """
        from sqlalchemy import cast, String, Text
        import sqlalchemy as sa

        if not filters:
            return []

        conditions = []
        meta_col = model_class.meta_data

        for key, value in filters.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op in MetadataFilter.OPERATORS:
                        try:
                            condition = MetadataFilter.OPERATORS[op](meta_col, key, op_value)
                            conditions.append(condition)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Filter error for {key} {op}: {e}")
            else:
                conditions.append(meta_col[key].as_string() == str(value))

        return conditions

    @staticmethod
    def pre_filter(
        db_session,
        model_class,
        filters: Optional[Dict[str, Any]],
        base_query=None,
    ):
        """
        Apply metadata filters as SQL WHERE clauses (pre-filtering).

        This filters vectors at the database level BEFORE ANN search,
        reducing the number of vectors that need to be searched in-memory.

        Args:
            db_session: SQLAlchemy session
            model_class: SQLAlchemy model class (e.g. Vector)
            filters: Filter dict (same format as ``post_filter``)
            base_query: Optional base query to chain from

        Returns:
            SQLAlchemy query with filter conditions applied, plus the filtered
            vector IDs if simple eq filters are used (for ANN fallback).
        """
        if not filters:
            return base_query if base_query is not None else db_session.query(model_class), None

        query = base_query if base_query is not None else db_session.query(model_class)
        conditions = MetadataFilter.build_filters(model_class, filters)
        if conditions:
            query = query.filter(*conditions)

        return query, None

    @staticmethod
    def post_filter(results: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-filter results in memory (when DB-level filtering isn't available).
        Used for in-memory index searches (HNSW, IVF, brute force).
        """
        if not filters:
            return results

        filtered = []
        for r in results:
            meta = r.get("metadata") or r.get("meta_data") or {}
            if MetadataFilter._matches(meta, filters):
                filtered.append(r)

        return filtered

    @staticmethod
    def _matches(metadata: Dict, filters: Dict) -> bool:
        for key, condition in filters.items():
            if isinstance(condition, dict):
                for op, val in condition.items():
                    if op == "exists":
                        if (key in metadata) != bool(val):
                            return False
                    elif op == "eq":
                        if not _check_eq(metadata.get(key), val):
                            return False
                    elif op == "neq":
                        if _check_eq(metadata.get(key), val):
                            return False
                    elif op in ("gt", "gte", "lt", "lte"):
                        meta_val = metadata.get(key)
                        if meta_val is None:
                            return False
                        try:
                            meta_f = float(meta_val)
                            cond_f = float(val)
                            if op == "gt" and not (meta_f > cond_f): return False
                            if op == "gte" and not (meta_f >= cond_f): return False
                            if op == "lt" and not (meta_f < cond_f): return False
                            if op == "lte" and not (meta_f <= cond_f): return False
                        except (ValueError, TypeError):
                            return False
                    elif op == "in":
                        if metadata.get(key) not in val:
                            return False
                    elif op == "nin":
                        if metadata.get(key) in val:
                            return False
                    elif op == "contains":
                        meta_str = str(metadata.get(key, ""))
                        if str(val) not in meta_str:
                            return False
                    elif op == "regex":
                        meta_str = str(metadata.get(key, ""))
                        if not re.search(str(val), meta_str):
                            return False
            else:
                if not _check_eq(metadata.get(key), condition):
                    return False
        return True


def _check_eq(a, b) -> bool:
    try:
        return a == b
    except Exception:
        return str(a) == str(b)

"""Billing and usage metering service for managed cloud platform (Phase 16)."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pricing per unit
PRICING = {
    "search": 0.000001,       # $0.000001 per query
    "insert": 0.000002,       # $0.000002 per vector
    "index_build": 0.01,      # $0.01 per 1k vectors
}


@dataclass
class UsageRecord:
    tenant_id: str
    operation: str            # search | insert | index_build
    count: int
    vector_dimensions: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LineItem:
    operation: str
    count: int
    unit_price: float
    subtotal: float


@dataclass
class Bill:
    tenant_id: str
    period_start: str
    period_end: str
    line_items: List[LineItem]
    subtotal: float
    total_usd: float


class BillingService:
    """Records usage events and generates bills."""

    def __init__(self, data_dir: str = "billing_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = self.data_dir / "billing_events.jsonl"

    # ---- Write ----------------------------------------------------------------

    def record_usage(
        self,
        tenant_id: str,
        operation: str,
        count: int,
        dimensions: int = 0,
    ) -> UsageRecord:
        """Append a usage event to billing_events.jsonl."""
        record = UsageRecord(
            tenant_id=tenant_id,
            operation=operation,
            count=count,
            vector_dimensions=dimensions,
        )
        with open(self._events_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        return record

    # ---- Read -----------------------------------------------------------------

    def _load_events(
        self,
        tenant_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        if not self._events_file.exists():
            return []
        events: List[Dict[str, Any]] = []
        with open(self._events_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("tenant_id") != tenant_id:
                    continue
                ts = datetime.fromisoformat(ev.get("timestamp", "2000-01-01"))
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                events.append(ev)
        return events

    # ---- Billing --------------------------------------------------------------

    def compute_bill(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Bill:
        """Compute invoice for a tenant over a date range."""
        events = self._load_events(tenant_id, start_date, end_date)

        totals: Dict[str, int] = {}
        for ev in events:
            op = ev.get("operation", "unknown")
            totals[op] = totals.get(op, 0) + ev.get("count", 0)

        line_items: List[LineItem] = []
        for op, count in totals.items():
            unit_price = PRICING.get(op, 0.0)
            if op == "index_build":
                subtotal = (count / 1000) * unit_price
            else:
                subtotal = count * unit_price
            line_items.append(LineItem(
                operation=op,
                count=count,
                unit_price=unit_price,
                subtotal=round(subtotal, 8),
            ))

        subtotal = sum(li.subtotal for li in line_items)
        return Bill(
            tenant_id=tenant_id,
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            line_items=line_items,
            subtotal=round(subtotal, 8),
            total_usd=round(subtotal, 8),
        )

    # ---- Export ---------------------------------------------------------------

    def export_csv(self, tenant_id: str, month: str) -> str:
        """Return CSV string for a billing month (format: YYYY-MM)."""
        try:
            year, mon = int(month[:4]), int(month[5:7])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid month format: {month!r}. Expected YYYY-MM.")
        start = datetime(year, mon, 1)
        if mon == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, mon + 1, 1) - timedelta(seconds=1)

        events = self._load_events(tenant_id, start, end)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["tenant_id", "operation", "count", "vector_dimensions", "timestamp"],
        )
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)
        return buf.getvalue()

    # ---- Summary --------------------------------------------------------------

    def get_usage_summary(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Return aggregated counts by operation for the last N days."""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        events = self._load_events(tenant_id, start, end)

        totals: Dict[str, int] = {}
        for ev in events:
            op = ev.get("operation", "unknown")
            totals[op] = totals.get(op, 0) + ev.get("count", 0)

        return {
            "tenant_id": tenant_id,
            "period_days": days,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "operations": totals,
            "total_events": len(events),
        }
